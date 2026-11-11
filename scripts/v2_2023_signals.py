"""Original top-two group slots with an independent raw-price oracle."""

from datetime import date
from math import fsum, isfinite

from beartype import beartype

from c10standard.audit import equal
from c10standard.models import TargetRecord
from rotbt.models import Dataset
from rotbt.phase2.models import FeatureStore
from scripts.v2_2023_models import Contract, Pool, Prepared


def features_for(datasets: list[Dataset]) -> FeatureStore:
    return FeatureStore(frames={d.symbol: d.frame.assign(
        mom252=d.frame.tr.pct_change(252, fill_method=None), raw_close=d.frame.close,
        turnover20=d.frame.turnover.rolling(20).mean(),
        positive20=d.frame.volume.gt(0).rolling(20).sum()) for d in datasets})


def feed_names(contract: Contract, prepared: Prepared) -> tuple[str, ...]:
    Pool.model_validate(prepared.pool)
    role = "NEW_POOL" if contract.role == "NEW_POOL" else "OLD_POOL"
    if contract.pool_sha256 != prepared.pool.sha256 or prepared.pool.role != role:
        raise ValueError("contract does not bind prepared pool")
    if contract.role.endswith("_BH"):
        return tuple(sorted((contract.role.removesuffix("_BH"), "SHY")))
    return tuple(sorted((*prepared.pool.risk_names, "SHY")))


def group_for(prepared: Prepared, symbol: str) -> str:
    return next(group for group, members in prepared.pool.groups.groups.items()
                if symbol in members)


@beartype
def target(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
    names = feed_names(contract, prepared)
    if contract.role.endswith("_BH"):
        return TargetRecord(day=day, q={contract.role.removesuffix("_BH"): 1.}, reason="oracle")
    scores: dict[str, float] = {}
    for symbol in names:
        frame = prepared.features.frames[symbol].loc[:str(day)]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("missing exact prior signal close")
        if len(frame) < 253:
            if symbol == "SHY":
                raise ValueError("insufficient 253-price cash warmup")
            continue
        row = frame.iloc[-1]
        values = [float(row[k]) for k in ("mom252", "raw_close", "turnover20", "positive20")]
        if not all(isfinite(v) for v in values):
            raise ValueError("nonfinite signal input")
        if symbol == "SHY" or (values[1] >= 5 and values[2] >= 1e7 and values[3] == 20):
            scores[symbol] = values[0]
    selected: list[str] = []
    used: set[str] = set()
    for symbol in sorted((s for s in scores if s != "SHY"), key=lambda s: (-scores[s], s)):
        group = group_for(prepared, symbol)
        if group in used:
            continue
        selected.append(symbol)
        used.add(group)
        if len(selected) == 2:
            break
    q = {s: .45 for s in selected if scores[s] > scores["SHY"]}
    q["SHY"] = max(0., 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")


@beartype
def oracle(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
    feed_names(contract, prepared)
    if contract.role.endswith("_BH"):
        return TargetRecord(day=day, q={contract.role.removesuffix("_BH"): 1.}, reason="oracle")
    frames = {d.symbol: d.frame.loc[:str(day)] for d in prepared.datasets}
    shy = frames["SHY"]
    if len(shy) < 253 or shy.index[-1].date() != day:
        raise ValueError("raw cash warmup/calendar requires 253 prices")
    cash = float(shy.tr.iloc[-1] / shy.tr.iloc[-253] - 1)
    if not isfinite(cash):
        raise ValueError("nonfinite raw cash return")
    scores: list[tuple[float, str, str]] = []
    for symbol in prepared.pool.risk_names:
        frame = frames[symbol]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("raw exact signal close missing")
        if len(frame) < 253:
            continue
        recent = frame.iloc[-20:]
        score = float(frame.tr.iloc[-1] / frame.tr.iloc[-253] - 1)
        if not isfinite(score):
            raise ValueError("nonfinite raw return")
        if (fsum(float(v) for v in recent.turnover) / 20 >= 1e7
                and all(float(v) > 0 for v in recent.volume) and float(frame.close.iloc[-1]) >= 5):
            scores.append((score, symbol, group_for(prepared, symbol)))
    picks: list[tuple[float, str]] = []
    occupied: set[str] = set()
    for score, symbol, group in sorted(scores, key=lambda x: (-x[0], x[1])):
        if group in occupied:
            continue
        picks.append((score, symbol))
        occupied.add(group)
        if len(picks) == 2:
            break
    q = {symbol: .9 / 2 for score, symbol in picks if score > cash}
    q["SHY"] = max(0., 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")


def verify_targets(contract: Contract, prepared: Prepared, days: tuple[date, ...]) -> None:
    for day in days:
        prefix = [Dataset(symbol=d.symbol, frame=d.frame.loc[:str(day)].copy(),
                          dividends=[e for e in d.dividends if e.ex_date <= day],
                          splits=[s for s in d.splits if s.day <= day]) for d in prepared.datasets]
        bounded = Prepared(pool=prepared.pool, datasets=prefix, features=features_for(prefix),
                           fees=prepared.fees)
        full = target(contract, prepared, day)
        for expected in (target(contract, bounded, day), oracle(contract, bounded, day)):
            if set(full.q) != set(expected.q):
                raise ValueError("full/prefix/raw target membership differs")
            for symbol, weight in full.q.items():
                equal(weight, expected.q[symbol], "full/prefix/raw target weight", 1e-12)
