"""Frozen group selection and an independent raw total-return oracle."""

from datetime import date
from math import fsum, isfinite

from beartype import beartype

from .models import Contract, Prepared, TargetRecord


def feed_names(contract: Contract, names: tuple[str, ...]) -> tuple[str, ...]:
    if contract.rule.endswith("_BH"):
        return tuple(sorted((contract.rule.removesuffix("_BH"), "SHY")))
    return (
        ("IEF", "SHY", "SPY") if contract.rule == "SPY60_IEF40" else tuple(sorted((*names, "SHY")))
    )


def group_for(prepared: Prepared, symbol: str) -> str:
    for group, members in prepared.strategy.groups.items():
        if symbol in members:
            return group
    raise ValueError(f"missing risk group: {symbol}")


def fixed_target(contract: Contract, day: date) -> TargetRecord | None:
    if contract.rule.endswith("_BH"):
        return TargetRecord(day=day, q={contract.rule.removesuffix("_BH"): 1.0}, reason="oracle")
    if contract.rule == "SPY60_IEF40":
        return TargetRecord(day=day, q={"SPY": 0.6, "IEF": 0.4}, reason="oracle")
    return None


@beartype
def target(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
    fixed = fixed_target(contract, day)
    if fixed is not None:
        return fixed
    top = {"C10": 1, "C11": 2, "C00": 3}[contract.rule]
    scores: dict[str, float] = {}
    for symbol in (*prepared.risk_names, "SHY"):
        if symbol != "SHY":
            group_for(prepared, symbol)
        frame = prepared.features.frames[symbol].loc[: str(day)]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("missing exact prior signal close")
        if len(frame) < 253:
            if symbol == "SHY":
                raise ValueError("insufficient cash momentum warmup")
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
        if len(selected) == top:
            break
    q = {s: 0.9 / top for s in selected if scores[s] > scores["SHY"]}
    q["SHY"] = max(0.0, 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")


@beartype
def oracle(contract: Contract, prepared: Prepared, day: date) -> TargetRecord:
    """Rebuild qualification and ranking from raw datasets, never feature targets."""
    fixed = fixed_target(contract, day)
    if fixed is not None:
        return fixed
    top = {"C10": 1, "C11": 2, "C00": 3}[contract.rule]
    frames = {d.symbol: d.frame.loc[: str(day)] for d in prepared.datasets}
    shy = frames["SHY"]
    if len(shy) < 253 or shy.index[-1].date() != day:
        raise ValueError("raw cash warmup/calendar")
    cash = float(shy.tr.iloc[-1] / shy.tr.iloc[-253] - 1)
    scores: list[tuple[float, str, str]] = []
    for symbol in prepared.risk_names:
        frame = frames[symbol]
        group = group_for(prepared, symbol)
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("raw exact signal close missing")
        if len(frame) < 253:
            continue
        recent = frame.iloc[-20:]
        score = float(frame.tr.iloc[-1] / frame.tr.iloc[-253] - 1)
        if not isfinite(score) or not isfinite(cash):
            raise ValueError("nonfinite raw return")
        if (
            fsum(float(v) for v in recent.turnover) / 20 >= 1e7
            and all(float(v) > 0 for v in recent.volume)
            and float(frame.close.iloc[-1]) >= 5
        ):
            scores.append((score, symbol, group))
    picks: list[tuple[float, str]] = []
    occupied: set[str] = set()
    for score, symbol, group in sorted(scores, key=lambda x: (-x[0], x[1])):
        if group in occupied:
            continue
        picks.append((score, symbol))
        occupied.add(group)
        if len(picks) == top:
            break
    q = {s: 0.9 / top for score, s in picks if score > cash}
    q["SHY"] = max(0.0, 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")
