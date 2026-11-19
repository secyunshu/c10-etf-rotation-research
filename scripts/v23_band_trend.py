"""Frozen V2.3 monthly study of a trade band and two short rankings."""

import argparse
import gzip
from collections import defaultdict
from datetime import date
from math import expm1, floor, fsum, isclose, log, sqrt
from pathlib import Path
from typing import Literal, cast, override
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype
from pydantic import TypeAdapter

from c10experiments.report import Fingerprints
from c10share.actions import HistoricalRotation
from c10share.io import verify_hashes
from c10standard import audit, native
from c10standard.io import ROOT, load_strategy, settings_for
from c10standard.models import Artifact, Contract, Prepared, TargetRecord
from c10standard.run import sha
from rotbt.models import Model, Result, Settings
from rotbt.phase2.models import FeatureStore
from scripts import annual_io
from scripts.annual_data import load_prepared
from scripts.annual_models import Pool
from scripts.conditional_sleeve import CloseContract
from scripts.current_standard import HISTORY, select_target
from scripts.ytd_annual_report import Statistics, summarize

STUDY = ROOT / "experiments/v23-band-and-trend"
Case = Literal["base252", "band1", "total25", "reg25"]
ShortCase = Literal["total25", "reg25"]
Cost = Literal[5, 20]
CASES: tuple[Case, ...] = ("base252", "band1", "total25", "reg25")
COSTS: tuple[Cost, ...] = (5, 20)
DAYS, BAND = 25, .01


class SummaryRow(Model):
    case: Case
    cost_bps: Cost
    changed_signals: int
    band_months: int
    legs_inside_band: int
    suppressed_share_adjustments: int
    skips_all_orders: int
    stats: Statistics


class BandEvidence(Model):
    date: date
    enabled: bool
    inside: tuple[str, ...]
    suppressed: tuple[str, ...]
    orders: int


@beartype
def allow_band(previous: TargetRecord | None, target: TargetRecord, held: frozenset[str]) -> bool:
    return previous is not None and previous.q == target.q and held == target.q.keys()


@beartype
def short_score(case: ShortCase, path: tuple[float, ...]) -> float:
    values = np.asarray(path)
    if len(values) != DAYS or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("ranking requires 25 finite positive closes")
    if case == "total25":
        return float(values[-1] / values[0] - 1)
    if np.ptp(values) == 0:
        return 0.
    x = np.arange(DAYS) - (DAYS - 1) / 2
    y = np.log(values)
    centered = y - y.mean()
    slope = float((x @ centered) / (x @ x))
    r2 = float((x @ centered) ** 2 / ((x @ x) * (centered @ centered)))
    return expm1(252 * slope) * r2


def reference_score(case: ShortCase, path: tuple[float, ...]) -> float:
    if case == "total25":
        return expm1(log(path[-1] / path[0]))
    if min(path) == max(path):
        return 0.
    # Independent fit: pairwise slopes weighted by squared time distance.
    y = tuple(log(p / path[0]) for p in path)
    pairs = [(j - i, y[j] - y[i]) for j in range(len(y)) for i in range(j)]
    slope = fsum(t * change for t, change in pairs) / fsum(t**2 for t, _ in pairs)
    average = fsum(y) / len(y)
    intercept = average - slope * (len(y) - 1) / 2
    residual = fsum((v - intercept - t * slope)**2 for t, v in enumerate(y))
    total = fsum((v - average)**2 for v in y)
    return expm1(252 * slope) * (1 - residual / total)


def select_short(pool: Pool, features: FeatureStore, day: date, case: ShortCase) -> TargetRecord:
    frames: dict[str, pd.DataFrame] = {}
    ranked: list[tuple[float, str, str]] = []
    groups = {s: group for group, names in pool.groups.groups.items() for s in names}
    for symbol in pool.risk_names:
        frame = features.frames[symbol].loc[:day.isoformat()]
        if len(frame) < 253 or frame.index[-1].date() != day:
            raise ValueError(f"missing V2.3 history for {symbol} on {day}")
        path = tuple(float(p) for p in frame.tr.iloc[-DAYS:])
        value = short_score(case, path)
        expected = reference_score(case, path)
        if not isclose(value, expected, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError(f"short score disagrees: {symbol} {day}")
        frames[symbol] = frame.assign(mom252=value)
        row = frame.iloc[-1]
        if row.raw_close >= 5 and row.turnover20 >= 1e7 and row.positive20 == 20:
            ranked.append((-expected, symbol, groups[symbol]))
    result = select_target(pool, FeatureStore(frames=frames), day)
    picks: list[str] = []
    used: set[str] = set()
    for _, symbol, group in sorted(ranked):
        if group not in used:
            used.add(group)
            picks.append(symbol)
            if len(picks) == 2:
                break
    sleeve = "SHY" if {"QQQ", "SPY"}.intersection(picks) else "QQQ"
    if result.q != {picks[0]: .45, picks[1]: .45, sleeve: .1}:
        raise ValueError(f"independent short ranking disagrees: {case} {day}")
    return result


class BandRotation(HistoricalRotation):
    prior_target: TargetRecord | None = None

    @override
    def _stage(
        self, weights: dict[str, float], day: date, reason: str, *, force: bool = False,
    ) -> None:
        if self.p2 is None:
            raise ValueError("trade band requires Phase 2 execution")
        held = frozenset(data._name for data in self.datas if self.getposition(data).size > 0)
        target = TargetRecord(day=day, q=weights, reason=reason)
        active = not force and allow_band(self.prior_target, target, held)
        self.p2 = self.p2.model_copy(update={"band": BAND if active else 0.})
        self.prior_target = target
        super()._stage(weights, day, reason, force=force)


def check_execution(artifact: Artifact, market: Prepared, case: Case) -> list[BandEvidence]:
    """Rebuild pre-trade open weights independently from the audited closing books."""
    account = artifact.account
    cal = xcals.get_calendar("XNYS")
    by_day = {cal.next_session(pd.Timestamp(s.day)).date(): s for s in artifact.targets}
    close_cash = {r.day: r.cash for r in account.daily}
    close_holding: dict[date, dict[str, int]] = defaultdict(dict)
    fills: dict[date, dict[str, int]] = defaultdict(dict)
    for h in account.holdings:
        if h.size:
            close_holding[h.day][h.symbol] = h.size
    for o in account.orders:
        if o.day not in by_day or o.symbol in fills[o.day]:
            raise ValueError("unexpected order time or duplicate order for symbol")
        fills[o.day][o.symbol] = o.size
    funds = {d.symbol: d.frame for d in market.datasets}
    prior_q: dict[str, float] = {}
    result: list[BandEvidence] = []
    for day, target in by_day.items():
        prev = close_holding[target.day].copy()
        enabled = case == "band1" and prior_q == target.q and prev.keys() == target.q.keys()
        for event in account.splits:
            if event.day == day and event.symbol in prev:
                prev[event.symbol] = round(prev[event.symbol] * event.shares_after_per_before)
        opens = {
            s: float(cast(float, funds[s].at[pd.Timestamp(day), "open"]))
            for s in prev.keys() | target.q.keys()
        }
        # Same-open entitlements enter broker cash after sizing, so subtract only prior locks.
        locked = fsum(
            d.net_amount for d in account.distributions if d.ex_date < day < d.release_date
        )
        investable = close_cash.get(target.day, 50000.) - locked + fsum(
            shares * opens[s] for s, shares in prev.items()
        )
        inside: list[str] = []
        suppressed: list[str] = []
        for symbol, opening in sorted(opens.items()):
            weight, owned = target.q.get(symbol, 0.), prev.get(symbol, 0)
            fill = fills[day].get(symbol, 0)
            adverse_open = opening * (1 + artifact.contract.cost / 10000)
            requested = floor(investable * weight / adverse_open)
            in_band = enabled and abs(owned * opening / investable - weight) < .01
            if in_band:
                inside.append(symbol)
                if requested != owned:
                    suppressed.append(symbol)
                requested = owned
            delta = requested - owned
            if (delta <= 0 and fill != delta) or (delta > 0 and not 0 <= fill <= delta):
                raise ValueError(
                    f"order outside native band or target: {case} {day} {symbol}: "
                    f"fill={fill} delta={delta} in_band={in_band}"
                )
        if not fills[day].keys() <= opens.keys():
            raise ValueError("fill outside held or target symbols")
        result.append(BandEvidence(date=day, enabled=enabled, inside=tuple(inside),
                                   suppressed=tuple(suppressed), orders=len(fills[day])))
        prior_q = target.q
    return result


def check_statistics(book: Result, stats: Statistics) -> None:
    selected = [r.equity for r in book.daily if stats.first <= r.day <= stats.last]
    opening = next((r.equity for r in reversed(book.daily) if r.day < stats.first), 50000.)
    equity = np.array([opening, *selected])
    changes = equity[1:] / equity[:-1] - 1
    fills = [o for o in book.orders if stats.first <= o.day <= stats.last]
    for actual, expected in (
        (stats.total_return, equity[-1] / opening - 1),
        (stats.cagr, (equity[-1] / opening) ** (252 / len(changes)) - 1),
        (stats.max_drawdown, max(1 - equity / np.maximum.accumulate(equity))),
        (stats.sharpe_rf0, float(changes.mean() / changes.std(ddof=1)) * sqrt(252)),
        (stats.turnover, fsum(abs(o.size * o.price) for o in fills) / float(equity[1:].mean())),
        (stats.fees, fsum(o.fee for o in fills)),
        (stats.orders, len(fills)),
        (stats.days, len(changes)),
    ):
        if actual is None or not isclose(actual, float(expected), abs_tol=1e-9, rel_tol=1e-10):
            raise ValueError(f"independent metric mismatch in {stats.period}")


def get_account(market: Prepared, schedule: list[TargetRecord], case: Case, cost: Cost,
                output: Path, execute: bool) -> Artifact:
    contract = CloseContract(cost_bps=cost)
    settings: Settings = settings_for(market.strategy, contract)
    settings.invested = 1.
    signals = {record.day: record for record in schedule}
    folder = output / "reference" / case / f"{cost}bp"

    def target(actual: Contract, prepared: Prepared, day: date) -> TargetRecord:
        if actual != contract or prepared is not market:
            raise ValueError("unexpected account identity")
        return signals[day]

    if execute:
        folder.mkdir(parents=True)
        with (patch.object(native, "target", target),
              patch.object(native, "HistoricalRotation",
                           BandRotation if case == "band1" else HistoricalRotation)):
            artifact = native.simulate(market, contract, settings, folder)
    else:
        proof = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, proof.hashes)
        raw = gzip.decompress((folder / "ACCOUNT.json.gz").read_bytes())
        artifact = Artifact.model_validate_json(raw)
    if artifact.contract.model_dump() != contract.model_dump():
        raise ValueError("saved contract mismatch")
    artifact.contract = contract
    with patch.object(audit, "oracle", target):
        audit.audit(artifact, market, settings)
    if case == "base252" and cost == 5:
        original = ROOT / HISTORY / "reference/C11"
        proof = Fingerprints.model_validate_json((original / "VERIFIED.json").read_bytes())
        verify_hashes(original, proof.hashes)
        expected = Artifact.model_validate_json((original / "ARTIFACT.json").read_bytes())
        if Artifact.model_validate_json(artifact.model_dump_json()) != expected:
            raise ValueError("5 bp baseline did not reproduce the published account")
    if execute:
        with (folder / "ACCOUNT.json.gz").open("xb") as handle:
            handle.write(gzip.compress(artifact.model_dump_json().encode(), mtime=0))
        native.write(folder / "VERIFIED.json", Fingerprints(
            hashes={"ACCOUNT.json.gz": sha(folder / "ACCOUNT.json.gz")},
        ))
        # Retain the native evidence on failure; successful account already has a lossless copy.
        (folder / "ARTIFACT.json").unlink()
    return artifact


def run(output: Path, *, execute: bool) -> None:
    source = annual_io.verify_lock(ROOT)
    paths = (STUDY / "PLAN.md", STUDY / "tests/test_rules.py", Path(__file__).resolve(),
             ROOT / "scripts/current_standard.py", ROOT / "scripts/conditional_sleeve.py",
             ROOT / "scripts/ytd_annual_report.py", ROOT / "requirements.lock",
             ROOT / HISTORY / "reference/C11/VERIFIED.json")
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    hashes[str(annual_io.STUDY / "SOURCE_LOCK.json")] = source
    binding = Fingerprints(hashes=hashes)
    if execute:
        output.mkdir(parents=True, exist_ok=True)
        native.write(output / "SOURCE.json", binding)
    elif Fingerprints.model_validate_json((output / "SOURCE.json").read_bytes()) != binding:
        raise ValueError("frozen research inputs changed")
    data = load_prepared("ANNUAL")
    market = Prepared(datasets=data.datasets, features=data.features, strategy=load_strategy(ROOT),
                      first=data.first, last=data.last,
                      risk_names=tuple(d.symbol for d in data.datasets if d.symbol != "SHY"))
    cal = xcals.get_calendar("XNYS")
    contract = CloseContract()
    initial = cal.previous_session(pd.Timestamp(contract.start)).date()
    days = (initial, *native.decision_days(contract))
    pools = {p.year: p for p in data.schedule.entries}
    schedules: dict[Case, list[TargetRecord]] = {}
    execution: dict[str, list[BandEvidence]] = {}
    rows: list[SummaryRow] = []
    for case in CASES:
        schedule = []
        for day in days:
            trade_day = cal.next_session(pd.Timestamp(day)).date()
            pool = pools[trade_day.year]
            if pool.source_asof >= trade_day:
                raise ValueError("annual pool unavailable before open")
            prefix = FeatureStore(frames={
                s: data.features.frames[s].loc[:day.isoformat()] for s in pool.risk_names
            })
            if case in ("total25", "reg25"):
                record = select_short(pool, data.features, day, case)
                truncated = select_short(pool, prefix, day, case)
            else:
                record = select_target(pool, data.features, day)
                truncated = select_target(pool, prefix, day)
            if record != truncated:
                raise ValueError(f"as-of signal mismatch: {case} {day}")
            schedule.append(record)
        schedules[case] = schedule
        changed = sum(a.q != b.q for a, b in zip(schedule, schedules["base252"], strict=True))
        for cost in COSTS:
            artifact = get_account(market, schedule, case, cost, output, execute)
            book = artifact.account
            evidence = check_execution(artifact, market, case)
            execution[f"{case}_{cost}bp"] = evidence
            for period in ("2022", "2023", "2024", "2025", "full"):
                stats = summarize(book.daily, book.orders, book.distributions, period)
                check_statistics(book, stats)
                selected = [e for e in evidence if stats.first <= e.date <= stats.last]
                rows.append(SummaryRow(
                    case=case, cost_bps=cost, changed_signals=changed,
                    band_months=sum(e.enabled for e in selected),
                    legs_inside_band=sum(len(e.inside) for e in selected),
                    suppressed_share_adjustments=sum(len(e.suppressed) for e in selected),
                    skips_all_orders=sum(bool(e.suppressed) and e.orders == 0 for e in selected),
                    stats=stats,
                ))
            print(f"{case} {cost} bp PASS", flush=True)
    signal_json = TypeAdapter(dict[Case, list[TargetRecord]]).dump_json(schedules, indent=2)
    execution_json = TypeAdapter(dict[str, list[BandEvidence]]).dump_json(execution, indent=2)
    for name, payload in (
        ("RESULTS.json", TypeAdapter(list[SummaryRow]).dump_json(rows, indent=2)),
        ("SIGNALS.json", signal_json),
        ("EXECUTION.json", execution_json),
    ):
        path = output / name
        if execute:
            with path.open("xb") as handle:
                handle.write(payload + b"\n")
        elif path.read_bytes() != payload + b"\n":
            raise ValueError(f"saved {name} differs from independently verified accounts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(args.output, execute=args.execute)
