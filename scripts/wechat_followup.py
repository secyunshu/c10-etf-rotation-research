"""Isolated, frozen V2.3 comparison of Guosen-inspired ETF rankings through 2025."""

import argparse
import gzip
import json
from datetime import date
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard import audit, native
from c10standard.io import ROOT, load_strategy, settings_for
from c10standard.models import Artifact, Contract, Prepared, TargetRecord
from c10standard.run import sha
from rotbt.models import Model
from rotbt.phase2.models import FeatureStore
from scripts import annual_io
from scripts.annual_data import load_prepared
from scripts.conditional_sleeve import CloseContract
from scripts.current_standard import HISTORY, select_target
from scripts.ytd_annual_report import Statistics, summarize

STUDY = ROOT / "experiments/wechat-followup-2025"
Case = Literal["base252", "total20", "overnight20", "intraday_reversal20", "high_recency252"]
CASES: tuple[Case, ...] = (
    "base252", "total20", "overnight20", "intraday_reversal20", "high_recency252",
)
Cost = Literal[5, 20]
COSTS: tuple[Cost, ...] = (5, 20)


class SessionReturns(Model):
    overnight: pd.Series
    intraday: pd.Series


class SummaryRow(Model):
    case: Case
    cost_bps: Cost
    changed_signals: int
    stats: Statistics


def session_returns(frame: pd.DataFrame) -> SessionReturns:
    prices = frame[["open", "close", "tr"]]
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("session return decomposition requires positive prices")
    intraday = pd.Series(np.log((prices.close / prices.open).to_numpy()), index=frame.index)
    total = pd.Series(np.log((prices.tr / prices.tr.shift()).to_numpy()), index=frame.index)
    return SessionReturns(overnight=total - intraday, intraday=intraday)


def high_age(prices: pd.Series, days: int = 252) -> pd.Series:
    def recency(path: NDArray[np.float64]) -> float:
        return -float(np.argmax(path[::-1]))

    return prices.rolling(days).apply(recency, raw=True)


def scores(case: Case, features: FeatureStore) -> FeatureStore:
    if case == "base252":
        return features
    updated: dict[str, pd.DataFrame] = {}
    for symbol, frame in features.frames.items():
        if case == "high_recency252":
            score = high_age(frame.tr)
        elif case == "total20":
            score = pd.Series(np.log((frame.tr / frame.tr.shift(20)).to_numpy()), index=frame.index)
        else:
            parts = session_returns(frame)
            leg = parts.overnight if case == "overnight20" else -parts.intraday
            score = leg.rolling(20).sum()
        updated[symbol] = frame.assign(mom252=score)
    return FeatureStore(frames=updated)


def account(market: Prepared, case: Case, cost: Cost,
            signals: dict[date, TargetRecord], execute: bool) -> Artifact:
    folder = STUDY / "reference" / case / f"{cost}bp"
    contract = CloseContract(cost_bps=cost)
    settings = settings_for(market.strategy, contract)
    settings.invested = 1.

    def target(actual: Contract, prepared: Prepared, day: date) -> TargetRecord:
        if actual != contract or prepared is not market:
            raise ValueError("signal applied to unexpected contract")
        return signals[day]

    if execute:
        raw = ROOT / "runs/wechat-followup-2025" / case / f"{cost}bp"
        raw.mkdir(parents=True)
        with patch.object(native, "target", target):
            artifact = native.simulate(market, contract, settings, raw)
    else:
        proof = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, proof.hashes)
        raw_account = gzip.decompress((folder / "ACCOUNT.json.gz").read_bytes())
        artifact = Artifact.model_validate_json(raw_account)
    if artifact.contract.model_dump() != contract.model_dump():
        raise ValueError("saved account identity mismatch")
    artifact.contract = contract
    with patch.object(audit, "oracle", target):
        audit.audit(artifact, market, settings)
    calendar = xcals.get_calendar("XNYS")
    due = {calendar.next_session(pd.Timestamp(day)).date() for day in signals}
    if any(o.day not in due for o in artifact.account.orders):
        raise ValueError("order outside next-open schedule")
    if case == "base252" and cost == 5:
        ref = ROOT / HISTORY / "reference/C11"
        proof = Fingerprints.model_validate_json((ref / "VERIFIED.json").read_bytes())
        verify_hashes(ref, proof.hashes)
        original = Artifact.model_validate_json((ref / "ARTIFACT.json").read_bytes())
        if Artifact.model_validate_json(artifact.model_dump_json()) != original:
            raise ValueError("252-day account differs from the published V2.3 reference")
    if execute:
        folder.mkdir(parents=True)
        native.export_books(folder, artifact)
        (folder / "ACCOUNT.json.gz").write_bytes(gzip.compress(artifact.model_dump_json().encode()))
        native.write(folder / "VERIFIED.json", Fingerprints(
            hashes={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        ))
    return artifact


def run(execute: bool) -> None:
    annual_io.verify_lock(ROOT)
    inputs = (
        STUDY / "PLAN.md", Path(__file__).resolve(), ROOT / "scripts/current_standard.py",
        ROOT / "scripts/conditional_sleeve.py", ROOT / "scripts/ytd_annual_report.py",
        ROOT / annual_io.STUDY / "SOURCE_LOCK.json", ROOT / HISTORY / "reference/C11/VERIFIED.json",
    )
    binding = Fingerprints(hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs})
    lock = STUDY / "SOURCE.json"
    if execute:
        native.write(lock, binding)
    elif Fingerprints.model_validate_json(lock.read_bytes()) != binding:
        raise ValueError("frozen experiment source differs")
    data = load_prepared("ANNUAL")
    market = Prepared(datasets=data.datasets, features=data.features, strategy=load_strategy(ROOT),
                      first=data.first, last=data.last,
                      risk_names=tuple(d.symbol for d in data.datasets if d.symbol != "SHY"))
    contract = CloseContract()
    cal = xcals.get_calendar("XNYS")
    initial = cal.previous_session(pd.Timestamp(contract.start)).date()
    days = (initial, *native.decision_days(contract))
    pools = {p.year: p for p in data.schedule.entries}
    rows: list[SummaryRow] = []
    all_signals: dict[Case, dict[date, TargetRecord]] = {}
    for case in CASES:
        features = scores(case, data.features)
        signals: dict[date, TargetRecord] = {}
        for day in days:
            execution = cal.next_session(pd.Timestamp(day)).date()
            pool = pools[execution.year]
            if pool.source_asof >= execution:
                raise ValueError("pool unavailable before execution")
            signals[day] = select_target(pool, features, day)
            prefix = FeatureStore(frames={s: data.features.frames[s].loc[:day.isoformat()]
                                          for s in pool.risk_names})
            if select_target(pool, scores(case, prefix), day) != signals[day]:
                raise ValueError("ranking changed after truncating future data")
        all_signals[case] = signals
        base = all_signals["base252"]
        changed = sum(s.q != base[d].q for d, s in signals.items())
        for cost in COSTS:
            book = account(market, case, cost, signals, execute).account
            for period in ("2022", "2023", "2024", "2025", "full"):
                rows.append(SummaryRow(case=case, cost_bps=cost, changed_signals=changed,
                                       stats=summarize(book.daily, book.orders, book.distributions,
                                                       period)))
            print(case, cost, "PASS", flush=True)
    outputs = (
        ("RESULTS.json", [r.model_dump(mode="json") for r in rows]),
        ("SIGNALS.json", {case: [s.model_dump(mode="json") for s in records.values()]
                          for case, records in all_signals.items()}),
    )
    for name, payload in outputs:
        path = STUDY / name
        if execute:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
        elif json.loads(path.read_text()) != payload:
            raise ValueError(f"archived {name} differs from recomputed evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args().execute)
