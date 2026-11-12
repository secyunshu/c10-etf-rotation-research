"""Registered US ETF momentum adaptations using the existing Backtrader broker."""

import argparse
import csv
import gzip
from datetime import date
from math import isfinite, log, sqrt
from pathlib import Path
from typing import Literal, override
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard import audit as financial
from c10standard import native
from c10standard.io import ROOT, load_strategy, settings_for
from c10standard.models import Artifact, Contract, Prepared, TargetRecord
from c10standard.run import sha
from rotbt.models import Dataset, Model
from scripts import annual_io, ytd_annual_io
from scripts.annual_data import load_prepared as load_history
from scripts.annual_models import Pool
from scripts.conditional_sleeve import CloseContract, allocation
from scripts.conditional_sleeve_ytd import ResetContract
from scripts.ytd_annual_data import load_prepared as load_ytd
from scripts.ytd_annual_models import LAST
from scripts.ytd_annual_models import Pool as YtdPool
from scripts.ytd_annual_report import summarize

STUDY = Path("experiments/hkcodex-momentum")
Case = Literal["base252", "fixed126", "adaptive", "drop"]
CASES: tuple[Case, ...] = ("base252", "fixed126", "adaptive", "drop")
Cost = Literal[5, 20]
COSTS: tuple[Cost, ...] = (5, 20)


class YtdContract(ResetContract):
    @property
    @override
    def through(self) -> date:
        return LAST


class Window(Model):
    label: str
    contract: CloseContract | YtdContract
    market: Prepared
    pools: dict[int, Pool | YtdPool]
    reference: Path


def adaptive_days(ranges: NDArray[np.float64]) -> int:
    if len(ranges) < 63 or not np.isfinite(ranges[-63:]).all():
        raise ValueError("63 finite true ranges required")
    return 126 if ranges[-20:].mean() > ranges[-63:].mean() * (1 + 1e-12) else 252


def recent_drop(prices: NDArray[np.float64]) -> bool:
    if len(prices) < 68 or np.any(prices[-68:] <= 0):
        raise ValueError("68 positive prices required")
    returns = np.diff(np.log(prices[-68:]))
    sigma = float(returns[:-4].std(ddof=1))
    if not isfinite(sigma) or not np.isfinite(returns[-4:]).all():
        raise ValueError("nonfinite shock data")
    recent = returns[-4:]
    return bool((recent[-3:] < -2.5 * sigma).any()) or any(
        bool((part < 0).all() and part.sum() < -2.5 * sqrt(3) * sigma)
        for part in (recent[:3], recent[1:])
    )


def slot_weights(picks: tuple[str, ...]) -> dict[str, float]:
    if len(picks) == 2:
        return allocation(picks)
    weights = {s: .45 for s in picks}
    weights["SHY"] = .45 * (2 - len(picks))
    sleeve = "SHY" if {"QQQ", "SPY"}.intersection(picks) else "QQQ"
    weights[sleeve] = weights.get(sleeve, 0.) + .10
    return weights


def normalized_ranges(frame: pd.DataFrame, prices: NDArray[np.float64]) -> NDArray[np.float64]:
    adj = prices / frame.close.to_numpy(dtype=float)
    high = frame.high.to_numpy(dtype=float) * adj
    low = frame.low.to_numpy(dtype=float) * adj
    previous = prices[:-1]
    ranges = np.maximum.reduce((high[1:] - low[1:], abs(high[1:] - previous),
                                abs(low[1:] - previous)))
    return np.asarray(ranges / previous, dtype=np.float64)


def select(case: Case, pool: Pool | YtdPool, datasets: list[Dataset], day: date) -> TargetRecord:
    scores: list[tuple[float, str]] = []
    eligible = set(pool.risk_names)
    groups = {s: group for group, members in pool.groups.groups.items() for s in members}
    for data in datasets:
        if data.symbol not in eligible:
            continue
        frame = data.frame.loc[:str(day)]
        if len(frame) < 253 or frame.index[-1].date() != day:
            raise ValueError(f"incomplete signal history: {data.symbol} {day}")
        recent = frame.iloc[-20:]
        if (recent.close.iloc[-1] < 5 or recent.turnover.mean() < 1e7
                or not (recent.volume > 0).all()):
            continue
        prices = frame.tr.to_numpy(dtype=float)
        if case == "drop" and recent_drop(prices):
            continue
        lookback = 126 if case == "fixed126" else 252
        if case == "adaptive":
            lookback = adaptive_days(normalized_ranges(frame, prices))
        score = log(float(prices[-1] / prices[-lookback - 1])) / lookback
        if not isfinite(score):
            raise ValueError("nonfinite momentum")
        scores.append((-score, data.symbol))
    picks: list[str] = []
    used: set[str] = set()
    for _, symbol in sorted(scores):
        group = groups[symbol]
        if group not in used:
            picks.append(symbol)
            used.add(group)
            if len(picks) == 2:
                break
    return TargetRecord(day=day, q=slot_weights(tuple(picks)), reason="oracle")


def windows() -> tuple[Window, Window]:
    history = load_history("ANNUAL")
    ytd = load_ytd("ANNUAL", "science")
    strategy = load_strategy(ROOT)
    return (
        Window(label="2022-2025", contract=CloseContract(),
               pools={p.year: p for p in history.schedule.entries},
               reference=ROOT / "experiments/annual-qqq-spy-sleeve/reference/C11/ARTIFACT.json",
               market=Prepared(datasets=history.datasets, features=history.features,
                               strategy=strategy, first=history.first, last=history.last,
                               risk_names=tuple(d.symbol for d in history.datasets
                                                if d.symbol != "SHY"))),
        Window(label="2026YTD", contract=YtdContract(),
               pools={p.year: p for p in ytd.schedule.entries},
               reference=ROOT / "experiments/conditional-sleeve-2026/reference/C11/ARTIFACT.json",
               market=Prepared(datasets=ytd.datasets, features=ytd.features,
                               strategy=strategy, first=ytd.first, last=ytd.last,
                               risk_names=tuple(d.symbol for d in ytd.datasets
                                                if d.symbol != "SHY"))),
    )


def planned_signals(window: Window, case: Case) -> dict[date, TargetRecord]:
    calendar = xcals.get_calendar("XNYS")
    days = (calendar.previous_session(pd.Timestamp(window.contract.start)).date(),
            *native.decision_days(window.contract))
    result = {}
    for day in days:
        pool = window.pools[calendar.next_session(pd.Timestamp(day)).year]
        prefix = [d.model_copy(update={"frame": d.frame.loc[:str(day)].copy()})
                  for d in window.market.datasets]
        target = select(case, pool, window.market.datasets, day)
        if target != select(case, pool, prefix, day):
            raise ValueError("physical prefix changed the signal")
        result[day] = target
    return result


def verified_account(window: Window, case: Case, cost: Cost, execute: bool,
                     signals: dict[date, TargetRecord]) -> Artifact:
    contract = window.contract.model_copy(update={"cost_bps": cost})
    settings = settings_for(window.market.strategy, contract)
    settings.invested = 1.
    folder = ROOT / STUDY / "reference" / window.label / case / f"{cost}bp"

    def target(actual: Contract, market: Prepared, day: date) -> TargetRecord:
        if actual != contract or market is not window.market:
            raise ValueError("account callback identity mismatch")
        return signals[day]

    if execute:
        raw = ROOT / "runs/hkcodex-momentum" / window.label / case / f"{cost}bp"
        raw.mkdir(parents=True)
        with patch.object(native, "target", target):
            artifact = native.simulate(window.market, contract, settings, raw)
    else:
        fingerprint = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, fingerprint.hashes)
        with gzip.open(folder / "ARTIFACT.json.gz", "rb") as handle:
            artifact = Artifact.model_validate_json(handle.read())
    if artifact.contract.model_dump() != contract.model_dump():
        raise ValueError("saved contract differs from experiment")
    artifact.contract = contract
    with patch.object(financial, "oracle", target):
        financial.audit(artifact, window.market, settings)
    calendar = xcals.get_calendar("XNYS")
    due = {calendar.next_session(pd.Timestamp(day)).date() for day in signals}
    if any(o.day not in due for o in artifact.account.orders):
        raise ValueError("trade outside scheduled next opens")
    if execute:
        folder.mkdir(parents=True)
        native.export_books(folder, artifact)
        with gzip.open(folder / "ARTIFACT.json.gz", "xb") as handle:
            handle.write(artifact.model_dump_json().encode())
        native.write(folder / "VERIFIED.json", Fingerprints(
            hashes={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        ))
    return artifact


def run(execute: bool) -> None:
    annual_io.verify_lock(ROOT)
    ytd_annual_io.verify_lock(ROOT)
    study = ROOT / STUDY
    markets = windows()
    inputs = (ROOT / "scripts/adaptive_momentum.py", study / "PLAN.md",
              ROOT / annual_io.STUDY / "SOURCE_LOCK.json",
              ROOT / ytd_annual_io.STUDY / "SOURCE_LOCK.json",
              ROOT / "scripts/conditional_sleeve.py", ROOT / "scripts/conditional_sleeve_ytd.py",
              *(window.reference for window in markets))
    fingerprints = Fingerprints(hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs})
    lock = study / "SOURCE.json"
    if execute:
        native.write(lock, fingerprints)
    elif Fingerprints.model_validate_json(lock.read_bytes()) != fingerprints:
        raise ValueError("study inputs changed")
    rows = []
    for window in markets:
        original = Artifact.model_validate_json(window.reference.read_bytes())
        base_signals = planned_signals(window, "base252")
        for case in CASES:
            signals = base_signals if case == "base252" else planned_signals(window, case)
            changed = sum(s.q != base_signals[day].q for day, s in signals.items())
            for cost in COSTS:
                artifact = verified_account(window, case, cost, execute, signals)
                book = artifact.account
                if case == "base252" and cost == 5:
                    if original.targets != artifact.targets or [
                        r for r in original.account.daily if r.day <= window.contract.through
                    ] != book.daily:
                        raise ValueError("5 bp baseline does not reproduce the saved account")
                periods: tuple[str, ...] = ("full",)
                if window.label == "2022-2025":
                    periods = ("2022", "2023", "2024", "2025", "full")
                for period in periods:
                    stats = summarize(book.daily, book.orders, book.distributions, period)
                    rows.append({"window": window.label, "case": case, "cost_bps": cost,
                                 "changed_signals": changed,
                                 **stats.model_dump(mode="json", exclude={"cagr"})})
                print(window.label, case, cost, "PASS", flush=True)
    report = study / "RESULTS.csv"
    if execute:
        with report.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        with report.open(encoding="utf-8") as handle:
            expected = [{k: str(v) for k, v in row.items()} for row in rows]
            if list(csv.DictReader(handle)) != expected:
                raise ValueError("saved results differ from audited ledgers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the 16 registered accounts")
    run(parser.parse_args().execute)
