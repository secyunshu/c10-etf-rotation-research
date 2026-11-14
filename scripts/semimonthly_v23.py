"""Compare V2.3 monthly and semi-monthly schedules on frozen data."""

import argparse
import gzip
import json
from datetime import date
from pathlib import Path
from typing import Literal, override
from unittest.mock import patch

import exchange_calendars as xcals
import pandas as pd

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard import audit, native
from c10standard.io import ROOT, load_strategy, settings_for
from c10standard.models import Artifact, Contract, Prepared, TargetRecord
from c10standard.run import sha
from rotbt.models import Model
from rotbt.phase2.models import FeatureStore
from scripts.annual_data import load_prepared as load_history
from scripts.annual_models import Pool
from scripts.conditional_sleeve import CloseContract
from scripts.conditional_sleeve_ytd import ResetContract
from scripts.current_standard import HISTORY, RELEASE, YTD, select_target, verify_release
from scripts.v2_2023_signals import features_for
from scripts.ytd_annual_data import load_prepared as load_ytd
from scripts.ytd_annual_models import LAST
from scripts.ytd_annual_models import Pool as YtdPool
from scripts.ytd_annual_report import Statistics, summarize

STUDY = ROOT / "experiments/semimonthly-v2-3"
Frequency = Literal["monthly", "semimonthly"]
FREQUENCIES: tuple[Frequency, ...] = ("monthly", "semimonthly")
Cost = Literal[5, 20]
COSTS: tuple[Cost, ...] = (5, 20)
monthly_days = native.decision_days


class CommonEnd(ResetContract):
    @property
    @override
    def through(self) -> date:
        return LAST


class StudyWindow(Model):
    name: str
    contract: CloseContract | CommonEnd
    market: Prepared
    pools: dict[int, Pool | YtdPool]
    reference: Path


class Summary(Model):
    window: str
    frequency: Frequency
    cost_bps: Cost
    decisions: int
    additional_changed_targets: int
    stats: Statistics


def decision_days(contract: Contract, frequency: Frequency) -> tuple[date, ...]:
    monthly = monthly_days(contract)
    if frequency == "monthly":
        return monthly
    calendar = xcals.get_calendar("XNYS")
    months = pd.period_range(contract.start, contract.through, freq="M")
    mid = [calendar.date_to_session(pd.Timestamp(p.year, p.month, 15), direction="previous").date()
           for p in months]
    return tuple(sorted(set(monthly) | {d for d in mid if contract.start <= d < contract.through}))


def windows() -> tuple[StudyWindow, StudyWindow]:
    history, ytd = load_history("ANNUAL"), load_ytd("ANNUAL", "science")
    strategy = load_strategy(ROOT)
    return (
        StudyWindow(name="2022-2025", contract=CloseContract(),
                    pools={p.year: p for p in history.schedule.entries},
                    reference=ROOT / HISTORY / "reference/C11",
                    market=Prepared(datasets=history.datasets, features=history.features,
                                    strategy=strategy, first=history.first, last=history.last,
                                    risk_names=tuple(d.symbol for d in history.datasets
                                                     if d.symbol != "SHY"))),
        StudyWindow(name="2026YTD", contract=CommonEnd(),
                    pools={p.year: p for p in ytd.schedule.entries},
                    reference=ROOT / YTD / "reference/C11",
                    market=Prepared(datasets=ytd.datasets, features=ytd.features,
                                    strategy=strategy, first=ytd.first, last=ytd.last,
                                    risk_names=tuple(d.symbol for d in ytd.datasets
                                                     if d.symbol != "SHY"))),
    )


def signals_for(window: StudyWindow, frequency: Frequency) -> list[TargetRecord]:
    calendar = xcals.get_calendar("XNYS")
    initial = calendar.previous_session(pd.Timestamp(window.contract.start)).date()
    result = []
    for day in (initial, *decision_days(window.contract, frequency)):
        next_open = calendar.next_session(pd.Timestamp(day)).date()
        pool = window.pools[next_open.year]
        if pool.source_asof >= next_open or next_open > window.contract.through:
            raise ValueError("signal cannot execute with the available pool and cutoff")
        record = select_target(pool, window.market.features, day)
        prefixes = [d.model_copy(update={"frame": d.frame.loc[:day.isoformat()]})
                    for d in window.market.datasets if d.symbol in pool.risk_names]
        cropped: FeatureStore = features_for(prefixes)
        if record != select_target(pool, cropped, day):
            raise ValueError("future data changed the target")
        result.append(record)
    return result


def account(window: StudyWindow, frequency: Frequency, cost: Cost,
            signals: list[TargetRecord], execute: bool) -> Artifact:
    contract = window.contract.model_copy(update={"cost_bps": cost})
    settings = settings_for(window.market.strategy, contract)
    settings.invested = 1.
    folder = STUDY / "reference" / window.name / frequency / f"{cost}bp"
    by_day = {r.day: r for r in signals}
    scheduled = decision_days(contract, frequency)

    def schedule(actual: Contract) -> tuple[date, ...]:
        if actual != contract:
            raise ValueError("wrong contract for schedule")
        return scheduled

    def target(actual: Contract, market: Prepared, day: date) -> TargetRecord:
        if actual != contract or market is not window.market:
            raise ValueError("wrong market for signal")
        return by_day[day]

    if execute:
        raw = ROOT / "runs/semimonthly-v2-3" / window.name / frequency / f"{cost}bp"
        raw.mkdir(parents=True)
        with (patch.object(native, "target", target),
              patch.object(native, "decision_days", schedule)):
            artifact = native.simulate(window.market, contract, settings, raw)
    else:
        proof = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, proof.hashes)
        saved = gzip.decompress((folder / "ACCOUNT.json.gz").read_bytes())
        artifact = Artifact.model_validate_json(saved)
    if artifact.contract.model_dump() != contract.model_dump():
        raise ValueError("account identity differs from experiment")
    artifact.contract = contract
    with patch.object(audit, "oracle", target), patch.object(audit, "decision_days", schedule):
        audit.audit(artifact, window.market, settings)
    calendar = xcals.get_calendar("XNYS")
    due = {calendar.next_session(pd.Timestamp(r.day)).date() for r in signals}
    if any(o.day not in due for o in artifact.account.orders):
        raise ValueError("order outside next-open schedule")
    if frequency == "monthly" and cost == 5:
        proof = Fingerprints.model_validate_json((window.reference / "VERIFIED.json").read_bytes())
        verify_hashes(window.reference, proof.hashes)
        original = Artifact.model_validate_json((window.reference / "ARTIFACT.json").read_bytes())
        if window.name == "2022-2025":
            if Artifact.model_validate_json(artifact.model_dump_json()) != original:
                raise ValueError("monthly historical account does not reproduce V2.3")
        elif original.targets != artifact.targets or [
            d for d in original.account.daily if d.day <= contract.through
        ] != artifact.account.daily:
            raise ValueError("2026 monthly account differs on the common cutoff")
    if execute:
        folder.mkdir(parents=True)
        native.export_books(folder, artifact)
        (folder / "ACCOUNT.json.gz").write_bytes(gzip.compress(artifact.model_dump_json().encode()))
        native.write(folder / "VERIFIED.json", Fingerprints(
            hashes={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        ))
    return artifact


def run(execute: bool) -> None:
    verify_release(ROOT)
    paths = [Path(__file__).resolve(), STUDY / "PLAN.md", ROOT / RELEASE / "RELEASE_LOCK.json"]
    proof = Fingerprints(hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    lock = STUDY / "SOURCE.json"
    if execute:
        native.write(lock, proof)
    elif Fingerprints.model_validate_json(lock.read_bytes()) != proof:
        raise ValueError("experiment source differs from frozen version")
    rows: list[Summary] = []
    traces: dict[str, list[TargetRecord]] = {}
    for window in windows():
        monthly = {s.day: s for s in signals_for(window, "monthly")}
        for frequency in FREQUENCIES:
            signals = list(monthly.values()) if frequency == "monthly" else signals_for(
                window, frequency
            )
            if [s for s in signals if s.day in monthly] != list(monthly.values()):
                raise ValueError("baseline decisions changed under the new schedule")
            extra_changes = sum(b.day not in monthly and a.q != b.q
                                for a, b in zip(signals[:-1], signals[1:], strict=True))
            traces[f"{window.name}/{frequency}"] = signals
            for cost in COSTS:
                book = account(window, frequency, cost, signals, execute).account
                periods: tuple[str, ...] = ("full",)
                if window.name == "2022-2025":
                    periods = ("2022", "2023", "2024", "2025", "full")
                for period in periods:
                    stats = summarize(book.daily, book.orders, book.distributions, period)
                    rows.append(Summary(window=window.name, frequency=frequency, cost_bps=cost,
                                        decisions=len(signals),
                                        additional_changed_targets=extra_changes, stats=stats))
                print(window.name, frequency, cost, "PASS", flush=True)
    outputs = (
        ("RESULTS.json", [r.model_dump(mode="json") for r in rows]),
        ("SIGNALS.json", {name: [s.model_dump(mode="json") for s in records]
                          for name, records in traces.items()}),
    )
    for name, payload in outputs:
        path = STUDY / name
        if execute:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
        elif json.loads(path.read_text()) != payload:
            raise ValueError(f"saved {name} differs from verified accounts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args().execute)
