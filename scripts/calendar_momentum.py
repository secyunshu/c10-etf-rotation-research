"""Compare three ETF ranking windows using the frozen annual pools and native broker."""

import argparse
import gzip
from datetime import date
from math import isfinite, prod
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import exchange_calendars as xcals
import pandas as pd
from pydantic import TypeAdapter

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard import audit as financial
from c10standard import native
from c10standard.io import ROOT, settings_for
from c10standard.models import Artifact, Contract, Prepared, TargetRecord
from c10standard.run import sha
from rotbt.models import Model
from scripts import adaptive_momentum, annual_io, ytd_annual_io
from scripts.adaptive_momentum import COSTS, Cost, Window
from scripts.conditional_sleeve import allocation
from scripts.ytd_annual_report import Statistics, summarize

STUDY = ROOT / "experiments/calendar-skip-month"
Case = Literal["base252", "calendar12", "french12_2"]
CASES: tuple[Case, ...] = ("base252", "calendar12", "french12_2")


class Signal(Model):
    target: TargetRecord
    start: date
    end: date
    execute_on: date
    pool_year: int


class ResultRow(Model):
    window: str
    case: Case
    cost_bps: Cost
    changed_vs_252: int
    stats: Statistics


def calendar_ends(day: date) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    cal = xcals.get_calendar("XNYS")
    month = pd.Period(day, freq="M")
    full = cal.date_to_session(month.end_time.normalize(), direction="previous")
    if full.date() != day:
        raise ValueError("signal must fall on exchange month end")
    start = cal.date_to_session((month - 12).end_time.normalize(), direction="previous")
    skip = cal.date_to_session((month - 1).end_time.normalize(), direction="previous")
    return start, full, skip


def calendar_score(total_return: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    base, last = float(total_return.loc[start]), float(total_return.loc[end])
    if min(base, last) <= 0 or not (isfinite(base) and isfinite(last)) or start >= end:
        raise ValueError("invalid total-return window")
    return last / base - 1


def signal(window: Window, case: Case, day: date) -> Signal:
    cal = xcals.get_calendar("XNYS")
    execute = cal.next_session(pd.Timestamp(day))
    pool = window.pools[execute.year]
    if pool.source_asof > execute.date():
        raise ValueError("annual pool source is unavailable at execution")
    start, full, skip = calendar_ends(day)
    end = skip if case == "french12_2" else full
    if case == "base252":
        start = cal.session_offset(full, -252)
        target = adaptive_momentum.select("base252", pool, window.market.datasets, day)
    else:
        ranked = []
        group = {s: g for g, names in pool.groups.groups.items() for s in names}
        for asset in window.market.datasets:
            if asset.symbol not in group:
                continue
            history = asset.frame.loc[:day.isoformat()]
            if len(history) < 253 or history.index[-1] != full:
                raise ValueError("invalid signal history")
            recent = history.iloc[-20:]
            if recent.close.iloc[-1] < 5 or recent.turnover.mean() < 1e7:
                continue
            if not (recent.volume > 0).all():
                continue
            series = history.tr.loc[start:end]
            score = calendar_score(series, start, end)
            if abs(score - (prod(1 + r for r in series.pct_change().iloc[1:]) - 1)) > 1e-12:
                raise ValueError("total-return endpoints disagree with daily compounding")
            ranked.append((-score, asset.symbol))
        picks: list[str] = []
        for _, symbol in sorted(ranked):
            if not any(group[s] == group[symbol] for s in picks):
                picks.append(symbol)
                if len(picks) == 2:
                    break
        target = TargetRecord(day=day, q=allocation(tuple(picks)), reason="oracle")
    return Signal(target=target, start=start.date(), end=end.date(), execute_on=execute.date(),
                  pool_year=pool.year)


def account(
    window: Window, case: Case, cost: Cost, signals: list[Signal], execute: bool
) -> Artifact:
    contract = window.contract.model_copy(update={"cost_bps": cost})
    settings = settings_for(window.market.strategy, contract)
    settings.invested = 1.
    folder = STUDY / "reference" / window.label / case / f"{cost}bp"
    target_by_day = {s.target.day: s.target for s in signals}

    def target(actual: Contract, market: Prepared, day: date) -> TargetRecord:
        if actual != contract or market is not window.market:
            raise ValueError("wrong account for signal")
        return target_by_day[day]

    if execute:
        raw = ROOT / "runs/calendar-skip-month" / window.label / case / f"{cost}bp"
        raw.mkdir(parents=True)
        with patch.object(native, "target", target):
            artifact = native.simulate(window.market, contract, settings, raw)
    else:
        proof = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, proof.hashes)
        saved = gzip.decompress((folder / "ARTIFACT.json.gz").read_bytes())
        artifact = Artifact.model_validate_json(saved)
    if artifact.contract.model_dump() != contract.model_dump():
        raise ValueError("saved contract differs from study")
    artifact.contract = contract
    with patch.object(financial, "oracle", target):
        financial.audit(artifact, window.market, settings)
    due = {s.execute_on for s in signals}
    if any(o.day not in due for o in artifact.account.orders):
        raise ValueError("order outside next-open schedule")
    if case == "base252":
        previous = ROOT / adaptive_momentum.STUDY / "reference" / window.label / case / f"{cost}bp"
        proof = Fingerprints.model_validate_json((previous / "VERIFIED.json").read_bytes())
        verify_hashes(previous, proof.hashes)
        saved = gzip.decompress((previous / "ARTIFACT.json.gz").read_bytes())
        if Artifact.model_validate_json(saved) != Artifact.model_validate_json(
            artifact.model_dump_json()
        ):
            raise ValueError("baseline differs from previous Backtrader account")
    if execute:
        folder.mkdir(parents=True)
        native.export_books(folder, artifact)
        compressed = gzip.compress(artifact.model_dump_json().encode())
        (folder / "ARTIFACT.json.gz").write_bytes(compressed)
        native.write(folder / "VERIFIED.json", Fingerprints(
            hashes={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        ))
    return artifact


def run(execute: bool) -> None:
    annual_io.verify_lock(ROOT)
    ytd_annual_io.verify_lock(ROOT)
    inputs = [Path(__file__).resolve(), STUDY / "PLAN.md",
              ROOT / adaptive_momentum.STUDY / "SOURCE.json",
              ROOT / annual_io.STUDY / "SOURCE_LOCK.json",
              ROOT / ytd_annual_io.STUDY / "SOURCE_LOCK.json"]
    prior = Fingerprints.model_validate_json(inputs[2].read_bytes())
    for name, expected in prior.hashes.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != expected:
            raise ValueError(f"invalid archived source: {name}")
    binding = Fingerprints(hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs})
    lock = STUDY / "SOURCE.json"
    if execute:
        native.write(lock, binding)
    elif Fingerprints.model_validate_json(lock.read_bytes()) != binding:
        raise ValueError("study source changed")
    rows: list[ResultRow] = []
    traces: dict[str, list[Signal]] = {}
    for window in adaptive_momentum.windows():
        cal = xcals.get_calendar("XNYS")
        days = [cal.previous_session(pd.Timestamp(window.contract.start)).date(),
                *native.decision_days(window.contract)]
        base = [signal(window, "base252", day) for day in days]
        for case in CASES:
            signals = base if case == "base252" else [signal(window, case, day) for day in days]
            for item in signals:
                prefix = [d.model_copy(update={"frame": d.frame.loc[:item.target.day.isoformat()]})
                          for d in window.market.datasets]
                cropped = window.model_copy(update={"market": window.market.model_copy(
                    update={"datasets": prefix}
                )})
                if signal(cropped, case, item.target.day) != item:
                    raise ValueError("signal differs on physically truncated data")
            traces[f"{window.label}/{case}"] = signals
            changed = sum(a.target.q != b.target.q for a, b in zip(base, signals, strict=True))
            for cost in COSTS:
                book = account(window, case, cost, signals, execute).account
                periods: tuple[str, ...] = ("full",)
                if window.label == "2022-2025":
                    periods = ("2022", "2023", "2024", "2025", "full")
                for period in periods:
                    rows.append(ResultRow(window=window.label, case=case, cost_bps=cost,
                                          changed_vs_252=changed, stats=summarize(
                                              book.daily, book.orders, book.distributions, period)))
                print(window.label, case, cost, "PASS", flush=True)
    table = TypeAdapter(list[ResultRow])
    trace = TypeAdapter(dict[str, list[Signal]])
    for name, payload in (("RESULTS.json", table.dump_json(rows, indent=2)),
                          ("SIGNALS.json", trace.dump_json(traces, indent=2))):
        path = STUDY / name
        payload += b"\n"
        if execute:
            with path.open("xb") as handle:
                handle.write(payload)
        elif path.read_bytes() != payload:
            raise ValueError(f"archived {name} differs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args().execute)
