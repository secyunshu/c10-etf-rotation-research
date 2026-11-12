"""Offline 2022–25 allocation experiment using the original Backtrader account."""

import argparse
import csv
from datetime import date
from math import isfinite, prod
from pathlib import Path
from typing import override
from unittest.mock import patch

import exchange_calendars as xcals
import pandas as pd
from pydantic import TypeAdapter

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard import native
from c10standard.audit import FinancialScope, equal, reconcile, reconcile_trade_books
from c10standard.io import ROOT, load_strategy, settings_for
from c10standard.models import Artifact, Contract, Rule, TargetRecord
from c10standard.models import Prepared as NativePrepared
from c10standard.run import sha
from scripts import annual_io, annual_signals
from scripts.annual_data import load_prepared
from scripts.annual_models import FIRST, LAST, Prepared
from scripts.annual_models import Contract as AnnualContract
from scripts.ytd_annual_report import summarize

STUDY = Path("experiments/annual-qqq-spy-sleeve")
RULES: tuple[Rule, ...] = ("C11", "SPY_BH", "QQQ_BH")


class CloseContract(Contract):
    @property
    @override
    def through(self) -> date:
        return LAST

    @property
    @override
    def terminal(self) -> bool:
        return False


def allocation(picks: tuple[str, ...]) -> dict[str, float]:
    if len(picks) != 2 or len(set(picks)) != 2 or "SHY" in picks:
        raise ValueError("requires two distinct ranked risk ETFs")
    sleeve = "SHY" if {"QQQ", "SPY"}.intersection(picks) else "QQQ"
    return {picks[0]: .45, picks[1]: .45, sleeve: .10}


def select(contract: AnnualContract, prepared: Prepared, day: date) -> TargetRecord:
    pool = annual_signals.active_pool(contract, prepared, day)
    eligible: list[tuple[float, str]] = []
    for symbol in pool.risk_names:
        frame = prepared.features.frames[symbol].loc[:str(day)]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError(f"missing signal close for {symbol}")
        if len(frame) < 253:
            continue
        momentum, price, turnover, positive = (
            float(frame.iloc[-1][field])
            for field in ("mom252", "raw_close", "turnover20", "positive20")
        )
        if not all(isfinite(v) for v in (momentum, price, turnover, positive)):
            raise ValueError(f"invalid signal data for {symbol}")
        if price >= 5 and turnover >= 1e7 and positive == 20:
            eligible.append((-momentum, symbol))
    picks: list[str] = []
    groups: set[str] = set()
    for _, symbol in sorted(eligible):
        group = annual_signals.group_for(pool, symbol)
        if group not in groups:
            groups.add(group)
            picks.append(symbol)
            if len(picks) == 2:
                break
    return TargetRecord(day=day, q=allocation(tuple(picks)), reason="oracle")


def run(execute: bool) -> None:
    output = ROOT / STUDY
    annual_io.verify_lock(ROOT)
    data = load_prepared("ANNUAL", ROOT)
    annual = AnnualContract(mode="ANNUAL", cost_bps=5, stage="science",
                            schedule_sha256=data.schedule.sha256)
    prepared = NativePrepared(datasets=data.datasets, features=data.features,
                              strategy=load_strategy(ROOT), first=FIRST, last=LAST,
                              risk_names=tuple(d.symbol for d in data.datasets
                                               if d.symbol != "SHY"))
    calendar = xcals.get_calendar("XNYS")
    start = CloseContract().start
    initial = calendar.previous_session(pd.Timestamp(start)).date()
    days = (initial, *native.decision_days(CloseContract()))
    reference = ROOT / annual_io.STUDY / "reference" / annual.key / "targets.json"
    archived = TypeAdapter(list[TargetRecord]).validate_json(reference.read_bytes())
    signals = {day: select(annual, data, day) for day in days}
    annual_signals.verify_targets(annual, data, days)
    if tuple(r.day for r in archived) != days:
        raise ValueError("signal calendar differs from the verified historical archive")
    for record in archived:
        raw = annual_signals.oracle(annual, data, record.day)
        if record.q != raw.q or signals[record.day].q != allocation(
            tuple(s for s in raw.q if s != "SHY")
        ):
            raise ValueError("ungated ranking differs from the historical two qualified picks")

    source_files = (ROOT / "scripts/conditional_sleeve.py", ROOT / "scripts/ytd_annual_report.py",
                    ROOT / annual_io.STUDY / "SOURCE_LOCK.json", reference, output / "PLAN.md")
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}
    lock = output / "SOURCE.json"
    if execute:
        native.write(lock, Fingerprints(hashes=hashes))
    elif Fingerprints.model_validate_json(lock.read_bytes()).hashes != hashes:
        raise ValueError("source or input hash changed")

    def target(contract: Contract, market: NativePrepared, day: date) -> TargetRecord:
        if contract.rule != "C11" or market is not prepared:
            raise ValueError("annual target used outside its study account")
        return signals[day]

    rows = []
    for rule in RULES:
        contract = CloseContract(rule=rule)
        settings = settings_for(prepared.strategy, contract)
        settings.invested = 1.0
        folder = output / "reference" / rule
        path = folder / "ARTIFACT.json"
        if execute:
            folder.mkdir(parents=True)
            with patch.object(native, "target", target):
                artifact = native.simulate(prepared, contract, settings, folder)
            native.export_books(folder, artifact)
        else:
            fingerprint = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
            verify_hashes(folder, fingerprint.hashes)
            artifact = Artifact.model_validate_json(path.read_bytes())

        expected = list(signals.values()) if rule == "C11" else [
            TargetRecord(day=initial, q={rule.removesuffix("_BH"): 1.0}, reason="oracle")
        ]
        if artifact.contract.model_dump() != contract.model_dump() or [
            (r.day, r.q) for r in artifact.targets
        ] != [(r.day, r.q) for r in expected]:
            raise ValueError("native account identity or applied targets differ")
        due = {calendar.next_session(pd.Timestamp(r.day)).date() for r in expected}
        if not artifact.account.orders or any(o.day not in due for o in artifact.account.orders):
            raise ValueError("fills must be on registered next-open dates")
        if rule != "C11" and len(artifact.account.orders) != 1:
            raise ValueError("buy and hold must have only its initial buy")
        feeds = [d for d in data.datasets if d.symbol in native.feed_names(contract, prepared)]
        reconcile(artifact.account, feeds, FinancialScope(start=start, through=LAST,
                                                        terminal=False, settings=settings))
        reconcile_trade_books(artifact.account, artifact.open_trades)
        previous = settings.initial_cash
        for nav, ret in zip(artifact.account.daily, artifact.native_returns, strict=True):
            if nav.day != ret.day:
                raise ValueError("TimeReturn calendar differs from account")
            equal(ret.value, nav.equity / previous - 1, "native daily return", 1e-12)
            previous = nav.equity
        equal(prod(1 + r.value for r in artifact.native_returns) - 1,
              previous / settings.initial_cash - 1, "compounded native return", 1e-12)
        if execute:
            native.write(folder / "VERIFIED.json", Fingerprints(
                hashes={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
            ))
        book = artifact.account
        for period in ("2022", "2023", "2024", "2025", "full"):
            rows.append({"strategy": rule, **summarize(book.daily, book.orders, book.distributions,
                                                      period).model_dump(mode="json")})

    report = output / "RESULTS.csv"
    if execute:
        with report.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with report.open(newline="", encoding="utf-8") as handle:
            if list(csv.DictReader(handle)) != [
                {key: "" if value is None else str(value) for key, value in row.items()}
                for row in rows
            ]:
                raise ValueError("saved report differs from recomputed account metrics")
    print(f"Verified {len(RULES)} Backtrader accounts and {len(rows)} report rows: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="create the three research accounts")
    run(parser.parse_args().execute)
