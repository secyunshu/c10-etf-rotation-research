"""Fresh 2026 sleeve and buy-and-hold accounts using the frozen annual pool."""

import argparse
import json
from datetime import date
from pathlib import Path
from typing import override
from unittest.mock import patch

import exchange_calendars as xcals
import pandas as pd
from pydantic import ConfigDict, Field, TypeAdapter

from c10experiments.report import Fingerprints
from c10share.io import verify_hashes
from c10standard import audit as financial
from c10standard import native
from c10standard.io import ROOT, load_strategy, settings_for
from c10standard.models import Artifact, Cohort, Contract, Prepared, TargetRecord
from c10standard.run import sha
from rotbt.models import Dataset, Model
from scripts import ytd_annual_io, ytd_annual_report, ytd_annual_signals
from scripts.conditional_sleeve import RULES, allocation
from scripts.ytd_annual_data import load_prepared
from scripts.ytd_annual_models import LAST as ARCHIVE_END
from scripts.ytd_annual_models import Contract as AnnualContract

STUDY = Path("experiments/conditional-sleeve-2026")
LAST = date(2026, 11, 11)


class ResetContract(Contract):
    cohort: Cohort = "reset2026"

    @property
    @override
    def through(self) -> date:
        return LAST


class ClosingBar(Model):
    model_config = ConfigDict(allow_inf_nan=False)
    symbol: str
    day: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(gt=0)
    turnover: float = Field(gt=0)


def append_bar(data: Dataset, bar: ClosingBar) -> Dataset:
    if (bar.symbol != data.symbol or bar.day != LAST
            or data.frame.index[-1].date() != ARCHIVE_END):
        raise ValueError("latest mark symbol or date differs from the registered append")
    if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
        raise ValueError("invalid OHLC bounds")
    row = bar.model_dump(exclude={"symbol", "day"})
    row["tr"] = float(data.frame.tr.iloc[-1] * bar.close / data.frame.close.iloc[-1])
    last = pd.DataFrame([row], index=pd.DatetimeIndex([bar.day], name="date"))
    return data.model_copy(update={"frame": pd.concat([data.frame, last])})


def run(execute: bool) -> None:
    output = ROOT / STUDY
    ytd_annual_io.verify_lock(ROOT)
    market = load_prepared("ANNUAL", "science")
    annual = AnnualContract(role="ANNUAL", stage="science", cost_bps=5,
                            schedule_sha256=market.schedule.sha256)
    calendar = xcals.get_calendar("XNYS")
    initial = calendar.previous_session(pd.Timestamp(ResetContract().start)).date()
    days = (initial, *native.decision_days(ResetContract()))
    if any(day > ARCHIVE_END for day in days):
        raise ValueError("supplemental mark is too early to form a new monthly signal")
    ytd_annual_signals.verify_targets(annual, market, days)
    signals = {}
    for day in days:
        original = ytd_annual_signals.oracle(annual, market, day)
        # The old gate follows ranking. Two passing picks prove the ungated picks are identical.
        picks = tuple(s for s in original.q if s != "SHY")
        signals[day] = TargetRecord(day=day, q=allocation(picks), reason="oracle")

    required = {name for signal in signals.values() for name in signal.q} | {"SPY", "QQQ", "SHY"}
    bar_file = output / "NOV11_BARS.json"
    bars = TypeAdapter(list[ClosingBar]).validate_json(bar_file.read_bytes())
    if len(bars) != len(required) or {bar.symbol for bar in bars} != required:
        raise ValueError("latest marks must cover every possible execution feed exactly once")
    datasets = [append_bar(next(d for d in market.datasets if d.symbol == bar.symbol), bar)
                for bar in bars]
    prepared = Prepared(datasets=datasets, features=ytd_annual_signals.features_for(datasets),
                        strategy=load_strategy(ROOT), risk_names=tuple(sorted(required - {"SHY"})),
                        first=market.first, last=LAST)
    inputs = (ROOT / "scripts/conditional_sleeve_ytd.py", ROOT / "scripts/conditional_sleeve.py",
              ROOT / ytd_annual_io.STUDY / "SOURCE_LOCK.json", output / "PLAN.md", bar_file,
              output / "NOV11_SOURCE.json")
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    lock = output / "SOURCE.json"
    if execute:
        native.write(lock, Fingerprints(hashes=hashes))
    elif Fingerprints.model_validate_json(lock.read_bytes()).hashes != hashes:
        raise ValueError("source or input hash changed")

    def target(contract: Contract, data: Prepared, day: date) -> TargetRecord:
        if data is not prepared or contract.through != LAST:
            raise ValueError("target callback bound to the wrong account")
        if contract.rule.endswith("_BH") and day == initial:
            symbol = contract.rule.removesuffix("_BH")
            return TargetRecord(day=day, q={symbol: 1.0}, reason="oracle")
        if contract.rule != "C11":
            raise ValueError("unexpected benchmark signal")
        return signals[day]

    summaries = {}
    for rule in RULES:
        contract = ResetContract(rule=rule)
        settings = settings_for(prepared.strategy, contract)
        settings.invested = 1.0
        folder = output / "reference" / rule
        if execute:
            folder.mkdir(parents=True)
            with patch.object(native, "target", target):
                artifact = native.simulate(prepared, contract, settings, folder)
            native.export_books(folder, artifact)
        else:
            proof = Fingerprints.model_validate_json((folder / "VERIFIED.json").read_bytes())
            verify_hashes(folder, proof.hashes)
            artifact = Artifact.model_validate_json((folder / "ARTIFACT.json").read_bytes())
        if artifact.contract.model_dump() != contract.model_dump():
            raise ValueError("saved account role or starting cohort mismatch")
        artifact.contract = contract
        with patch.object(financial, "oracle", target):
            financial.audit(artifact, prepared, settings)
        due = {calendar.next_session(pd.Timestamp(r.day)).date() for r in artifact.targets}
        book = artifact.account
        if not book.orders or book.orders[0].day != contract.start or any(
            o.day not in due for o in book.orders
        ) or (rule.endswith("_BH") and len(book.orders) != 1):
            raise ValueError("fill schedule differs from the registered account")
        if execute:
            native.write(folder / "VERIFIED.json", Fingerprints(
                hashes={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
            ))
        with patch.object(ytd_annual_report, "LAST", LAST):
            stats = ytd_annual_report.summarize(book.daily, book.orders, book.distributions, "full")
        summaries[rule] = stats.model_dump(mode="json", exclude={"cagr", "period"})
    report = output / "RESULTS.json"
    if execute:
        with report.open("x", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)
    elif json.loads(report.read_text()) != summaries:
        raise ValueError("saved summary differs from the verified daily ledgers")
    print(f"Verified {len(RULES)} fresh Backtrader accounts through {LAST}: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="create the three research accounts")
    run(parser.parse_args().execute)
