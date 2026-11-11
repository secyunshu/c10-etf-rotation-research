"""Recompute annual-pool results from saved native ledgers, with no broker runs."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from statistics import stdev
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from beartype import beartype
from pydantic import TypeAdapter

from c10experiments.report import Fingerprints, Table, compare_table
from c10share.io import verify_hashes
from c10share.report import read_csv
from c10share.verify import compare_json
from c10standard.io import load_accounts, load_strategy, reference_folder
from c10standard.models import NativeScalar, Receipt, TargetRecord
from rotbt.analysis import analyze
from rotbt.models import (
    ClosedTrade,
    DailyRecord,
    Distribution,
    HoldingRecord,
    Model,
    OrderRecord,
    ResearchPeriod,
    Result,
    Settings,
    Spec,
)
from scripts.annual_data import load_prepared
from scripts.annual_io import ROOT, STUDY, verify_lock
from scripts.annual_models import Artifact, Binding, Contract, Prepared
from scripts.annual_native import audit


class Case(Model):
    role: str
    cost: int
    result: Result
    targets: list[TargetRecord]
    timings: list[dict[str, Any]] = []

    @property
    def key(self) -> str:
        return f"{self.role}_{self.cost}bp"


@beartype
def validate_days(daily: list[DailyRecord]) -> None:
    days = [
        x.date() for x in xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2025-12-31")
    ]
    if [d.day for d in daily] != days:
        raise ValueError("continuous fouryear calendar mismatch")
    if daily[-1].positions:
        raise ValueError("final account not liquidated")


def load_case(role: str, cost: int, folder: Path, prepared: Prepared, root: Path) -> Case:
    timings = []
    contract: Contract | None = None
    if role == "ANNUAL":
        contract = Contract.model_validate({"mode": "ANNUAL", "cost_bps": cost,
                                            "stage": "science",
                                            "schedule_sha256": prepared.schedule.sha256})
        binding = Binding.model_validate_json((folder / "BINDING.json").read_bytes())
        if binding.contract != contract or binding.source_sha256 != verify_lock(root):
            raise ValueError("annual report identity/source binding mismatch")
        if (folder / "REFERENCE.json").exists():
            fp = Fingerprints.model_validate_json((folder / "REFERENCE.json").read_bytes())
            if "BINDING.json" not in fp.hashes:
                raise ValueError("reference missing binding")
            verify_hashes(folder, fp.hashes)
        else:
            receipt = Receipt.model_validate_json((folder / "VERIFIED.json").read_bytes())
            verify_hashes(folder, receipt.hashes)
        if Contract.model_validate_json((folder / "CONTRACT.json").read_bytes()) != contract:
            raise ValueError("native account differs from requested annual variant")
        timings = json.loads((folder / "timings.json").read_text())
    daily = list(read_csv(folder / "daily.csv.gz", DailyRecord))
    validate_days(daily)
    holdings = list(read_csv(folder / "holdings.csv.gz", HoldingRecord))
    names = {h.symbol for h in holdings}
    result = Result(
        spec=Spec(name=role, layer="hold", top=3, lookback=252),
        daily=daily,
        holdings=holdings,
        orders=list(read_csv(folder / "orders.csv.gz", OrderRecord)),
        distributions=list(read_csv(folder / "dividends.csv.gz", Distribution)),
        trades=list(read_csv(folder / "trades.csv.gz", ClosedTrade)),
        native_total_return=NativeScalar.model_validate_json(
            (folder / "native_total_return.json").read_bytes()
        ).native_total_return,
        splits=[
            s
            for d in prepared.datasets
            if d.symbol in names
            for s in d.splits
            if date(2022, 1, 3) <= s.day <= date(2025, 12, 31)
        ],
    )
    targets = TypeAdapter(list[TargetRecord]).validate_json((folder / "targets.json").read_bytes())
    if role == "ANNUAL":
        if contract is None:
            raise ValueError("annual contract missing")
        artifact = Artifact.model_validate(
            {
                "contract": contract,
                "account": result,
                "targets": targets,
                "decision_times": timings,
                "native_returns": json.loads((folder / "native_returns.json").read_text()),
                "open_trades": json.loads((folder / "open_trades.json").read_text()),
            }
        )
        audit(artifact, prepared)
    settings = Settings(
        initial_cash=50000,
        invested=0.95,
        slip_bps=cost,
        tax=0,
        fees=prepared.fees,
        research_period=ResearchPeriod(first_year=2022, last_year=2025, split_year=2023),
    )
    computed = analyze(result, settings)
    compare_json(
        json.loads((folder / "metrics.json").read_text()),
        [m.model_dump(mode="json") for m in computed],
        "archivedmetrics",
    )
    return Case(role=role, cost=cost, result=result, targets=targets, timings=timings)


def build(root: Path, native_output: Path, output: Path) -> None:
    locked = verify_lock(root)
    prepared = load_prepared("ANNUAL", root)
    accounts = load_accounts(root)  # Metadata only, then filter before loading historicalpayloads.
    cases = []
    for cost in (5, 20):
        contract = Contract(
            mode="ANNUAL", cost_bps=cost, stage="science", schedule_sha256=prepared.schedule.sha256
        )
        cases.append(load_case("ANNUAL", cost, native_output / contract.key, prepared, root))
        for rule in ("C11", "SPY_BH", "QQQ_BH"):
            account = next(
                a
                for a in accounts
                if a.rule == rule and a.cost_bps == cost and a.cohort == "historical2022_2025"
            )
            cases.append(load_case(rule, cost, reference_folder(account, root), prepared, root))
    rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    year_open_orders: list[dict[str, Any]] = []
    calendar = xcals.get_calendar("XNYS")
    first_opens = {
        calendar.date_to_session(f"{y}-01-01", direction="next").date() for y in range(2022, 2026)
    }
    for case in cases:
        settings = Settings(
            initial_cash=50000,
            invested=0.95,
            slip_bps=case.cost,
            tax=0,
            fees=load_strategy(root).fees,
            research_period=ResearchPeriod(first_year=2022, last_year=2025, split_year=2023),
        )
        previous = 50000.0
        changes = []
        for d in case.result.daily:
            changes.append((d.day, d.equity / previous - 1))
            previous = d.equity
        for m in analyze(case.result, settings):
            if m.period not in ("2022", "2023", "2024", "2025", "full"):
                continue
            sample = [v for day, v in changes if m.period == "full" or day.year == int(m.period)]
            rows.append(
                {
                    "key": case.key,
                    "role": case.role,
                    "cost_bps": case.cost,
                    **m.model_dump(mode="json"),
                    "annualized_volatility": stdev(sample) * 252**0.5,
                }
            )
        targets.extend(
            {"key": case.key, "day": str(t.day), "symbol": symbol, "account_weight": weight * 0.95}
            for t in case.targets
            for symbol, weight in t.q.items()
        )
        timings.extend({"key": case.key, **t} for t in case.timings)
        year_open_orders.extend(
            {"key": case.key, **o.model_dump(mode="json")}
            for o in case.result.orders
            if o.day in first_opens
        )
    output.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("ANNUAL.csv", [r for r in rows if r["period"] != "full"], ("key", "period")),
        ("SUMMARY.csv", [r for r in rows if r["period"] == "full"], ("key", "period")),
        ("TARGETS.csv", targets, ("key", "day", "symbol")),
        ("TIMINGS.csv", timings, ("key", "price_asof")),
        ("YEAR_OPEN_ORDERS.csv", year_open_orders, ("key", "day", "symbol")),
    )
    cells = 0
    hashes = {}
    for name, data, keys in definitions:
        table = Table(name=name, keys=keys, frame=pd.DataFrame(data))
        path = output / name
        if not path.exists():
            table.frame.to_csv(path, index=False, float_format="%.17g")
        cells += compare_table(table, path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "status": "PASS",
        "cases": 8,
        "annual_rows": 32,
        "summary_rows": 8,
        "checked_table_cells": cells,
        "source_lock": locked,
        "native_attempts": 0,
        "table_hashes": hashes,
    }
    path = output / "REPORT_VERIFICATION.json"
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise ValueError("immutable report receipt differs")
    else:
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(
        args.root,
        args.native_output or args.root / STUDY / "reference",
        args.output or args.root / STUDY / "results",
    )


if __name__ == "__main__":
    main()
