"""Historical-only tables and exposure diagnostics from saved Backtrader books."""

import argparse
import hashlib
import json
from math import isclose
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from beartype import beartype
from pydantic import TypeAdapter

from c10experiments.report import Case, Fingerprints, Table, compare_table
from c10share.io import verify_hashes
from c10share.report import read_csv
from c10standard.io import ROOT, load_accounts, reference_folder
from c10standard.models import Contract, NativeScalar, Receipt, TargetRecord
from c10standard.report import Book, Nav, annual_rows, monthly_rows, summary_row, verify_book
from rotbt.models import DailyRecord, Distribution, HoldingRecord, Metrics, OrderRecord
from scripts.qqq_overlay_run import Binding, Trial, verify_lock

STUDY = Path("experiments/qqq-overlay")


@beartype
def validate_history(contract: Contract, rows: tuple[DailyRecord, ...]) -> None:
    if not contract.terminal:
        raise ValueError("only historical2022_2025 is permitted")
    expected = [
        d.date()
        for d in xcals.get_calendar("XNYS").sessions_in_range(contract.start, contract.through)
    ]
    if [r.day for r in rows] != expected:
        raise ValueError("native calendar differs from complete historical window")
    if rows[-1].positions != 0:
        raise ValueError("historical terminal must be liquidated")


@beartype
def read_book(
    variant: str,
    contract: Contract,
    folder: Path,
    root: Path,
    *,
    science: bool,
) -> Case:
    if science:
        binding = Binding.model_validate_json((folder / "BINDING.json").read_bytes())
        if binding.trial != Trial(variant="OVERLAY", contract=contract) or variant != "OVERLAY":
            raise ValueError("report signal identity mismatch")
        if binding.source_lock_sha256 != verify_lock(root):
            raise ValueError("report source binding mismatch")
        if (folder / "REFERENCE.json").exists():
            fingerprints = Fingerprints.model_validate_json(
                (folder / "REFERENCE.json").read_bytes()
            )
            if "BINDING.json" not in fingerprints.hashes:
                raise ValueError("reference must include signal binding")
            verify_hashes(folder, fingerprints.hashes)
        else:
            receipt = Receipt.model_validate_json((folder / "VERIFIED.json").read_bytes())
            verify_hashes(folder, receipt.hashes)
        if Contract.model_validate_json((folder / "CONTRACT.json").read_bytes()) != contract:
            raise ValueError("native contract mismatch")
    daily = read_csv(folder / "daily.csv.gz", DailyRecord)
    validate_history(contract, daily)
    scalar = NativeScalar.model_validate_json((folder / "native_total_return.json").read_bytes())
    if not isclose(
        daily[-1].equity / 50_000 - 1, scalar.native_total_return, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError("native return differs from NAV")
    book = Book(
        contract=contract,
        initial_cash=50_000,
        daily=tuple(Nav(day=r.day, equity=r.equity) for r in daily),
        orders=read_csv(folder / "orders.csv.gz", OrderRecord),
        dividends=read_csv(folder / "dividends.csv.gz", Distribution),
        archived_metrics=TypeAdapter(tuple[Metrics, ...]).validate_json(
            (folder / "metrics.json").read_bytes()
        ),
    )
    verify_book(book)
    return Case(
        key=f"{variant}_{contract.cost_bps}bp",
        variant=variant,
        folder=folder,
        book=book,
        targets=TypeAdapter(tuple[TargetRecord, ...]).validate_json(
            (folder / "targets.json").read_bytes()
        ),
        holdings=read_csv(folder / "holdings.csv.gz", HoldingRecord),
    )


def cases_for(root: Path, native_output: Path) -> list[Case]:
    accounts = load_accounts(root)  # Metadata only; no2026 reference payloads are read.
    cases = []
    for cost in (5, 20):
        contract = Contract(cost_bps=cost)
        cases.append(
            read_book(
                "OVERLAY", contract, native_output / "OVERLAY" / contract.key, root, science=True
            )
        )
        for rule in ("C11", "SPY_BH", "QQQ_BH"):
            matches = [
                a
                for a in accounts
                if a.rule == rule and a.cost_bps == cost and a.cohort == "historical2022_2025"
            ]
            if len(matches) != 1:
                raise ValueError("exactly one historical control required")
            account = matches[0]
            place = reference_folder(account, root)
            cases.append(
                read_book(
                    "V2" if rule == "C11" else rule, account.contract, place, root, science=False
                )
            )
    if len({c.key for c in cases}) != 8:
        raise ValueError("two science plus six controls required")
    return cases


def tables(cases: list[Case]) -> list[Table]:
    summary: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    months: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    exposure: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    calendar = xcals.get_calendar("XNYS")
    for case in cases:
        for dest, rows in (
            (summary, [summary_row(case.book)]),
            (annual, annual_rows(case.book)),
            (months, monthly_rows(case.book)),
        ):
            dest.extend(
                {"key": case.key, "variant": case.variant, **row.model_dump(mode="json")}
                for row in rows
            )
        for target in case.targets:
            targets.extend(
                {
                    "key": case.key,
                    "variant": case.variant,
                    "day": str(target.day),
                    "symbol": s,
                    "account_weight": w * (1 if case.variant == "OVERLAY" else 0.95),
                }
                for s, w in target.q.items()
            )
            if case.variant == "OVERLAY":
                branches.append(
                    {
                        "key": case.key,
                        "decision_day": str(target.day),
                        "execution_day": str(
                            calendar.next_session(pd.Timestamp(target.day)).date()
                        ),
                        "branch": "QQQ_SELECTED" if target.q.get("QQQ") == 0.4 else "QQQ_OVERLAY",
                        "qqq_target": target.q.get("QQQ", 0),
                        "shy_target": target.q["SHY"],
                        "cash_target": 1 - sum(target.q.values()),
                    }
                )
        daily = read_csv(case.folder / "daily.csv.gz", DailyRecord)
        for year in (2022, 2023, 2024, 2025, None):
            selected = [d for d in daily if year is None or d.day.year == year]
            nav = {d.day: d.equity for d in selected}
            weights = {
                symbol: sum(
                    h.size * h.close / nav[h.day]
                    for h in case.holdings
                    if h.symbol == symbol and h.day in nav
                )
                / len(selected)
                for symbol in ("QQQ", "SHY", "VGT", "SOXX")
            }
            exposure.append(
                {
                    "key": case.key,
                    "variant": case.variant,
                    "cost_bps": case.book.contract.cost_bps,
                    "period": str(year) if year else "full2022_2025",
                    "mean_qqq_nav": weights["QQQ"],
                    "mean_shy_nav": weights["SHY"],
                    "mean_qqq_vgt_soxx_nav": sum(weights[s] for s in ("QQQ", "VGT", "SOXX")),
                    "mean_cash_nav": sum(d.cash / d.equity for d in selected) / len(selected),
                    "cash_includes_receivable": True,
                }
            )
    return [
        Table(name=name, keys=keys, frame=pd.DataFrame(rows))
        for name, keys, rows in (
            ("SUMMARY.csv", ("key", "period"), summary),
            ("ANNUAL.csv", ("key", "period"), annual),
            ("MONTHLY.csv", ("key", "period"), months),
            ("TARGETS.csv", ("key", "day", "symbol"), targets),
            ("EXPOSURE.csv", ("key", "period"), exposure),
            ("BRANCHES.csv", ("key", "decision_day"), branches),
        )
    ]


def build(root: Path, native_output: Path, output: Path) -> None:
    lock = verify_lock(root)
    cases = cases_for(root, native_output)
    result = tables(cases)
    output.mkdir(parents=True, exist_ok=True)
    checked = 0
    for table in result:
        file = output / table.name
        if not file.exists():
            table.frame.to_csv(file, index=False, float_format="%.17g")
        checked += compare_table(table, file)
    receipt = {
        "status": "PASS",
        "source_lock_sha256": lock,
        "science": 2,
        "controls": 6,
        "native_attempts": 0,
        "table_cells": checked,
        "archived_metric_values": sum(verify_book(c.book) for c in cases),
        "tables": {
            t.name: {
                "rows": len(t.frame),
                "sha256": hashlib.sha256((output / t.name).read_bytes()).hexdigest(),
            }
            for t in result
        },
    }
    file = output / "REPORT_VERIFICATION.json"
    if file.exists():
        if json.loads(file.read_text()) != receipt:
            raise ValueError("saved report receipt changed")
    else:
        file.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
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
