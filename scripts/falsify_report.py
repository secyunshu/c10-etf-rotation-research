"""Finite factorial wealth contrasts and path equality from saved native books."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from c10experiments.report import Table, compare_table
from c10share.io import verify_hashes
from c10share.report import read_csv
from c10share.verify import compare_frames, compare_json
from c10standard.io import load_accounts, reference_folder
from c10standard.models import BOOKS, Contract, NativeScalar, Receipt, TargetRecord
from c10standard.report import Book, Nav, annual_rows, summary_row, verify_book
from rotbt.models import DailyRecord, Distribution, Metrics, Model, OrderRecord
from scripts.falsify_io import ROOT, STUDY, verify_lock
from scripts.falsify_models import Account, CaseBinding, baseline_schedules, catalog, load_cases


class Loaded(Model):
    case: str
    folder: Path
    book: Book
    targets: tuple[TargetRecord, ...]

    @property
    def key(self) -> str:
        return f"{self.case}_{self.book.contract.cost_bps}bp"


def contrasts(summary: pd.DataFrame, cost: int) -> pd.DataFrame:
    wealth = {}
    for case in "ABCDE":
        matched = summary[(summary["case"] == case) & (summary.cost_bps == cost)]
        if len(matched) != 1:
            raise ValueError("exactly one final wealth per case and cost required")
        wealth[case] = float(matched.final_equity.iloc[0])
    pairs = ("B-A", "D-C", "C-A", "D-B", "D-E")
    rows = [
        {"cost_bps": cost, "contrast": pair, "wealth_difference": wealth[pair[0]] - wealth[pair[2]]}
        for pair in pairs
    ]
    rows.append(
        {
            "cost_bps": cost,
            "contrast": "D-B-C+A",
            "wealth_difference": wealth["D"] - wealth["B"] - wealth["C"] + wealth["A"],
        }
    )
    return pd.DataFrame(rows)


def load_one(
    case: str, folder: Path, contract: Contract, root: Path, account: Account | None = None
) -> Loaded:
    if account is not None:
        saved = CaseBinding.model_validate_json((folder / "CASE_BINDING.json").read_bytes())
        if saved.account != account or saved.source_sha256 != verify_lock(root):
            raise ValueError("counterfactual outer identity/source mismatch")
    if (folder / "REFERENCE.json").exists():
        reference = json.loads((folder / "REFERENCE.json").read_text())
        if account is not None and "CASE_BINDING.json" not in reference["hashes"]:
            raise ValueError("reference lacks counterfactual identity")
        verify_hashes(folder, reference["hashes"])
    elif account is not None:
        receipt = Receipt.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, receipt.hashes)
    daily = read_csv(folder / "daily.csv.gz", DailyRecord)
    expected = [
        d.date() for d in xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2025-12-31")
    ]
    if [d.day for d in daily] != expected or daily[-1].positions:
        raise ValueError("complete fouryear calendar and terminal liquidation required")
    scalar = NativeScalar.model_validate_json((folder / "native_total_return.json").read_bytes())
    if not np.isclose(
        daily[-1].equity / 50000 - 1, scalar.native_total_return, rtol=1e-10, atol=1e-10
    ):
        raise ValueError("native return differs from NAV")
    book = Book(
        contract=contract,
        initial_cash=50000,
        daily=tuple(Nav(day=d.day, equity=d.equity) for d in daily),
        orders=read_csv(folder / "orders.csv.gz", OrderRecord),
        dividends=read_csv(folder / "dividends.csv.gz", Distribution),
        archived_metrics=TypeAdapter(tuple[Metrics, ...]).validate_json(
            (folder / "metrics.json").read_bytes()
        ),
    )
    verify_book(book)
    targets = TypeAdapter(tuple[TargetRecord, ...]).validate_json(
        (folder / "targets.json").read_bytes()
    )
    return Loaded(case=case, folder=folder, book=book, targets=targets)


def loaded_cases(root: Path, output: Path) -> list[Loaded]:
    result = []
    policies = load_cases(root)
    for account in catalog(policies, *baseline_schedules(root)):
        if account.stage == "science":
            result.append(
                load_one(
                    account.case.id,
                    output / account.key,
                    Contract(cost_bps=account.cost_bps),
                    root,
                    account,
                )
            )
    references = load_accounts(root)
    for cost in (5, 20):
        for case, rule in (("A", "C11"), ("SPY", "SPY_BH"), ("QQQ", "QQQ_BH")):
            reference_account = next(
                a
                for a in references
                if a.rule == rule and a.cost_bps == cost and a.cohort == "historical2022_2025"
            )
            result.append(
                load_one(
                    case,
                    reference_folder(reference_account, root),
                    reference_account.contract,
                    root,
                )
            )
        folder = root / "experiments/v2-annual-repool/reference" / f"ANNUAL_{cost}bp_science"
        result.append(load_one("D", folder, Contract(cost_bps=cost), root))
    if len(result) != 14 or len({r.key for r in result}) != 14:
        raise ValueError("14 unique report cases required")
    return result


def tables(cases: list[Loaded]) -> list[Table]:
    summaries: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    for case in cases:
        row = summary_row(case.book)
        summaries.append(
            {
                "key": case.key,
                "case": case.case,
                **row.model_dump(mode="json"),
                "final_equity": row.closing_equity,
            }
        )
        annual.extend(
            {"key": case.key, "case": case.case, **r.model_dump(mode="json")}
            for r in annual_rows(case.book)
        )
    summary = pd.DataFrame(summaries)
    paired = pd.concat([contrasts(summary, c) for c in (5, 20)], ignore_index=True)
    equality = []
    for cost in (5, 20):
        mapping = {r.case: r for r in cases if r.book.contract.cost_bps == cost}
        for left, right in (("B", "D"), ("C", "A"), ("D", "E")):
            a, b = mapping[left], mapping[right]
            nav_a = np.asarray([x.equity for x in a.book.daily])
            nav_b = np.asarray([x.equity for x in b.book.daily])
            target_difference = sum(
                x.day != y.day or x.q != y.q for x, y in zip(a.targets, b.targets, strict=True)
            )
            rows_equal = True
            differences: dict[str, float | None] = {}
            for name in BOOKS:
                first = pd.read_csv(a.folder / f"{name}.csv.gz", float_precision="round_trip")
                second = pd.read_csv(b.folder / f"{name}.csv.gz", float_precision="round_trip")
                # Feed unions can add explicit zero holdings without economic impact.
                if name == "holdings":
                    first, second = first[first["size"] != 0], second[second["size"] != 0]
                try:
                    differences[name] = compare_frames(
                        first.reset_index(drop=True), second.reset_index(drop=True), name
                    )
                except ValueError:
                    rows_equal = False
                    differences[name] = None
            equality.append(
                {
                    "cost_bps": cost,
                    "comparison": f"{left}_vs_{right}",
                    "different_target_dates": target_difference,
                    "daily_nav_exact": bool(np.array_equal(nav_a, nav_b)),
                    "max_nav_difference": float(np.max(np.abs(nav_a - nav_b))),
                    "economic_five_books_equal": rows_equal,
                    **{f"{k}_max_difference": v for k, v in differences.items()},
                }
            )
    return [
        Table(name="SUMMARY.csv", keys=("key",), frame=summary),
        Table(name="ANNUAL.csv", keys=("key", "period"), frame=pd.DataFrame(annual)),
        Table(name="WEALTH_CONTRASTS.csv", keys=("cost_bps", "contrast"), frame=paired),
        Table(
            name="PATH_EQUALITY.csv", keys=("cost_bps", "comparison"), frame=pd.DataFrame(equality)
        ),
    ]


def build(root: Path, native_output: Path, output: Path) -> None:
    lock = verify_lock(root)
    cases = loaded_cases(root, native_output)
    output.mkdir(parents=True, exist_ok=True)
    cells = 0
    hashes = {}
    for table in tables(cases):
        file = output / table.name
        if not file.exists():
            table.frame.to_csv(file, index=False, float_format="%.17g")
        cells += compare_table(table, file)
        hashes[table.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    receipt = {
        "status": "PASS",
        "source_lock": lock,
        "roles": len(cases),
        "new_accounts": 6,
        "reused_accounts": 8,
        "native_attempts": 0,
        "table_cells": cells,
        "metric_values": sum(verify_book(c.book) for c in cases),
        "table_hashes": hashes,
    }
    path = output / "REPORT_VERIFICATION.json"
    if path.exists():
        compare_json(json.loads(path.read_text()), receipt, "immutable receipt")
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
