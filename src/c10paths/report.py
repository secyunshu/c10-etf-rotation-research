"""Describe the three preregistered path hypotheses using saved native books only."""

import argparse
import hashlib
import json
from datetime import date
from math import isclose
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype
from numpy.typing import NDArray
from pydantic import TypeAdapter

from c10experiments.report import (
    Case,
    Fingerprints,
    Table,
    compare_table,
    historical_returns,
    report_period,
)
from c10experiments.statistics import max_bootstrap
from c10share.io import verify_hashes
from c10share.report import read_csv
from c10standard.audit import Description
from c10standard.io import ROOT
from c10standard.models import COHORTS, Contract, NativeScalar, Receipt, TargetRecord
from c10standard.report import Book, Nav, load_books, monthly_rows, verify_book
from rotbt.models import DailyRecord, Distribution, HoldingRecord, Metrics, OrderRecord

from .io import Binding, Trial, verify_lock

STUDY = Path("experiments/wechat-momentum-paths")
FAMILY = ("P01", "P02", "P03")
CONTROLS = ("C11", "SPY_BH", "QQQ_BH", "SPY60_IEF40")


@beartype
def validate_daily(contract: Contract, daily: tuple[DailyRecord, ...]) -> None:
    expected = [d.date() for d in xcals.get_calendar("XNYS").sessions_in_range(
        contract.start, contract.through)]
    if [r.day for r in daily] != expected:
        raise ValueError("saved native calendar does not match full contract")
    if contract.terminal and daily[-1].positions != 0:
        raise ValueError("historical terminal account must be liquidated")


@beartype
def load_case(variant: str, contract: Contract, folder: Path, root: Path = ROOT) -> Case:
    identity = Binding.model_validate_json((folder / "BINDING.json").read_bytes())
    expected = Trial.model_validate({"variant": variant, "contract": contract})
    if identity.trial != expected:
        raise ValueError("report signal identity mismatch")
    if identity.source_lock_sha256 != hashlib.sha256(
        (root / STUDY / "SOURCE_LOCK.json").read_bytes()
    ).hexdigest():
        raise ValueError("report source lock binding mismatch")
    if (folder / "REFERENCE.json").exists():
        binding = Fingerprints.model_validate_json((folder / "REFERENCE.json").read_bytes())
        if "BINDING.json" not in binding.hashes:
            raise ValueError("reference must hash the outer signal binding")
        verify_hashes(folder, binding.hashes)
    else:
        receipt = Receipt.model_validate_json((folder / "VERIFIED.json").read_bytes())
        verify_hashes(folder, receipt.hashes)
    actual = Contract.model_validate_json((folder / "CONTRACT.json").read_bytes())
    if actual != contract:
        raise ValueError("report native contract mismatch")
    daily = read_csv(folder / "daily.csv.gz", DailyRecord)
    validate_daily(contract, daily)
    scalar = NativeScalar.model_validate_json((folder / "native_total_return.json").read_bytes())
    if not daily or not isclose(daily[-1].equity / 50_000 - 1, scalar.native_total_return,
                               abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError("saved native scalar differs from NAV")
    book = Book(
        contract=contract, initial_cash=50_000,
        daily=tuple(Nav(day=r.day, equity=r.equity) for r in daily),
        orders=read_csv(folder / "orders.csv.gz", OrderRecord),
        dividends=read_csv(folder / "dividends.csv.gz", Distribution),
        archived_metrics=TypeAdapter(tuple[Metrics, ...]).validate_json(
            (folder / "metrics.json").read_bytes()) if contract.terminal else (),
        archived_ytd=None if contract.terminal else Description.model_validate_json(
            (folder / "ytd.json").read_bytes()),
    )
    verify_book(book)
    return Case(
        key=f"{variant}_{contract.cost_bps}bp_{contract.cohort}", variant=variant,
        folder=folder, book=book,
        targets=TypeAdapter(tuple[TargetRecord, ...]).validate_json(
            (folder / "targets.json").read_bytes()),
        holdings=read_csv(folder / "holdings.csv.gz", HoldingRecord),
    )


def load_cases(root: Path, native_output: Path) -> list[Case]:
    cases = []
    for variant in FAMILY:
        for cohort in COHORTS:
            for cost in (5, 20):
                contract = Contract(cost_bps=cost, cohort=cohort)
                cases.append(load_case(variant, contract,
                                       native_output / variant / contract.key, root))
    controls, _ = load_books(root)
    for book in controls:
        if book.contract.rule not in CONTROLS:
            continue
        verify_book(book)
        folder = root / "standard/reference" / book.contract.key
        cases.append(Case(
            key=book.contract.key,
            variant="BASE252" if book.contract.rule == "C11" else book.contract.rule,
            folder=folder, book=book,
            targets=TypeAdapter(tuple[TargetRecord, ...]).validate_json(
                (folder / "targets.json").read_bytes()),
            holdings=read_csv(folder / "holdings.csv.gz", HoldingRecord),
        ))
    if len(cases) != 42 or len({c.key for c in cases}) != 42:
        raise ValueError("exactly18 new and24 reused accounts required")
    return cases


@beartype
def period_table(cases: list[Case], *, annual: bool) -> Table:
    rows = []
    for case in cases:
        periods = (
            [str(y) for y in sorted({r.day.year for r in case.book.daily})]
            if annual else ["full2022_2025" if case.book.contract.terminal else "2026YTD"]
        )
        for period in periods:
            row = report_period(case.book, period)
            rows.append({"key": case.key, "variant": case.variant, **row.model_dump(mode="json")})
    return Table(name="ANNUAL.csv" if annual else "SUMMARY.csv", keys=("key", "period"),
                 frame=pd.DataFrame(rows))


@beartype
def excess_matrix(cases: list[Case], cost: int) -> NDArray[np.float64]:
    arrays = []
    for variant in ("BASE252", *FAMILY):
        matches = [c for c in cases if c.variant == variant
                   and c.book.contract.cost_bps == cost
                   and c.book.contract.cohort == "historical2022_2025"]
        if len(matches) != 1:
            raise ValueError("each family member needs exactly one matching historical account")
        arrays.append(historical_returns(matches[0].book))
    return np.asarray(np.column_stack(arrays[1:]) - arrays[0][:, None], dtype=float)


def tables(cases: list[Case]) -> list[Table]:
    output = [period_table(cases, annual=False), period_table(cases, annual=True)]
    months: list[dict[str, Any]] = []
    allocation: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for case in cases:
        months.extend({"key": case.key, "variant": case.variant, **r.model_dump(mode="json")}
                      for r in monthly_rows(case.book))
        first = case.book.contract.start if case.book.contract.terminal else date(2026, 1, 2)
        nav = {r.day: r.equity for r in case.book.daily if r.day >= first}
        shy = {h.day: h.size * h.close / nav[h.day] for h in case.holdings
               if h.symbol == "SHY" and h.day in nav}
        allocation.append({"key": case.key, "variant": case.variant,
                           "mean_daily_shy_nav_weight": sum(shy.values()) / len(nav),
                           "days": len(nav)})
        for target in case.targets:
            targets.extend({"key": case.key, "variant": case.variant, "day": str(target.day),
                            "symbol": symbol, "predeployment_weight": weight,
                            "target_weight": 0.95 * weight}
                           for symbol, weight in target.q.items())
    output.extend([
        Table(name="MONTHLY.csv", keys=("key", "period"), frame=pd.DataFrame(months)),
        Table(name="SHY_EXPOSURE.csv", keys=("key",), frame=pd.DataFrame(allocation)),
        Table(name="TARGETS.csv", keys=("key", "day", "symbol"), frame=pd.DataFrame(targets)),
    ])
    statistics = []
    for cost in (5, 20):
        differences = excess_matrix(cases, cost)
        for block in (10, 21, 63):
            result = max_bootstrap(differences, FAMILY, block=block, repetitions=10_000)
            for i, variant in enumerate(FAMILY):
                statistics.append({"cost_bps": cost, "block": block, "variant": variant,
                                   "family_p": result.family_p,
                                   "adjusted_p": result.adjusted_p[i],
                                   "mean_daily_excess_bp": result.mean_bp[i],
                                   "pointwise_low_daily_bp": result.ci_low_bp[i],
                                   "pointwise_high_daily_bp": result.ci_high_bp[i]})
    output.append(Table(name="LOCAL_BOOTSTRAP.csv", keys=("cost_bps", "block", "variant"),
                        frame=pd.DataFrame(statistics)))
    return output


def build(root: Path, native_output: Path, output: Path) -> None:
    verify_lock(root)
    cases = load_cases(root, native_output)
    result = tables(cases)
    output.mkdir(parents=True, exist_ok=True)
    checked = 0
    for table in result:
        path = output / table.name
        if path.exists():
            checked += compare_table(table, path)
        else:
            table.frame.to_csv(path, index=False, float_format="%.17g")
            checked += compare_table(table, path)
    receipt: dict[str, Any] = {
        "status": "PASS", "new_accounts": 18, "reused_controls": 24,
        "native_attempts_by_report": 0, "bootstrap_runs": 6,
        "bootstrap_repetitions_each": 10000, "checked_table_cells": checked,
        "checked_archived_statistics": sum(verify_book(c.book) for c in cases),
        "tables": {t.name: {"rows": len(t.frame), "sha256": hashlib.sha256(
            (output / t.name).read_bytes()).hexdigest()} for t in result},
    }
    path = output / "REPORT_VERIFICATION.json"
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise ValueError("immutable report receipt changed")
    else:
        path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.root, args.native_output or args.root / STUDY / "reference",
          args.output or args.root / STUDY / "results")


if __name__ == "__main__":
    main()
