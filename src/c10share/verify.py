"""Verify saved ledgers and signals without starting a Backtrader account."""

import argparse
import json
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from rotbt.analysis import analyze
from rotbt.models import (
    ClosedTrade,
    DailyRecord,
    Distribution,
    HoldingRecord,
    Metrics,
    OrderRecord,
    Result,
    Spec,
)

from .actions import HistoricalRotation
from .io import ROOT, feed_names, load_accounts, load_market, load_task, weights_for
from .models import BOOKS, Account, Prepared, ReferenceCheck, RunComparison, TargetRecord, Verified


def compare_frames(expected: pd.DataFrame, actual: pd.DataFrame, label: str) -> float:
    """Require identical ordering/schema and at most 1e-10 absolute numeric drift."""
    if expected.shape != actual.shape or list(expected.columns) != list(actual.columns):
        raise ValueError(f"{label}: shape or columns mismatch")
    maximum = 0.0
    for column in expected:
        left, right = expected[column], actual[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"{label}/{column}: nonfinite numeric value")
            difference = float(np.max(np.abs(a - b))) if len(a) else 0.0
            maximum = max(maximum, difference)
            if difference > 1e-10:
                raise ValueError(f"{label}/{column}: numeric mismatch; max_abs={difference:.17g}")
        elif not left.equals(right):
            raise ValueError(f"{label}/{column}: exact categorical mismatch")
    return maximum


def read_book(folder: Path, book: str) -> pd.DataFrame:
    return pd.read_csv(folder / f"{book}.csv.gz")


def load_reference(account: Account, prepared: Prepared, root: Path = ROOT) -> Result:
    """Reconstruct typed broker exports for native-account reconciliation."""
    task = load_task(account, root)
    folder = root / "data/reference" / account.name
    verified = Verified.model_validate_json((folder / "verified.json").read_text())
    tables = {name: read_book(folder, name).to_dict("records") for name in BOOKS}
    return Result(
        spec=Spec(name=task.spec.sid, layer="hold", lookback=252, top=3),
        daily=TypeAdapter(list[DailyRecord]).validate_python(tables["daily"]),
        orders=TypeAdapter(list[OrderRecord]).validate_python(tables["orders"]),
        holdings=TypeAdapter(list[HoldingRecord]).validate_python(tables["holdings"]),
        distributions=TypeAdapter(list[Distribution]).validate_python(tables["dividends"]),
        trades=TypeAdapter(list[ClosedTrade]).validate_python(tables["trades"]),
        native_total_return=verified.native_return,
        splits=[
            event
            for data in prepared.datasets
            if data.symbol in feed_names(account.rule)
            for event in data.splits
            if task.start <= event.day <= task.end
        ],
    )


def verify_reference(account: Account, prepared: Prepared, root: Path = ROOT) -> ReferenceCheck:
    task = load_task(account, root)
    result = load_reference(account, prepared, root)
    folder = root / "data/reference" / account.name
    saved_metrics = TypeAdapter(list[Metrics]).validate_json((folder / "metrics.json").read_text())
    metrics = analyze(result, task.settings)
    difference = compare_frames(
        pd.DataFrame([m.model_dump() for m in saved_metrics]),
        pd.DataFrame([m.model_dump() for m in metrics]),
        "metrics",
    )
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(task.start, task.end)
    if [row.day for row in result.daily] != [stamp.date() for stamp in sessions]:
        raise ValueError("reference daily calendar mismatch")
    expected_days = [calendar.previous_session(pd.Timestamp(task.start)).date()]
    if not account.rule.endswith("_BH"):
        expected_days += [
            stamp.date()
            for stamp in sessions
            if stamp.date() < task.end and HistoricalRotation.scheduled(stamp.date(), "M", ())
        ]
    records = TypeAdapter(list[TargetRecord]).validate_json((folder / "targets.json").read_text())
    if [record.day for record in records] != expected_days:
        raise ValueError("target decision calendar mismatch")
    target_difference = 0.0
    for i, record in enumerate(records):
        expected_q = weights_for(account.rule, prepared, record.day)
        if (
            record.reason != ("initial" if i == 0 else "rebalance")
            or record.q.keys() != expected_q.keys()
        ):
            raise ValueError("target identity/reason mismatch")
        if not all(np.isfinite(value) for value in (*record.q.values(), *expected_q.values())):
            raise ValueError(f"nonfinite target weight: {record.day}")
        delta = max(abs(record.q[name] - value) for name, value in expected_q.items())
        target_difference = max(target_difference, delta)
        if delta > 1e-10:
            raise ValueError(f"target numeric mismatch: {record.day}")
    verified = Verified.model_validate_json((folder / "verified.json").read_text())
    if result.daily[-1].positions != 0:
        raise ValueError("terminal account not liquidated")
    compare_json(
        verified.model_dump(),
        {
            "status": "PASS",
            "native_return": result.native_total_return,
            "equity_return": next(
                metric.total_return for metric in metrics if metric.period == "full"
            ),
            "days": len(result.daily),
            "orders": len(result.orders),
            "cash_events": len(result.distributions),
            "min_available": min(row.available for row in result.daily),
            "final_locked": result.daily[-1].locked,
            "terminal_positions": 0,
        },
        "verified",
    )
    return ReferenceCheck(
        account=account.name,
        days=len(result.daily),
        orders=len(result.orders),
        targets=len(records),
        max_metric_difference=difference,
        max_target_difference=target_difference,
    )


def compare_run(account: Account, output: Path, root: Path = ROOT) -> RunComparison:
    """Compare all economic outputs; gzip container timestamps are immaterial."""
    reference = root / "data/reference" / account.name
    expected_files = {f"{book}.csv.gz" for book in BOOKS} | {
        "metrics.json",
        "targets.json",
        "task.json",
        "verified.json",
    }
    if {p.name for p in output.iterdir()} != expected_files:
        raise ValueError("missing or extra native output files")
    differences = {
        book: compare_frames(read_book(reference, book), read_book(output, book), book)
        for book in BOOKS
    }
    for name in ("metrics", "targets", "task", "verified"):
        expected: Any = json.loads((reference / f"{name}.json").read_text())
        actual: Any = json.loads((output / f"{name}.json").read_text())
        differences[name] = compare_json(expected, actual, name)
    return RunComparison(account=account.name, max_abs_difference=differences)


def compare_json(expected: Any, actual: Any, label: str) -> float:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"{label}: missing or extra keys")
        return max(
            (compare_json(value, actual[key], f"{label}/{key}") for key, value in expected.items()),
            default=0.0,
        )
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"{label}: sequence length mismatch")
        return max(
            (compare_json(a, b, label) for a, b in zip(expected, actual, strict=True)), default=0.0
        )
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        delta = abs(float(expected) - float(actual))
        if not np.isfinite(delta) or delta > 1e-10:
            raise ValueError(f"{label}: numeric mismatch; max_abs={delta:.17g}")
        return delta
    if type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"{label}: exact value mismatch")
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    prepared = load_market(args.root)
    results = [
        verify_reference(account, prepared, args.root).model_dump(mode="json")
        for account in load_accounts(args.root)
    ]
    print(json.dumps({"status": "PASS", "native_attempts": 0, "accounts": results}, indent=2))


if __name__ == "__main__":
    main()
