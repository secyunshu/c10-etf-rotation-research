"""Offline verification rejects drift rather than silently rewriting published results."""

from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from c10experiments.report import (
    FAMILY,
    Case,
    Table,
    compare_table,
    excess_matrix,
    historical_returns,
    report_period,
)
from c10experiments.statistics import max_bootstrap
from c10standard.models import Contract
from c10standard.report import Book, Nav


def test_ytd_uses_explicit_case_capital_and_never_extrapolates_cagr() -> None:
    book = Book(
        contract=Contract(rule="C10", cohort="continuous2022_2026"),
        initial_cash=50_000,
        daily=(
            Nav(day=date(2025, 12, 31), equity=150_000),
            Nav(day=date(2026, 1, 2), equity=135_000),
            Nav(day=date(2026, 1, 5), equity=165_000),
        ),
    )
    row = report_period(book, "2026YTD")
    assert row.rule == "C10" and row.opening_equity == 150_000
    assert row.total_return == pytest.approx(0.1) and row.max_drawdown == pytest.approx(0.1)
    assert row.cagr_252 is None
    reset = book.model_copy(
        update={"contract": Contract(rule="C10", cohort="reset2026"), "daily": book.daily[1:]}
    )
    assert report_period(reset, "2026YTD").opening_equity == 50_000
    broken = book.model_copy(update={"daily": book.daily[1:]})
    with pytest.raises(ValueError, match="December 31"):
        report_period(broken, "2026YTD")


def test_historical_bootstrap_includes_first_day_and_rejects_future_dates() -> None:
    days = [
        x.date() for x in xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2025-12-31")
    ]
    book = Book(
        contract=Contract(rule="C11"),
        initial_cash=50_000,
        daily=tuple(Nav(day=day, equity=49_000) for day in days),
    )
    changes = historical_returns(book)
    assert changes.shape == (1003,) and changes[0] == pytest.approx(-0.02)
    assert np.all(changes[1:] == 0)
    assert FAMILY == ("E01", "E02", "E03", "E04", "E05", "E06")
    broken = book.model_copy(
        update={"daily": book.daily[:-1] + (Nav(day=date(2026, 1, 2), equity=49_000),)}
    )
    with pytest.raises(ValueError, match="1003"):
        historical_returns(broken)


def test_csv_verification_requires_case_identity_exact_columns_and_numeric_results(
    tmp_path: Path,
) -> None:
    table = Table(
        name="SUMMARY.csv",
        keys=("key", "period"),
        frame=pd.DataFrame(
            [{"key": "E01_5bp", "period": "2026YTD", "total_return": 0.1, "cagr_252": None}]
        ),
    )
    archive = tmp_path / "SUMMARY.csv"
    table.frame.to_csv(archive, index=False)
    assert compare_table(table, archive) == 4
    for content in (
        "key,period,total_return,cagr_252\nE02_5bp,2026YTD,0.1,\n",
        "key,period,total_return,cagr_252,extra\nE01_5bp,2026YTD,0.1,,1\n",
        "key,period,total_return,cagr_252\nE01_5bp,2026YTD,0.2,\n",
        "key,period,total_return,cagr_252\nE01_5bp,2026YTD,0.1,0.2\n",
    ):
        archive.write_text(content)
        with pytest.raises(ValueError):
            compare_table(table, archive)


def test_bootstrap_family_retains_all_six_including_identical_paths() -> None:
    days = [
        x.date() for x in xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2025-12-31")
    ]
    book = Book(
        contract=Contract(rule="C11"),
        initial_cash=50_000,
        daily=tuple(Nav(day=day, equity=50_000) for day in days),
    )
    cases = [
        Case(key=name, variant=name, folder=Path("."), book=book, targets=(), holdings=())
        for name in ("BASE252", *FAMILY)
    ]
    panel = excess_matrix(cases, 5)
    assert panel.shape == (1003, 6) and np.all(panel == 0)
    for changed in (cases[:-1], [*cases, cases[-1]]):
        with pytest.raises(ValueError, match="all six"):
            excess_matrix(changed, 5)
    result = max_bootstrap(panel, FAMILY, block=20, repetitions=20)
    assert result.adjusted_p == [1.0] * 6
    assert result.mean_bp == [0.0] * 6
    assert result.ci_low_bp == result.ci_high_bp == [0.0] * 6
