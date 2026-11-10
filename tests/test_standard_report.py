"""Reporting tests use saved NAV only; they never execute a native account."""

from datetime import date
from math import prod
from pathlib import Path

import pytest

from c10standard.models import Contract
from c10standard.report import (
    Book,
    Nav,
    annual_rows,
    load_books,
    monthly_rows,
    summarize,
    verify_book,
)


def book() -> Book:
    return Book(
        contract=Contract(cohort="continuous2022_2026"),
        initial_cash=50_000,
        daily=(
            Nav(day=date(2025, 1, 2), equity=45_000),
            Nav(day=date(2025, 12, 31), equity=60_000),
            Nav(day=date(2026, 1, 2), equity=54_000),
            Nav(day=date(2026, 1, 30), equity=66_000),
            Nav(day=date(2026, 2, 27), equity=72_000),
        ),
    )


def test_first_day_loss_and_year_opening_are_not_dropped() -> None:
    rows = annual_rows(book())
    assert [row.opening_equity for row in rows] == [50_000, 60_000]
    assert [row.total_return for row in rows] == pytest.approx([0.2, 0.2])
    assert [row.max_drawdown for row in rows] == pytest.approx([0.1, 0.1])
    assert rows[-1].cagr_252 is None


def test_continuous_ytd_uses_own_prior_nav_and_reset_uses_new_cash() -> None:
    continuous = summarize(book(), date(2026, 1, 2), date(2026, 2, 27), "2026_ytd")
    reset = book().model_copy(
        update={"contract": Contract(cohort="reset2026"), "daily": book().daily[2:]}
    )
    fresh = summarize(reset, date(2026, 1, 2), date(2026, 2, 27), "2026_ytd")
    assert continuous.opening_equity == 60_000
    assert fresh.opening_equity == 50_000
    assert continuous.total_return == pytest.approx(0.2)
    assert fresh.total_return == pytest.approx(0.44)
    assert continuous.cagr_252 is None and fresh.cagr_252 is None


def test_months_and_years_compound_to_the_same_continuous_account() -> None:
    months = monthly_rows(book())
    assert months[2].opening_equity == 60_000
    assert prod(1 + row.total_return for row in months) == pytest.approx(72_000 / 50_000)
    assert prod(1 + row.total_return for row in annual_rows(book())) == pytest.approx(1.44)


def test_blank_or_duplicate_nav_fails_before_a_report_can_claim_success() -> None:
    for rows in ((), (book().daily[0], book().daily[0])):
        invalid = book().model_copy(update={"daily": rows})
        with pytest.raises(ValueError, match="nonempty|increasing"):
            summarize(invalid, date(2025, 1, 2), date(2025, 12, 31), "2025")
    with pytest.raises(ValueError, match="empty"):
        summarize(book(), date(2024, 1, 1), date(2024, 12, 31), "2024")


def test_saved_standard_keeps_negative_top_one_reference_and_separate_2026_denominators() -> None:
    books, hashes = load_books(Path(__file__).resolve().parents[1])
    assert len(books) == 36 and hashes >= 288
    assert sum(verify_book(row) for row in books) == 1260
    selected = [row for row in books if row.contract.rule == "C10" and row.contract.cost_bps == 5]
    ytd = [
        summarize(row, date(2026, 1, 2), date(2026, 11, 6), "2026_ytd")
        for row in selected
        if not row.contract.terminal
    ]
    reset = next(row for row in ytd if row.cohort == "reset2026")
    continuous = next(row for row in ytd if row.cohort == "continuous2022_2026")
    assert reset.total_return == pytest.approx(-0.04429611211252127)
    assert reset.max_drawdown == pytest.approx(0.3795304348831363)
    assert reset.opening_equity == 50_000
    assert continuous.opening_equity == pytest.approx(145405.90101035297)
