"""A new signal must not win by changing the capital base or comparison sample."""

from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pytest

from c10experiments.report import Case
from c10paths.io import Binding, Trial
from c10paths.report import excess_matrix, load_case, period_table, validate_daily
from c10share.report import read_csv
from c10standard.io import ROOT
from c10standard.models import Contract
from c10standard.report import Book, Nav
from rotbt.models import DailyRecord


def history(variant: str, value: float, cost: int = 5) -> Case:
    contract = Contract(cost_bps=cost)
    dates = xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2025-12-31")
    book = Book(
        contract=contract,
        initial_cash=50_000,
        daily=tuple(Nav(day=d.date(), equity=value) for d in dates),
    )
    return Case(
        key=f"{variant}_{cost}", variant=variant, folder=Path("unused"), book=book,
        targets=(), holdings=(),
    )


def test_fixed_three_way_excess_keeps_first_day_loss_and_identical_candidate() -> None:
    cases = [history("BASE252", 50_000)]
    cases += [history("P01", 49_000), history("P02", 50_000), history("P03", 51_000)]
    matrix = excess_matrix(list(reversed(cases)), 5)
    assert matrix.shape == (1003, 3)
    assert matrix[0] == pytest.approx([-0.02, 0, 0.02])
    assert np.all(matrix[1:] == 0)
    with pytest.raises(ValueError, match="exactly one"):
        excess_matrix(cases[:-1], 5)
    with pytest.raises(ValueError, match="exactly one"):
        excess_matrix(cases + [cases[-1]], 5)


def test_other_costs_and_continuous_copies_cannot_inflate_family_sample() -> None:
    cases = [history(v, 50_000) for v in ("BASE252", "P01", "P02", "P03")]
    extra = history("P01", 10_000, 20)
    continuous = cases[1].model_copy(update={"book": cases[1].book.model_copy(update={
        "contract": Contract(cohort="continuous2022_2026")
    })})
    assert np.all(excess_matrix(cases + [extra, continuous], 5) == 0)


def test_continuous_ytd_uses_own_prior_nav_and_never_reports_short_period_cagr() -> None:
    book = Book(
        contract=Contract(cohort="continuous2022_2026"), initial_cash=50_000,
        daily=(Nav(day=date(2025, 12, 31), equity=150_000),
               Nav(day=date(2026, 1, 2), equity=135_000),
               Nav(day=date(2026, 11, 6), equity=165_000)),
    )
    case = Case(key="P01_continuous", variant="P01", folder=Path("unused"), book=book,
                targets=(), holdings=())
    table = period_table([case], annual=False)
    row = table.frame.iloc[0]
    assert row.opening_equity == 150_000
    assert row.total_return == pytest.approx(0.1)
    assert row.max_drawdown == pytest.approx(0.1)
    assert row.cagr_252 is None
    bad = case.model_copy(update={"book": book.model_copy(update={"daily": book.daily[1:]})})
    with pytest.raises(ValueError, match="December 31"):
        period_table([bad], annual=False)


def test_complete_other_signal_folder_cannot_be_relabelled_as_the_winner(tmp_path: Path) -> None:
    # All variants share C11's native contract; that shell cannot establish signal identity.
    binding = Binding(trial=Trial(variant="P02", contract=Contract()),
                      source_lock_sha256="a" * 64)
    (tmp_path / "BINDING.json").write_text(binding.model_dump_json())
    with pytest.raises(ValueError, match="signal identity"):
        load_case("P01", Contract(), tmp_path)


@pytest.mark.parametrize("cohort", ["historical2022_2025", "reset2026", "continuous2022_2026"])
def test_reporting_rejects_missing_days_even_if_metrics_and_hashes_are_self_consistent(
    cohort: str,
) -> None:
    contract = Contract(cohort=cohort)
    daily = read_csv(ROOT / "standard/reference" / contract.key / "daily.csv.gz", DailyRecord)
    validate_daily(contract, daily)
    for corrupt in (daily[1:], daily[:1] + daily, daily[:-1]):
        with pytest.raises(ValueError, match="calendar"):
            validate_daily(contract, corrupt)


def test_historical_open_liquidation_cannot_be_reported_as_a_still_invested_account() -> None:
    contract = Contract()
    daily = read_csv(ROOT / "standard/reference" / contract.key / "daily.csv.gz", DailyRecord)
    corrupt = daily[:-1] + (daily[-1].model_copy(update={"positions": 1}),)
    with pytest.raises(ValueError, match="liquidated"):
        validate_daily(contract, corrupt)
