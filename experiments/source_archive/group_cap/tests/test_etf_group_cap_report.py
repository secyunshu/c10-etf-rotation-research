"""Reporting must retain prior closing capital and losses on the first session."""

from datetime import date

import pandas as pd
import pytest

from scripts.etf_group_cap_report import summarize


def test_first_session_loss_is_part_of_return_and_drawdown() -> None:
    daily = pd.DataFrame(
        {"equity": [47_500.0, 49_000.0, 51_000.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
    )
    orders = pd.DataFrame({"day": ["2026-01-02"], "size": [100], "price": [100.0], "fee": [2.0]})
    result = summarize(daily, orders, date(2026, 1, 2), date(2026, 1, 6), 50_000.0)
    assert result.total_return == pytest.approx(0.02)
    assert result.max_drawdown == pytest.approx(0.05)
    assert result.fees == 2.0


def test_continuous_year_uses_own_prior_close_and_excludes_previous_orders() -> None:
    daily = pd.DataFrame(
        {"equity": [150_000.0, 135_000.0, 165_000.0]},
        index=pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05"]),
    )
    orders = pd.DataFrame(
        {
            "day": ["2025-12-31", "2026-01-02"],
            "size": [100, 200],
            "price": [100.0, 110.0],
            "fee": [5.0, 3.0],
        }
    )
    result = summarize(daily, orders, date(2026, 1, 2), date(2026, 1, 5), 150_000.0)
    assert result.total_return == pytest.approx(0.10)
    assert result.max_drawdown == pytest.approx(0.10)
    assert result.fees == 3.0
    assert result.orders == 1


def test_missing_observations_do_not_become_a_zero_return_result() -> None:
    daily = pd.DataFrame({"equity": [50_000.0]}, index=pd.to_datetime(["2025-12-31"]))
    orders = pd.DataFrame(columns=["day", "size", "price", "fee"])
    with pytest.raises(ValueError, match="empty"):
        summarize(daily, orders, date(2026, 1, 2), date(2026, 1, 5), 50_000.0)
