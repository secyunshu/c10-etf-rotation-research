"""Four-year reporting preserves continuous starting capital and excludes later years."""

from datetime import date

import pytest

from rotbt.models import DailyRecord
from scripts.annual_report import validate_days


def test_account_summary_requires_full_four_year_calendar() -> None:
    short = [
        DailyRecord(
            day=date(2023, 1, 3),
            equity=50_000,
            cash=50_000,
            locked=0,
            available=50_000,
            positions=0,
            gross_exposure=0,
        )
    ]
    with pytest.raises(ValueError, match="calendar"):
        validate_days(short)


def test_future_sample_is_rejected_before_annual_return_reporting() -> None:
    future = [
        DailyRecord(
            day=date(2026, 1, 2),
            equity=60_000,
            cash=60_000,
            locked=0,
            available=60_000,
            positions=0,
            gross_exposure=0,
        )
    ]
    with pytest.raises(ValueError, match="calendar"):
        validate_days(future)
