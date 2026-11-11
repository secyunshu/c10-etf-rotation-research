"""A new capital start cannot be confused with a slice of an older account."""

from datetime import date

import pytest

from rotbt.models import DailyRecord
from scripts.v2_2023_models import Contract
from scripts.v2_2023_report import validate_period


def test_new_start_report_rejects_old_capital_window_and_bridge_identity() -> None:
    contract = Contract(role="NEW_POOL", cost_bps=5, stage="science", pool_sha256="a" * 64)
    old = DailyRecord(
        day=date(2022, 1, 3),
        equity=50_000,
        cash=50_000,
        locked=0,
        available=50_000,
        positions=0,
        gross_exposure=0,
    )
    with pytest.raises(ValueError, match="calendar"):
        validate_period(contract, [old])
    with pytest.raises(ValueError, match="science"):
        validate_period(
            Contract(role="OLD_POOL", cost_bps=5, stage="bridge", pool_sha256="a" * 64), []
        )
