from datetime import date

import numpy as np
import pandas as pd
import pytest

from rotbt.models import Dataset, SplitEvent
from scripts.intramonth_cycle import labels, log_returns, split_close


def test_pretom_is_six_days_and_disjoint_from_last_four() -> None:
    days = labels(21)
    assert days[11:17] == ["pretom"] * 6
    assert days[-4:] == ["turn"] * 4
    assert days.count("other") == 11
    assert labels(21, shift=1)[12:18] == ["pretom"] * 6


def test_first_session_keeps_opening_capital_as_return_base() -> None:
    nav = pd.Series([51_000., 50_500.])
    result = log_returns(nav, opening=50_000.)
    assert result.iloc[0] == pytest.approx(np.log(1.02))
    assert np.expm1(result.sum()) == pytest.approx(.01)


def test_high_ratio_ignores_a_split_and_a_later_split_cannot_rewrite_prefix() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    data = Dataset(symbol="ETF", frame=pd.DataFrame({"close": [100., 50., 25.]}, index=dates),
                   dividends=[], splits=[
                       SplitEvent(symbol="ETF", day=date(2025, 1, 3),
                                  shares_after_per_before=2., source="test"),
                       SplitEvent(symbol="ETF", day=date(2025, 1, 6),
                                  shares_after_per_before=2., source="test"),
                   ])
    assert split_close(data, dates[1]).tolist() == [100., 100.]
