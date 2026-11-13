import numpy as np
import pandas as pd
import pytest

from scripts.wechat_followup import high_age, session_returns


def test_split_and_distribution_do_not_look_like_overnight_crashes() -> None:
    frame = pd.DataFrame({"open": [100., 50., 48.], "close": [100., 50., 49.],
                          "tr": [100., 100., 102.]})
    parts = session_returns(frame)
    assert parts.overnight.iloc[1] == pytest.approx(0.)
    assert parts.overnight.iloc[2] == pytest.approx(np.log(1.02) - np.log(49 / 48))
    assert (parts.overnight + parts.intraday).iloc[2] == pytest.approx(np.log(1.02))


def test_equal_highs_use_the_most_recent_peak_and_ignore_the_future() -> None:
    prices = pd.Series([1., 3., 2., 3., 2., 4.])
    assert high_age(prices, 4).iloc[-2] == -1.
    assert high_age(prices.iloc[:-1], 4).iloc[-1] == -1.
