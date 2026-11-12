import numpy as np
import pandas as pd
import pytest

from scripts.adaptive_momentum import adaptive_days, normalized_ranges, recent_drop, slot_weights


def test_relative_range_switches_only_when_recent_volatility_increases() -> None:
    steady = np.full(63, .01)
    assert adaptive_days(steady) == 252
    steady[-20:] = .02
    assert adaptive_days(steady) == 126
    steady[-20:] = .005
    assert adaptive_days(steady) == 252


@pytest.mark.parametrize("recent", [(.003, -.040, .015, .010),
                                     (.003, -.030, .001, .003),
                                     (.003, -.016, -.016, -.016),
                                     (-.016, -.016, -.016, .003)])
def test_shock_and_both_recent_losing_streaks_can_block_a_candidate(
    recent: tuple[float, ...]
) -> None:
    stable = np.resize(np.array([-.01, .01]), 63)
    prices = 100 * np.exp(np.r_[0., np.cumsum(np.r_[stable, recent])])
    assert recent_drop(prices)


def test_orderly_decline_is_not_a_shock() -> None:
    stable = np.resize(np.array([-.01, .01]), 63)
    prices = 100 * np.exp(np.r_[0., np.cumsum(np.r_[stable, -.004, -.004, -.004, -.004])])
    assert not recent_drop(prices)


def test_split_does_not_create_a_false_volatility_jump() -> None:
    raw = pd.DataFrame({"close": [100., 50.], "high": [101., 51.], "low": [99., 49.]})
    assert normalized_ranges(raw, np.array([100., 100.])) == pytest.approx([.04])


def test_filtered_empty_slot_preserves_conditional_sleeve() -> None:
    assert slot_weights(("SMH",)) == {"SMH": .45, "QQQ": .10, "SHY": .45}
    assert slot_weights(("SPY",)) == {"SPY": .45, "SHY": .55}
