from datetime import date
from math import exp

import numpy as np
import pytest

from c10standard.models import TargetRecord
from scripts.v23_band_trend import allow_band, short_score


def target(symbols: tuple[str, str, str]) -> TargetRecord:
    return TargetRecord(day=date(2024, 1, 31),
                        q=dict(zip(symbols, (.45, .45, .10), strict=True)), reason="oracle")


def test_band_cannot_interfere_with_new_picks_or_missing_holding() -> None:
    origin = target(("XLE", "SMH", "QQQ"))
    held = frozenset(origin.q)
    assert allow_band(origin, target(("SMH", "XLE", "QQQ")), held)
    assert not allow_band(origin, target(("XLE", "XLV", "QQQ")), held)
    assert not allow_band(None, origin, held)
    assert not allow_band(origin, origin, frozenset(("XLE", "SMH")))
    assert not allow_band(origin, origin, held | {"SPY"})
    assert not allow_band(origin, origin.model_copy(update={"q": {
        "XLE": .44, "SMH": .46, "QQQ": .10,
    }}), held)


def test_scores_measure_same_25_closes_and_preserve_downtrend_sign() -> None:
    path = tuple(100 * exp(.001 * t) for t in range(25))
    assert short_score("reg25", path) == pytest.approx(exp(.252) - 1)
    assert short_score("total25", path) == pytest.approx(exp(.024) - 1)
    assert short_score("reg25", path[::-1]) == pytest.approx(exp(-.252) - 1)
    assert short_score("reg25", (100.,) * 25) == 0
    for invalid in (path[1:], (0.,) + path[1:], (float("nan"),) + path[1:]):
        with pytest.raises(ValueError):
            short_score("reg25", invalid)


def test_regression_rewards_consistent_trend_with_same_endpoints() -> None:
    smooth = np.exp(np.linspace(0., .024, 25))
    noisy = smooth.copy()
    noisy[1:-1] *= np.exp(np.sin(np.arange(23)) * .04)
    assert short_score("total25", tuple(smooth)) == short_score("total25", tuple(noisy))
    assert short_score("reg25", tuple(smooth)) > short_score("reg25", tuple(noisy))
