"""Fixed path formulas, causal data boundaries and unchanged portfolio policy."""

import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from c10experiments.io import prefix_market
from c10paths import signals
from c10paths.models import VARIANTS, Variant
from c10standard.models import NAMES, RISK, Prepared, Strategy
from rotbt.models import Dataset
from rotbt.phase2.models import FeatureStore

DAY = date(2022, 3, 31)
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _sample() -> Prepared:
    days = xcals.get_calendar("XNYS").sessions_in_range("2020-12-31", DAY).tz_localize(None)
    datasets = []
    for number, symbol in enumerate(NAMES):
        rate = 0.0001 if symbol == "SHY" else (number + 2) * 0.00001
        frame = pd.DataFrame(
            {
                "tr": 100 * np.exp(rate * np.arange(len(days))),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1e6,
                "turnover": 20e6,
            },
            index=days,
        )
        datasets.append(Dataset(symbol=symbol, frame=frame, dividends=[]))
    prepared = Prepared(
        datasets=datasets,
        features=FeatureStore(frames={}),
        strategy=Strategy.model_validate_json((ROOT / "standard/STRATEGY.json").read_text()),
        risk_names=RISK,
        first=days[0].date(),
        last=DAY,
    )
    return prefix_market(prepared, DAY)


@pytest.fixture
def market() -> Prepared:
    return _sample().model_copy(deep=True)


def _set_levels(market: Prepared, symbol: str, levels: list[float]) -> None:
    next(d for d in market.datasets if d.symbol == symbol).frame["tr"] = levels


def _levels(market: Prepared, growth: float) -> list[float]:
    return [100.0 * growth**i for i in range(len(market.datasets[0].frame))]


def test_rank_ties_use_fixed_34_wright_scale_and_ignore_shy_and_liquidity(market: Prepared) -> None:
    for symbol in RISK:
        _set_levels(market, symbol, _levels(market, 1.0 if symbol in RISK[:17] else 2.0))
    _set_levels(market, "SHY", _levels(market, 4.0))
    next(d for d in market.datasets if d.symbol == RISK[-1]).frame["turnover"] = 1e6
    prepared = prefix_market(market, DAY)
    scores = signals.score_table("P01", prepared, DAY)
    expected = 8.5 / math.sqrt((34**2 - 1) / 12)
    assert set(scores.assets) == set(RISK)
    assert scores.assets[RISK[0]].rank == pytest.approx(-expected)
    assert scores.assets[RISK[-1]].rank == pytest.approx(expected)
    assert not scores.assets[RISK[-1]].eligible
    assert signals.oracle_scores("P01", prepared, DAY).assets[RISK[-1]].rank == pytest.approx(
        expected
    )


def test_rank_120_days_include_both_endpoints_and_skip_last_20_returns(market: Prepared) -> None:
    for symbol in RISK:
        _set_levels(market, symbol, _levels(market, 1.0))
    size = len(market.datasets[0].frame)
    daily = [0.0] * size
    for offset in (139, 20):
        daily[size - 1 - offset] = 1.0
    for offset in (140, 19, 0):
        daily[size - 1 - offset] = -0.5
    levels = [100.0]
    for change in daily[1:]:
        levels.append(levels[-1] * (1 + change))
    _set_levels(market, "AGG", levels)
    prepared = prefix_market(market, DAY)
    expected = 2 / 120 * 16.5 / math.sqrt((34**2 - 1) / 12)
    assert signals.score_table("P01", prepared, DAY).assets["AGG"].rank == pytest.approx(expected)
    assert signals.oracle_scores("P01", prepared, DAY).assets["AGG"].rank == pytest.approx(expected)


def test_equal_rank_sums_preserve_ticker_ties_regardless_of_daily_path_order(
    market: Prepared,
) -> None:
    size = len(market.datasets[0].frame)
    for number, symbol in enumerate(RISK):
        daily = [0.0] * size
        for offset, position in enumerate(range(size - 140, size - 20)):
            daily[position] = 1.0 if (offset < 60) == (number < 17) else -0.5
        daily[-1] = 1.0
        levels = [100.0]
        for change in daily[1:]:
            levels.append(levels[-1] * (1 + change))
        _set_levels(market, symbol, levels)
    _set_levels(market, "SHY", _levels(market, 1.0))
    prepared = prefix_market(market, DAY)
    for function in (signals.score_table, signals.oracle_scores):
        scores = function("P01", prepared, DAY)
        assert all(score.rank == 0.0 for score in scores.assets.values())
    for function in (signals.target, signals.oracle):
        assert function("P01", prepared, DAY).q == {
            "AGG": 0.45,
            "BIL": 0.45,
            "SHY": 1 - 0.9,
        }


def test_smooth_literal_uses_240_daily_returns_and_can_exceed_one(market: Prepared) -> None:
    _set_levels(market, "AGG", _levels(market, 1.01))
    # The cash hurdle remains meaningful even when SHY has no path movement.
    _set_levels(market, "SHY", _levels(market, 1.0))
    prepared = prefix_market(market, DAY)
    expected = (1.01**240 - 1) / (240 * 0.01)
    assert expected > 1
    for function in (signals.score_table, signals.oracle_scores):
        assert function("P02", prepared, DAY).assets["AGG"].rank == pytest.approx(expected)


def test_smooth_zero_denominator_fails_even_for_an_ineligible_asset(market: Prepared) -> None:
    _set_levels(market, "AGG", _levels(market, 1.0))
    next(d for d in market.datasets if d.symbol == "AGG").frame["turnover"] = 1e6
    prepared = prefix_market(market, DAY)
    for function in (signals.score_table, signals.oracle_scores):
        with pytest.raises(ValueError, match="denominator"):
            function("P02", prepared, DAY)


def test_high_uses_previous_250_levels_excluding_today_and_older_peak(market: Prepared) -> None:
    levels = _levels(market, 1.0)
    levels[-252], levels[-251], levels[-1] = 1000.0, 120.0, 132.0
    _set_levels(market, "AGG", levels)
    prepared = prefix_market(market, DAY)
    for function in (signals.score_table, signals.oracle_scores):
        assert function("P03", prepared, DAY).assets["AGG"].rank == pytest.approx(0.1)


def test_top_two_group_slots_are_chosen_before_strict_r252_gate_without_refill(
    market: Prepared,
) -> None:
    for symbol in RISK:
        _set_levels(market, symbol, _levels(market, 1.0))
    _set_levels(market, "SHY", _levels(market, 1.0))
    for symbol, last, first in (
        ("SPY", 150.0, 150.0),  # Highest path score, but equals cash at R252.
        ("QQQ", 140.0, 80.0),  # Same economic group: must not consume a slot.
        ("GLD", 130.0, 100.0),
        ("XLE", 120.0, 90.0),  # Would pass, but must not replace failed SPY.
    ):
        levels = _levels(market, 1.0)
        levels[-253], levels[-1] = first, last
        _set_levels(market, symbol, levels)
    prepared = prefix_market(market, DAY)
    for function in (signals.target, signals.oracle):
        result = function("P03", prepared, DAY)
        assert result.q == {"GLD": 0.45, "SHY": 0.55}
        assert result.day == DAY


@pytest.mark.parametrize("variant", VARIANTS)
def test_raw_oracle_and_physical_prefix_match_all_scores_and_targets(
    market: Prepared, variant: Variant
) -> None:
    day = date(2021, 12, 31)
    prefix = prefix_market(market, day)
    full_scores = signals.score_table(variant, market, day)
    raw_scores = signals.oracle_scores(variant, market, day)
    truncated_scores = signals.score_table(variant, prefix, day)
    assert full_scores.shy_r252 == pytest.approx(raw_scores.shy_r252)
    for symbol in RISK:
        assert full_scores.assets[symbol].rank == pytest.approx(raw_scores.assets[symbol].rank)
        assert full_scores.assets[symbol].r252 == pytest.approx(raw_scores.assets[symbol].r252)
        assert full_scores.assets[symbol].eligible == raw_scores.assets[symbol].eligible
        assert full_scores.assets[symbol] == truncated_scores.assets[symbol]
    assert signals.target(variant, market, day) == signals.oracle(variant, prefix, day)
    assert signals.target(variant, market, day) == signals.target(variant, prefix, day)
    # Corrupting all future levels must not contaminate an earlier decision.
    for frame in market.features.frames.values():
        frame.loc[frame.index > str(day), "tr"] = float("nan")
    for dataset in market.datasets:
        dataset.frame.loc[dataset.frame.index > str(day), "tr"] = -1.0
    assert signals.score_table(variant, market, day) == full_scores
    assert signals.oracle_scores(variant, market, day) == raw_scores


@pytest.mark.parametrize("variant", VARIANTS)
def test_signal_requires_real_monthend_and_complete_253_point_warmup(
    market: Prepared, variant: Variant
) -> None:
    for function in (signals.score_table, signals.oracle_scores):
        with pytest.raises(ValueError, match="month.end"):
            function(variant, market, date(2022, 3, 30))
        with pytest.raises(ValueError, match="253|warmup"):
            function(variant, market, date(2021, 11, 30))


@pytest.mark.parametrize("fault", ["pool", "missing", "duplicate", "nan", "negative"])
def test_rank_reference_rejects_incomplete_or_invalid_input_instead_of_shrinking_pool(
    market: Prepared, fault: str
) -> None:
    if fault == "pool":
        market.risk_names = RISK[:-1]
    else:
        frame = market.features.frames["AGG"]
        dataset = next(d for d in market.datasets if d.symbol == "AGG")
        if fault == "missing":
            market.features.frames["AGG"] = frame.drop(frame.index[-50])
            dataset.frame = dataset.frame.drop(dataset.frame.index[-50])
        elif fault == "duplicate":
            market.features.frames["AGG"] = pd.concat([frame, frame.iloc[-1:]])
            dataset.frame = pd.concat([dataset.frame, dataset.frame.iloc[-1:]])
        else:
            value = float("nan") if fault == "nan" else -1.0
            frame.iloc[-50, frame.columns.get_loc("tr")] = value
            dataset.frame.iloc[-50, dataset.frame.columns.get_loc("tr")] = value
    for function in (signals.score_table, signals.oracle_scores):
        with pytest.raises(ValueError):
            function("P01", market, DAY)


def test_nonfinite_qualification_and_duplicate_raw_dataset_fail_explicitly(
    market: Prepared,
) -> None:
    market.features.frames["AGG"].loc[str(DAY), "turnover20"] = float("nan")
    with pytest.raises(ValueError):
        signals.target("P03", market, DAY)
    market.datasets.append(market.datasets[0])
    with pytest.raises(ValueError):
        signals.oracle("P03", market, DAY)


def test_score_oracle_is_independent_of_prepared_features_and_production_functions(
    market: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = signals.oracle("P03", market, DAY)
    market.features = FeatureStore(frames={})

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("oracle must not call production calculation")

    monkeypatch.setattr(signals, "score_table", forbidden)
    monkeypatch.setattr(signals, "target", forbidden)
    assert signals.oracle("P03", market, DAY) == expected


@pytest.mark.parametrize("column", ["tr", "close", "volume", "turnover"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_raw_invalid_values_fail_before_entering_a_rank_reference_pool(
    market: Prepared, column: str, bad: float
) -> None:
    frame = market.datasets[0].frame
    frame.loc[frame.index[-35], column] = bad
    with pytest.raises(ValueError, match="nonfinite or nonpositive"):
        signals.oracle_scores("P01", market, DAY)


@pytest.mark.parametrize("variant", ["P01", "P02"])
@pytest.mark.parametrize("source", ["features", "raw"])
def test_overflowing_daily_returns_fail_instead_of_producing_a_finite_misleading_score(
    market: Prepared, variant: Variant, source: str
) -> None:
    # Positive finite levels can still overflow a ratio; ranking inf would hide the error.
    for frame in (
        market.features.frames["AGG"],
        next(d for d in market.datasets if d.symbol == "AGG").frame,
    ):
        frame.loc[frame.index[-100], "tr"] = 1e-300
        frame.loc[frame.index[-99], "tr"] = 1e100
    function = signals.score_table if source == "features" else signals.oracle_scores
    with pytest.raises(ValueError, match="nonfinite.*return"):
        function(variant, market, DAY)


@pytest.mark.parametrize("source", ["features", "raw"])
def test_smooth_denominator_overflow_fails_instead_of_reporting_zero_momentum(
    market: Prepared, source: str
) -> None:
    # Each daily ratio is finite, but 120 large increases overflow their L1 sum.
    levels = [1.0 if index % 2 == 0 else 1e307 for index in range(253)]
    assert all(math.isfinite(b / a - 1) for a, b in zip(levels[:-1], levels[1:], strict=True))
    for frame in (
        market.features.frames["AGG"],
        next(d for d in market.datasets if d.symbol == "AGG").frame,
    ):
        frame.loc[frame.index[-253:], "tr"] = levels
    function = signals.score_table if source == "features" else signals.oracle_scores
    with pytest.raises(ValueError, match="denominator"):
        function("P02", market, DAY)
