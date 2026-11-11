"""Causal path scores and independent raw-data formula/qualification oracles."""

import math
from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype

from c10standard.models import NAMES, RISK, Prepared, TargetRecord
from c10standard.targets import group_for
from rotbt.phase2.models import FeatureStore

from .models import Score, Scores, Variant


def _feature_prefixes(prepared: Prepared, day: date) -> FeatureStore:
    if prepared.risk_names != RISK or set(prepared.features.frames) != set(NAMES):
        raise ValueError("fixed 34 risk assets and SHY required")
    calendar = xcals.get_calendar("XNYS")
    month = pd.Period(day, freq="M")
    if calendar.sessions_in_range(month.start_time, month.end_time.normalize())[-1].date() != day:
        raise ValueError("signal must be the true XNYS month-end")
    expected = calendar.sessions_in_range(prepared.first, day).tz_localize(None)
    if len(expected) < 253:
        raise ValueError("253 observations required for warmup")
    frames: dict[str, pd.DataFrame] = {}
    for symbol in NAMES:
        source = prepared.features.frames[symbol]
        if not isinstance(source.index, pd.DatetimeIndex):
            raise ValueError("signal input requires date index")
        frame = source.loc[: str(day)]
        if not frame.index.equals(expected):
            raise ValueError("incomplete, duplicated or misaligned signal calendar")
        tr = frame.tr.to_numpy(dtype=float)
        if not np.isfinite(tr).all() or (tr <= 0).any():
            raise ValueError("nonfinite or nonpositive total return input")
        row = frame.iloc[-1]
        values = [float(row[k]) for k in ("mom252", "raw_close", "turnover20", "positive20")]
        if (
            not all(math.isfinite(v) for v in values)
            or values[1] <= 0
            or values[2] < 0
            or not 0 <= values[3] <= 20
        ):
            raise ValueError("invalid signal qualification input")
        if symbol != "SHY":
            group_for(prepared, symbol)
        frames[symbol] = frame
    return FeatureStore(frames=frames)


@beartype
def score_table(variant: Variant, prepared: Prepared, day: date) -> Scores:
    """Use prepared features, retaining ineligible assets in the rank reference pool."""
    frames = _feature_prefixes(prepared, day).frames
    levels = pd.DataFrame({s: frames[s].tr.iloc[-253:] for s in RISK})
    if variant == "P01":
        daily = levels.pct_change(fill_method=None).iloc[-140:-20]
        if not np.isfinite(daily.to_numpy(dtype=float)).all():
            raise ValueError("nonfinite daily return")
        ranks = daily.rank(axis=1, method="average", ascending=True)
        # The affine transform commutes with the mean. Sum exact half-integer
        # ranks first so equal rank sums retain the registered ticker tie-break.
        path = (ranks.mean() - 17.5) / math.sqrt((34**2 - 1) / 12)
    elif variant == "P02":
        daily = levels.pct_change(fill_method=None).iloc[-240:]
        if not np.isfinite(daily.to_numpy(dtype=float)).all():
            raise ValueError("nonfinite daily return")
        absolute_returns = daily.abs().sum()
        if (
            not np.isfinite(absolute_returns.to_numpy(dtype=float)).all()
            or (absolute_returns <= 0).any()
        ):
            raise ValueError("Smooth240 denominator must be finite and positive")
        path = (levels.iloc[-1] / levels.iloc[-241] - 1) / absolute_returns
    else:
        path = levels.iloc[-1] / levels.iloc[-251:-1].max() - 1
    assets: dict[str, Score] = {}
    for symbol in RISK:
        row = frames[symbol].iloc[-1]
        assets[symbol] = Score(
            rank=float(path[symbol]),
            r252=float(row.mom252),
            eligible=(
                float(row.raw_close) >= 5
                and float(row.turnover20) >= 1e7
                and float(row.positive20) == 20
            ),
        )
    return Scores(
        variant=variant,
        day=day,
        assets=assets,
        shy_r252=float(frames["SHY"].iloc[-1].mom252),
    )


@beartype
def target(variant: Variant, prepared: Prepared, day: date) -> TargetRecord:
    scores = score_table(variant, prepared, day)
    eligible = [s for s in RISK if scores.assets[s].eligible]
    selected: list[str] = []
    used: set[str] = set()
    for symbol in sorted(eligible, key=lambda s: (-scores.assets[s].rank, s)):
        group = group_for(prepared, symbol)
        if group in used:
            continue
        selected.append(symbol)
        used.add(group)
        if len(selected) == 2:
            break
    q = {s: 0.45 for s in selected if scores.assets[s].r252 > scores.shy_r252}
    q["SHY"] = 1 - sum(q.values())
    return TargetRecord(day=day, q=q, reason="oracle")


@beartype
def oracle_scores(variant: Variant, prepared: Prepared, day: date) -> Scores:
    """Recompute from raw normalized TR; no feature or production-score access."""
    calendar = xcals.get_calendar("XNYS")
    stamp = pd.Timestamp(day)
    monthend = calendar.date_to_session(stamp + pd.offsets.MonthEnd(0), direction="previous")
    if monthend.date() != day:
        raise ValueError("raw oracle requires true month-end")
    if prepared.risk_names != RISK:
        raise ValueError("raw oracle requires fixed 34 risk assets")
    sources = {dataset.symbol: dataset.frame for dataset in prepared.datasets}
    if len(sources) != len(prepared.datasets) or set(sources) != set(NAMES):
        raise ValueError("raw datasets must contain unique fixed 35 assets")
    sessions = list(calendar.sessions_in_range(prepared.first, day).tz_localize(None))
    if len(sessions) < 253:
        raise ValueError("raw oracle requires 253 warmup observations")
    levels: dict[str, list[float]] = {}
    qualification: dict[str, bool] = {}
    returns252: dict[str, float] = {}
    for symbol, source in sources.items():
        if not isinstance(source.index, pd.DatetimeIndex):
            raise ValueError("raw input requires date index")
        frame = source.loc[: str(day)]
        if list(frame.index) != sessions:
            raise ValueError("raw calendar incomplete, duplicated or misaligned")
        for column in ("tr", "close", "turnover", "volume"):
            values = [float(value) for value in frame[column]]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("raw nonfinite or nonpositive input")
        prices = [float(value) for value in frame.tr.iloc[-253:]]
        levels[symbol] = prices
        returns252[symbol] = prices[-1] / prices[0] - 1
        qualification[symbol] = (
            float(frame.close.iloc[-1]) >= 5
            and math.fsum(float(value) for value in frame.turnover.iloc[-20:]) / 20 >= 1e7
            and all(float(value) > 0 for value in frame.volume.iloc[-20:])
        )
        if symbol != "SHY":
            group_for(prepared, symbol)
    ranks: dict[str, float] = {}
    if variant == "P01":
        daily_ranks: dict[str, list[float]] = {symbol: [] for symbol in RISK}
        # Prices contain t-252..t. Return endpoints 113..232 are t-139..t-20.
        for position in range(113, 233):
            cross_section = {
                symbol: levels[symbol][position] / levels[symbol][position - 1] - 1
                for symbol in RISK
            }
            if any(not math.isfinite(value) for value in cross_section.values()):
                raise ValueError("raw nonfinite daily return")
            for symbol, value in cross_section.items():
                below = sum(other < value for other in cross_section.values())
                tied = sum(other == value for other in cross_section.values())
                average_rank = 1 + below + (tied - 1) / 2
                daily_ranks[symbol].append(average_rank)
        ranks = {
            symbol: (math.fsum(values) / 120 - 17.5) / math.sqrt(96.25)
            for symbol, values in daily_ranks.items()
        }
    else:
        for symbol in RISK:
            prices = levels[symbol]
            if variant == "P02":
                daily = [prices[i] / prices[i - 1] - 1 for i in range(13, 253)]
                if any(not math.isfinite(value) for value in daily):
                    raise ValueError("raw nonfinite daily return")
                try:
                    denominator = math.fsum(abs(value) for value in daily)
                except OverflowError as error:
                    raise ValueError("raw Smooth240 denominator overflow") from error
                if not math.isfinite(denominator) or denominator <= 0:
                    raise ValueError("raw Smooth240 denominator must be finite and positive")
                ranks[symbol] = (prices[252] / prices[12] - 1) / denominator
            else:
                ranks[symbol] = prices[252] / max(prices[2:252]) - 1
    return Scores(
        variant=variant,
        day=day,
        assets={
            symbol: Score(
                rank=ranks[symbol], r252=returns252[symbol], eligible=qualification[symbol]
            )
            for symbol in RISK
        },
        shy_r252=returns252["SHY"],
    )


@beartype
def oracle(variant: Variant, prepared: Prepared, day: date) -> TargetRecord:
    scores = oracle_scores(variant, prepared, day)
    ordered = sorted(
        (-score.rank, symbol, score.r252)
        for symbol, score in scores.assets.items()
        if score.eligible
    )
    q: dict[str, float] = {}
    groups: set[str] = set()
    for _, symbol, r252 in ordered:
        group = group_for(prepared, symbol)
        if group in groups:
            continue
        groups.add(group)
        if r252 > scores.shy_r252:
            q[symbol] = 0.45
        if len(groups) == 2:
            break
    q["SHY"] = 1 - math.fsum(q.values())
    return TargetRecord(day=day, q=q, reason="oracle")
