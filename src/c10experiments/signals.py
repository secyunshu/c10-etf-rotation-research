"""Frozen experiment scores and a separately computed raw-data oracle."""

import math
from datetime import date
from statistics import stdev

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from beartype import beartype

from c10standard.models import Prepared, TargetRecord

from .models import Experiment, Score, Variant


@beartype
def calendar_endpoints(day: date) -> dict[int, date]:
    calendar = xcals.get_calendar("XNYS")
    month = pd.Period(day, freq="M")
    points = {
        lag: calendar.sessions_in_range(
            (month - lag).start_time, (month - lag).end_time.normalize()
        )[-1].date()
        for lag in (0, 1, 3, 6, 12)
    }
    if points[0] != day:
        raise ValueError("signal must be the actual XNYS month-end")
    return points


@beartype
def metric(variant: Variant, frame: pd.DataFrame, day: date) -> Score:
    prefix = frame.loc[: str(day)]
    if len(prefix) < 253 or prefix.index[-1].date() != day:
        raise ValueError("253 observations and exact signal close required")
    row = prefix.iloc[-1]
    r252 = float(row.mom252)
    rank, gate, trend = r252, r252, True
    if variant == "E01":
        rank = float(prefix.tr.iloc[-22] / prefix.tr.iloc[-253] - 1)
        gate = rank
    elif variant in ("W13612", "E02"):
        points = calendar_endpoints(day)
        levels = []
        for lag in (0, 1, 3, 6, 12):
            stamp = pd.Timestamp(points[lag])
            if stamp not in prefix.index:
                raise ValueError("exact calendar endpoint missing")
            levels.append(float(prefix.tr.loc[stamp]))
        if any(not math.isfinite(v) or v <= 0 for v in levels):
            raise ValueError("calendar endpoint nonfinite or nonpositive")
        weights = (12.0, 4.0, 2.0, 1.0) if variant == "W13612" else (1.0, 1.0, 1.0, 1.0)
        rank = (
            math.fsum(
                weight * (levels[0] / denominator - 1)
                for denominator, weight in zip(levels[1:], weights, strict=True)
            )
            / 4
        )
        gate = rank
    elif variant == "E03":
        volatility = float(row.vol63)
        if not math.isfinite(volatility) or volatility < 0:
            raise ValueError("volatility must be finite/nonnegative before flooring")
        rank = r252 / max(volatility, 0.01)
    elif variant == "E04":
        rank = float(row.reg252)
    elif variant == "E06":
        tr, average = float(row.tr), float(row.sma200)
        if not math.isfinite(tr) or not math.isfinite(average) or tr <= 0 or average <= 0:
            raise ValueError("long-trend inputs must be finite and positive")
        trend = tr > average
    return Score(rank=rank, r252=r252, gate=gate, above_trend=trend)


def group_for(prepared: Prepared, symbol: str) -> str:
    for group, members in prepared.strategy.groups.items():
        if symbol in members:
            return group
    raise ValueError(f"missing fixed economic group: {symbol}")


@beartype
def target(experiment: Experiment, prepared: Prepared, day: date) -> TargetRecord:
    calendar_endpoints(day)
    names = prepared.risk_names
    if len(names) != len(set(names)) or "SHY" in names:
        raise ValueError("duplicate/defensive symbol in risk universe")
    scores: dict[str, Score] = {}
    for symbol in (*names, "SHY"):
        if symbol != "SHY":
            group_for(prepared, symbol)
        prefix = prepared.features.frames[symbol].loc[: str(day)]
        if prefix.empty or prefix.index[-1].date() != day:
            raise ValueError("exact signal day missing")
        if len(prefix) < 253:
            if symbol == "SHY":
                raise ValueError("cash warmup insufficient")
            continue
        row = prefix.iloc[-1]
        values = [float(row[k]) for k in ("raw_close", "turnover20", "positive20")]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite qualification")
        score = metric(experiment.variant, prefix, day)
        if symbol == "SHY" or (values[0] >= 5 and values[1] >= 1e7 and values[2] == 20):
            scores[symbol] = score
    selected: list[str] = []
    used: set[str] = set()
    for symbol in sorted((s for s in scores if s != "SHY"), key=lambda s: (-scores[s].rank, s)):
        group = group_for(prepared, symbol)
        if group in used:
            continue
        selected.append(symbol)
        used.add(group)
        if len(selected) == experiment.k:
            break
    hurdle = max(scores["SHY"].gate, 0.0) if experiment.variant == "E05" else scores["SHY"].gate
    q = {
        s: 0.9 / experiment.k for s in selected if scores[s].gate > hurdle and scores[s].above_trend
    }
    q["SHY"] = max(0.0, 1 - sum(q.values()))
    return TargetRecord(day=day, q=q, reason="oracle")


@beartype
def oracle(experiment: Experiment, prepared: Prepared, day: date) -> TargetRecord:
    """Rebuild rank/gates from raw TR and liquidity, never metric() or features."""
    calendar = xcals.get_calendar("XNYS")
    stamp = pd.Timestamp(day)
    if calendar.date_to_session(stamp + pd.offsets.MonthEnd(0), direction="previous").date() != day:
        raise ValueError("raw oracle requires true month-end")
    frames = {d.symbol: d.frame.loc[: str(day)] for d in prepared.datasets}
    if len(frames) != len(prepared.datasets):
        raise ValueError("duplicate raw datasets")
    candidates: list[tuple[float, str, float, bool, str]] = []
    cash: float | None = None
    for symbol in (*prepared.risk_names, "SHY"):
        frame = frames[symbol]
        if frame.empty or frame.index[-1].date() != day:
            raise ValueError("raw exact signal day missing")
        if len(frame) < 253:
            if symbol == "SHY":
                raise ValueError("raw cash warmup insufficient")
            continue
        levels = [float(v) for v in frame.tr.iloc[-253:]]
        if any(not math.isfinite(v) or v <= 0 for v in levels):
            raise ValueError("raw TR invalid")
        r252 = levels[-1] / levels[0] - 1
        rank, gate, trend = r252, r252, True
        if experiment.variant == "E01":
            rank = levels[-22] / levels[0] - 1
            gate = rank
        elif experiment.variant in ("W13612", "E02"):
            returns = []
            for lag in (1, 3, 6, 12):
                shifted = stamp - pd.DateOffset(months=lag)
                endpoint = calendar.date_to_session(
                    shifted + pd.offsets.MonthEnd(0), direction="previous"
                )
                if endpoint not in frame.index:
                    raise ValueError("raw calendar endpoint missing")
                denominator = float(frame.tr.loc[endpoint])
                if not math.isfinite(denominator) or denominator <= 0:
                    raise ValueError("raw endpoint invalid")
                returns.append(levels[-1] / denominator - 1)
            weights = (
                (12.0, 4.0, 2.0, 1.0) if experiment.variant == "W13612" else (1.0, 1.0, 1.0, 1.0)
            )
            rank = sum(weight * value for weight, value in zip(weights, returns, strict=True)) / 4
            gate = rank
        elif experiment.variant == "E03":
            daily = [b / a - 1 for a, b in zip(levels[-64:-1], levels[-63:], strict=True)]
            volatility = stdev(daily) * math.sqrt(252)
            if not math.isfinite(volatility) or volatility < 0:
                raise ValueError("raw volatility invalid")
            rank = r252 / max(volatility, 0.01)
        elif experiment.variant == "E04":
            x = np.arange(252, dtype=float)
            y = np.log(np.asarray(levels[-252:], dtype=float))
            x = x - x.mean()
            y = y - y.mean()
            xx = math.fsum(float(a) * float(a) for a in x)
            yy = math.fsum(float(b) * float(b) for b in y)
            xy = math.fsum(float(a) * float(b) for a, b in zip(x, y, strict=True))
            rank = 0.0 if yy == 0 else 252 * xy / xx * min(1.0, xy * xy / (xx * yy))
        elif experiment.variant == "E06":
            trend = levels[-1] > math.fsum(levels[-200:]) / 200
        if not math.isfinite(rank) or not math.isfinite(gate):
            raise ValueError("raw score invalid")
        if symbol == "SHY":
            cash = gate
            continue
        recent = frame.iloc[-20:]
        turnover, volume = [float(v) for v in recent.turnover], [float(v) for v in recent.volume]
        close = float(frame.close.iloc[-1])
        if any(not math.isfinite(v) for v in (*turnover, *volume, close)):
            raise ValueError("raw qualification nonfinite")
        if close >= 5 and math.fsum(turnover) / 20 >= 1e7 and all(v > 0 for v in volume):
            candidates.append((rank, symbol, gate, trend, group_for(prepared, symbol)))
    if cash is None:
        raise ValueError("raw cash score missing")
    selected: list[tuple[str, float, bool]] = []
    used: set[str] = set()
    for _, symbol, gate, trend, group in sorted(candidates, key=lambda row: (-row[0], row[1])):
        if group in used:
            continue
        selected.append((symbol, gate, trend))
        used.add(group)
        if len(selected) == experiment.k:
            break
    threshold = max(cash, 0.0) if experiment.variant == "E05" else cash
    allocation = {
        s: 0.9 / experiment.k for s, gate, trend in selected if gate > threshold and trend
    }
    allocation["SHY"] = max(0.0, 1 - sum(allocation.values()))
    return TargetRecord(day=day, q=allocation, reason="oracle")
