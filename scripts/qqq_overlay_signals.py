"""One frozen overlay; original C11 ranking, qualification and group slots."""

from datetime import date
from math import fsum, isfinite

from beartype import beartype

import c10standard.targets as standard
from c10standard.models import Contract, Prepared, TargetRecord


@beartype
def target(prepared: Prepared, day: date) -> TargetRecord:
    if "QQQ" not in prepared.features.frames or "QQQ" not in prepared.risk_names:
        raise ValueError("QQQ is missing from the frozen universe or features")
    frame = prepared.features.frames["QQQ"].loc[:str(day)]
    if len(frame) < 253 or frame.index[-1].date() != day:
        raise ValueError("QQQ requires 253 observations and exact signal close")
    row = frame.iloc[-1]
    close, turnover, positive = (float(row[k]) for k in ("raw_close", "turnover20", "positive20"))
    if (not all(isfinite(v) for v in (close, turnover, positive))
            or close < 5 or turnover < 1e7 or positive != 20):
        raise ValueError("QQQ fails physical trade qualification")
    original = standard.target(Contract(), prepared, day)
    weights = {s: .4 for s, weight in original.q.items() if s != "SHY" and weight > 0}
    if "QQQ" in weights:
        weights["SHY"] = .95 - sum(weights.values())
    else:
        weights["QQQ"] = .15
        weights["SHY"] = 1 - sum(weights.values())
    return TargetRecord(day=day, q=weights, reason="oracle")


@beartype
def oracle(prepared: Prepared, day: date) -> TargetRecord:
    """Independently qualify raw QQQ and convert the independent raw C11 oracle."""
    datasets = [d for d in prepared.datasets if d.symbol == "QQQ"]
    if len(datasets) != 1 or "QQQ" not in prepared.risk_names:
        raise ValueError("QQQ raw data or frozen universe entry missing")
    raw = datasets[0].frame.loc[:str(day)]
    if len(raw) < 253 or raw.index[-1].date() != day:
        raise ValueError("QQQ raw calendar or warmup is invalid")
    recent = raw.iloc[-20:]
    close = float(raw.close.iloc[-1])
    turnover = fsum(float(value) for value in recent.turnover) / 20
    if (not isfinite(close) or close < 5 or not isfinite(turnover) or turnover < 1e7
            or not all(isfinite(float(v)) and float(v) > 0 for v in recent.volume)):
        raise ValueError("QQQ raw trade qualification failed")
    original = standard.oracle(Contract(), prepared, day)
    risk = [s for s in original.q if s != "SHY" and original.q[s] > 0]
    weights = {symbol: 2 / 5 for symbol in risk}
    if original.q.get("QQQ", 0) > 0:
        weights["SHY"] = (19 - 8 * len(risk)) / 20
    else:
        weights["QQQ"] = 3 / 20
        weights["SHY"] = (17 - 8 * len(risk)) / 20
    return TargetRecord(day=day, q=weights, reason="oracle")
