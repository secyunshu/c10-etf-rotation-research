"""Pure causal ranking and target weights; no orders, returns or episode state.

Exit triggers, actual holding ages, scheduling and trade bands belong to the runner.
The runner supplies excluded/risk-exited symbols through SignalContext.blocked.
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .models import CORE, EXPANDED, Allocation, FeatureStore, P2Spec, SignalContext, TargetSignal

type Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
type Matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
BUCKETS = (("SPY", "QQQ", "EFA"), ("TLT", "IEF"), ("GLD",), ("DBC",))


def _value(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        raise ValueError(f"missing feature {column}")
    value = float(frame[column].iloc[-1])
    if not math.isfinite(value):
        raise ValueError(f"missing/warmup feature {column}")
    return value


def _covariance(frames: list[pd.DataFrame]) -> Matrix:
    if any("ret" not in frame or len(frame) < 64 for frame in frames):
        raise ValueError("missing 63-day covariance history")
    index = frames[0].index[-63:]
    if any(not frame.index[-63:].equals(index) for frame in frames):
        raise ValueError("misaligned covariance history")
    values = np.column_stack([frame["ret"].iloc[-63:].to_numpy(dtype=float) for frame in frames])
    if not np.isfinite(values).all():
        raise ValueError("nonfinite covariance returns")
    cov: Matrix = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    if not np.isfinite(cov).all():
        raise ValueError("nonfinite covariance matrix")
    return cov


def _cap_weights(raw: Vector, cap: float) -> Vector:
    if cap <= 0 or cap > 1 or len(raw) * cap < 1 - 1e-12:
        raise ValueError("infeasible allocation cap")
    result = np.zeros(len(raw))
    free = np.ones(len(raw), dtype=bool)
    remaining = 1.0
    while free.any():
        proposed = remaining * raw[free] / raw[free].sum()
        exceeds = proposed > cap
        if not exceeds.any():
            result[free] = proposed
            break
        positions = np.flatnonzero(free)[exceeds]
        result[positions] = cap
        free[positions] = False
        remaining = 1 - float(result.sum())
    return result


def _risk_weights(spec: P2Spec, frames: list[pd.DataFrame]) -> tuple[Vector, str]:
    count = len(frames)
    if not count:
        raise ValueError("no risk slots")
    cap = spec.cap if spec.cap is not None else 1.0
    if cap * count < 1 - 1e-12:
        raise ValueError("infeasible allocation cap")
    if spec.allocation == "equal":
        return _cap_weights(np.ones(count), cap), ""
    if spec.allocation == "inverse":
        vols = np.array([_value(frame, "vol63") for frame in frames])
        if (vols < 0).any():
            raise ValueError("negative volatility")
        return _cap_weights(1 / np.maximum(vols, 0.01), cap), ""
    sample = _covariance(frames)
    covariance = 0.5 * sample + 0.5 * np.diag(np.diag(sample))
    # Positive scalar conditioning preserves the exact minimizer and constraints.
    # Daily variances can be much smaller than ftol, causing false early success.
    trace = float(np.trace(covariance))
    objective_scale = 1 / trace if trace > 0 else 1.0
    covariance = covariance * objective_scale

    def objective(w: Vector) -> float:
        return float(w @ covariance @ w)

    def gradient(w: Vector) -> Vector:
        return 2 * covariance @ w

    def full_investment(w: Vector) -> float:
        return float(w.sum() - 1)

    def constraint_gradient(w: Vector) -> Vector:
        return np.ones(len(w))

    try:
        result = minimize(
            objective,
            np.full(count, 1 / count),
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, cap)] * count,
            constraints={"type": "eq", "fun": full_investment, "jac": constraint_gradient},
            options={"ftol": 1e-10, "maxiter": 500},
        )
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return np.zeros(count), f"SLSQP: {exc}"
    weights: Vector = np.asarray(result.x, dtype=float)
    if (
        not result.success
        or weights.shape != (count,)
        or not np.isfinite(weights).all()
        or abs(float(weights.sum()) - 1) > 1e-8
        or (weights < -1e-8).any()
        or (weights > cap + 1e-8).any()
    ):
        return np.zeros(count), f"SLSQP constraints/convergence failure: {result.message}"
    # Numerical cleanup only within the accepted tolerance; no optimizer fallback.
    weights = np.clip(weights, 0.0, cap)
    return weights / weights.sum(), ""


def _d_selection(
    spec: P2Spec,
    ranked: list[int],
    eligible: list[bool],
    momentum: Vector,
    symbols: tuple[str, ...],
    context: SignalContext,
) -> list[int]:
    actual = {held.symbol: held for held in context.holdings if held.shares > 0}
    selected = [i for i in range(len(symbols)) if symbols[i] in actual and eligible[i]]
    if len(selected) > spec.top:
        raise ValueError("actual risk holdings exceed configured slots")
    locked = {i for i in selected if actual[symbols[i]].age < spec.min_days}
    rank_order = {index: position for position, index in enumerate(ranked)}
    for challenger in ranked:
        if not eligible[challenger] or challenger in selected:
            continue
        if len(selected) < spec.top:
            selected.append(challenger)
            continue
        replaceable = [i for i in selected if i not in locked]
        if not replaceable:
            continue
        weakest = max(replaceable, key=lambda i: rank_order[i])
        # Compare return levels to avoid cancellation turning exact h into > h.
        if momentum[challenger] > momentum[weakest] + spec.hurdle:
            selected.remove(weakest)
            selected.append(challenger)
    return sorted(selected, key=lambda i: rank_order[i])


def signal(spec: P2Spec, features: FeatureStore, context: SignalContext) -> TargetSignal:
    """Return absolute account weights summing to 0.95; absence means target zero.

    Numerical optimizer failures return empty allocations with an explicit fault.
    Missing data always raises ValueError. Only rows <= context.day are read.
    """
    universe = EXPANDED if spec.universe == "expanded" else CORE
    symbols = tuple(s for s in universe if s != "SHY" and s not in spec.removed)
    if not spec.windows or any(w <= 0 for w in spec.windows) or spec.top <= 0:
        raise ValueError("invalid windows or risk slot count")
    visible: list[pd.DataFrame] = []
    names = (*symbols, "SHY")
    for symbol in names:
        if symbol not in features.frames:
            raise ValueError(f"missing symbol {symbol}")
        source = features.frames[symbol]
        if not isinstance(source.index, pd.DatetimeIndex):
            raise ValueError(f"{symbol}: invalid date index")
        if (
            source.index.has_duplicates
            or not source.index.is_monotonic_increasing
            or source.index.tz is not None
        ):
            raise ValueError(f"{symbol}: invalid date index")
        frame = source.loc[: pd.Timestamp(context.day)]
        if frame.empty or frame.index[-1] != pd.Timestamp(context.day):
            raise ValueError(f"{symbol}: missing signal day {context.day}")
        visible.append(frame)
    shy_index = len(symbols)
    q = np.zeros(len(names))
    if spec.fixed:
        if any(s not in names for s in spec.fixed):
            raise ValueError("fixed asset missing from universe")
        if spec.bucket_caps:
            for economic_bucket in BUCKETS:
                members = [s for s in economic_bucket if s in spec.fixed]
                if not members:
                    raise ValueError("economic benchmark missing bucket")
                for symbol in members:
                    q[names.index(symbol)] = 0.25 / len(members)
        else:
            for symbol in spec.fixed:
                q[names.index(symbol)] += 1 / len(spec.fixed)
    else:
        if spec.top > len(symbols):
            raise ValueError("too few risk assets for requested slots")
        momentum = np.array([[_value(frame, f"mom{w}") for w in spec.windows] for frame in visible])
        mean_momentum = momentum.mean(axis=1)
        if "regression" in spec.score:
            scores = np.array(
                [[_value(frame, f"reg{w}") for w in spec.windows] for frame in visible[:-1]]
            )
        else:
            scores = momentum[:-1].copy()
            if "volatility" in spec.score:
                vols = np.array([_value(frame, "vol63") for frame in visible[:-1]])
                if (vols < 0).any():
                    raise ValueError("negative volatility")
                scores /= np.maximum(vols[:, None], 0.01)
        if spec.score.startswith("rank_"):
            percentiles = np.zeros_like(scores)
            for j in range(scores.shape[1]):
                order = sorted(range(len(symbols)), key=lambda i: (-scores[i, j], i))
                for rank, i in enumerate(order):
                    percentiles[i, j] = 1 - rank / max(len(symbols) - 1, 1)
            final_scores = percentiles.mean(axis=1)
        else:
            final_scores = scores.mean(axis=1)
        ranked = sorted(range(len(symbols)), key=lambda i: (-final_scores[i], i))
        eligible = []
        for i, symbol in enumerate(symbols):
            valid = symbol not in context.blocked
            if spec.filter in ("cash", "both"):
                valid = valid and mean_momentum[i] > mean_momentum[shy_index]
            if spec.filter in ("sma", "both"):
                valid = valid and _value(visible[i], "tr") > _value(visible[i], "sma200")
            eligible.append(bool(valid))
        if spec.hysteresis:
            chosen = _d_selection(spec, ranked, eligible, mean_momentum, symbols, context)
            if chosen:
                if spec.allocation != "equal":
                    raise ValueError("hysteresis supports frozen equal-weight templates only")
                for i in chosen:
                    q[i] = 1 / spec.top
            q[shy_index] = 1 - float(q.sum())
        else:
            chosen = ranked[: spec.top]
            raw, error = _risk_weights(spec, [visible[i] for i in chosen])
            if error:
                return TargetSignal(allocations=(), numerical_failure=True, error=error)
            for i, weight in zip(chosen, raw, strict=True):
                q[i if eligible[i] else shy_index] += weight
    if spec.bucket_caps:
        q[:-1] = np.minimum(q[:-1], 0.5)
        for bucket, limit in zip(BUCKETS, (0.5, 0.5, 0.5, 0.25), strict=True):
            indices = [names.index(s) for s in bucket if s in symbols]
            total = float(q[indices].sum())
            if total > limit:
                q[indices] *= limit / total
        q[shy_index] = 1 - float(q[:-1].sum())
    if spec.vol_target > 0:
        active = np.flatnonzero(q > 0)
        covariance = _covariance([visible[i] for i in active])
        active_q = q[active]
        variance = float(active_q @ covariance @ active_q)
        if not math.isfinite(variance) or variance < -1e-12:
            raise ValueError("invalid portfolio covariance")
        sigma = math.sqrt(252 * max(variance, 0.0))
        scale = min(1.0, spec.vol_target / max(sigma, 0.01))
        q[:-1] *= scale
        q[-1] = 1 - float(q[:-1].sum())
    return TargetSignal(
        allocations=tuple(
            Allocation(symbol=symbol, weight=float(0.95 * weight))
            for symbol, weight in zip(names, q, strict=True)
            if weight > 0
        )
    )
