"""Causal gross-TR features, one symbol per date-indexed frame.

Columns: tr, ret (daily simple return), vol20/vol63 (annual sample volatility),
sma200, momL (L-day simple return), regL (252 * log-OLS slope * R squared),
for L=21,42,63,126,189,252. Regression uses L prices; momentum uses L+1.
Warmup is NaN; missing/nonfinite input is a ValueError, never forward-filled.
"""

import numpy as np
import pandas as pd

from ..models import Dataset
from .models import FeatureStore

WINDOWS = (21, 42, 63, 126, 189, 252)


def _regression(values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    y = values - values.mean()
    xx = float(x @ x)
    yy = float(y @ y)
    if yy == 0:
        return 0.0
    xy = float(x @ y)
    return 252 * xy / xx * min(1.0, xy * xy / (xx * yy))


def build_features(datasets: list[Dataset]) -> FeatureStore:
    frames: dict[str, pd.DataFrame] = {}
    if not datasets:
        raise ValueError("empty datasets")
    for dataset in datasets:
        if dataset.symbol in frames:
            raise ValueError(f"duplicate symbol {dataset.symbol}")
        source = dataset.frame
        if "tr" not in source or source.empty:
            raise ValueError(f"{dataset.symbol}: missing gross TR data")
        if not isinstance(source.index, pd.DatetimeIndex):
            raise ValueError(f"{dataset.symbol}: expected DatetimeIndex")
        if (
            source.index.has_duplicates
            or source.index.hasnans
            or not source.index.is_monotonic_increasing
            or source.index.tz is not None
            or not source.index.equals(source.index.normalize())
        ):
            raise ValueError(f"{dataset.symbol}: invalid date index")
        tr = source["tr"].astype(float)
        if not np.isfinite(tr.to_numpy()).all() or (tr <= 0).any():
            raise ValueError(f"{dataset.symbol}: invalid gross TR")
        frame = pd.DataFrame({"tr": tr, "ret": tr.pct_change(fill_method=None)})
        for period in (20, 63):
            frame[f"vol{period}"] = frame.ret.rolling(period).std(ddof=1) * np.sqrt(252)
        frame["sma200"] = tr.rolling(200).mean()
        log_tr = pd.Series(np.log(tr.to_numpy()), index=tr.index)
        for window in WINDOWS:
            frame[f"mom{window}"] = tr.pct_change(window, fill_method=None)
            frame[f"reg{window}"] = log_tr.rolling(window).apply(_regression, raw=True)
        frames[dataset.symbol] = frame
    return FeatureStore(frames=frames)
