import numpy as np
from numpy.typing import NDArray

from rotbt.models import BootstrapResult


def max_bootstrap(
    differences: NDArray[np.float64],
    names: tuple[str, ...],
    block: int = 5,
    repetitions: int = 10000,
) -> BootstrapResult:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
        raise ValueError("invalid excess return matrix")
    days = len(values)
    rng = np.random.default_rng(20261105 + block)
    observed = values.mean(axis=0)
    boot = np.empty((repetitions, len(names)))
    for i in range(repetitions):
        starts = rng.integers(0, days, size=int(np.ceil(days / block)))
        idx = ((starts[:, None] + np.arange(block)) % days).reshape(-1)[:days]
        boot[i] = values[idx].mean(axis=0)
    maxima = (boot - observed).max(axis=1)
    adjusted = (1 + (maxima[:, None] >= observed).sum(axis=0)) / (repetitions + 1)
    family = (1 + int((maxima >= observed.max()).sum())) / (repetitions + 1)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return BootstrapResult(
        names=names,
        block=block,
        family_p=float(family),
        adjusted_p=adjusted.tolist(),
        mean_bp=(observed * 10000).tolist(),
        ci_low_bp=(low * 10000).tolist(),
        ci_high_bp=(high * 10000).tolist(),
    )
