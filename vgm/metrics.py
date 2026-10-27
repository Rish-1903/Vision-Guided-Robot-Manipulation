"""Summary statistics used in every evaluation stage."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def describe(values: Iterable[float]) -> dict:
    """n, mean, std, median, 95th percentile and max of finite values."""
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))])
    if arr.size == 0:
        return {'n': 0, 'mean': None, 'std': None, 'median': None, 'p95': None, 'max': None}
    return {'n': int(arr.size), 'mean': float(arr.mean()), 'std': float(arr.std(ddof=0)),
            'median': float(np.median(arr)), 'p95': float(np.percentile(arr, 95)),
            'max': float(arr.max())}


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """95 % Wilson score interval for a binomial success rate."""
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(successes: int, n: int) -> dict:
    lo, hi = wilson_interval(successes, n)
    return {'successes': int(successes), 'n': int(n),
            'rate': (successes / n) if n else None, 'ci95_low': lo, 'ci95_high': hi}


def success_at_thresholds(errors_mm: Iterable[float | None], thresholds_mm=(2, 5, 10, 20, 30)) -> dict:
    """Fraction of attempts with error below each threshold; None counts as failure."""
    errs = list(errors_mm)
    n = len(errs)
    out = {}
    for t in thresholds_mm:
        k = sum(1 for e in errs if e is not None and math.isfinite(float(e)) and float(e) < t)
        out[f'<{t}mm'] = rate(k, n)
    return out
