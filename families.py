"""Function families and the scan statistic itself.

For an interval ``I = [a, b]`` and a family ``F``::

    S(I) = 1 - (SR_I + SR_O) / SR_A

where SR_A, SR_I, SR_O are sums of squared residuals from fitting F to the
whole signal, to the inside of I, and to the outside. S is 0 when splitting
buys nothing and approaches 1 when the split explains the signal entirely.

Because every family divides by its *own* SR_A, scores are comparable across
intervals within a family but **not** across families: the same interval
scores differently under a constant fit than a kernel fit, since the kernel's
SR_A is already smaller. Compare families by rank and interval agreement.

This module holds ``F_0`` (constant) and ``F_d`` (polynomial), which are cheap
enough to implement directly with prefix sums. ``F_KR`` lives in
:mod:`regscan.kernel`.
"""

from __future__ import annotations

import numpy as np

from .config import ScanConfig


def _sse_constant(ps1: np.ndarray, ps2: np.ndarray, lo: int, hi: int) -> float:
    """SSE of a constant fit to x[lo:hi] via prefix sums, in O(1)."""
    k = hi - lo
    if k <= 0:
        return 0.0
    s1 = ps1[hi] - ps1[lo]
    s2 = ps2[hi] - ps2[lo]
    return max(0.0, float(s2 - s1 * s1 / k))


def _sse_outside_constant(ps1, ps2, n, a, b) -> float:
    k = n - (b - a)
    if k <= 0:
        return 0.0
    s1 = (ps1[a] - ps1[0]) + (ps1[n] - ps1[b])
    s2 = (ps2[a] - ps2[0]) + (ps2[n] - ps2[b])
    return max(0.0, float(s2 - s1 * s1 / k))


def _sse_poly(x: np.ndarray, idx: np.ndarray, degree: int) -> float:
    if idx.size <= degree + 1:
        return 0.0
    t = idx.astype(float)
    coef = np.polyfit(t, x[idx], degree)
    resid = x[idx] - np.polyval(coef, t)
    return float(resid @ resid)


def scan_constant(x: np.ndarray, w: int, cfg: ScanConfig | None = None):
    """Scan with ``F_0``. Returns ``(score, (a, b))``, b exclusive-safe.

    O(n * r) with prefix sums, where r is the range cap.
    """
    cfg = cfg or ScanConfig()
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return 0.0, (0, 0)

    ps1 = np.concatenate([[0.0], np.cumsum(x)])
    ps2 = np.concatenate([[0.0], np.cumsum(x * x)])
    sra = _sse_constant(ps1, ps2, 0, n)
    if sra <= 1e-18:
        return 0.0, (0, 0)

    lo_bound = cfg.buffer
    hi_bound = n - cfg.buffer
    w_min = max(1, int(cfg.min_width * n))
    w_max = min(3 * int(w), int(cfg.max_width * n), hi_bound - lo_bound)

    best, best_ab = -np.inf, (0, 0)
    for a in range(lo_bound, hi_bound - w_min + 1):
        b_hi = min(a + w_max, hi_bound)
        for b in range(a + w_min, b_hi + 1):
            s = 1.0 - (
                _sse_constant(ps1, ps2, a, b)
                + _sse_outside_constant(ps1, ps2, n, a, b)
            ) / sra
            if s > best:
                best, best_ab = s, (a, b - 1)
    return (float(best), best_ab) if np.isfinite(best) else (0.0, (0, 0))


def scan_poly(x: np.ndarray, w: int, degree: int = 1, cfg: ScanConfig | None = None):
    """Scan with ``F_d``, a degree-*d* polynomial fit inside and outside."""
    cfg = cfg or ScanConfig()
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 * (degree + 2):
        return 0.0, (0, 0)

    idx = np.arange(n)
    sra = _sse_poly(x, idx, degree)
    if sra <= 1e-18:
        return 0.0, (0, 0)

    lo_bound, hi_bound = cfg.buffer, n - cfg.buffer
    w_min = max(degree + 2, int(cfg.min_width * n))
    w_max = min(3 * int(w), int(cfg.max_width * n), hi_bound - lo_bound)

    best, best_ab = -np.inf, (0, 0)
    for a in range(lo_bound, hi_bound - w_min + 1):
        for b in range(a + w_min, min(a + w_max, hi_bound) + 1):
            inside = idx[a:b]
            outside = np.concatenate([idx[:a], idx[b:]])
            if outside.size <= degree + 1:
                continue
            s = 1.0 - (
                _sse_poly(x, inside, degree) + _sse_poly(x, outside, degree)
            ) / sra
            if s > best:
                best, best_ab = s, (a, b - 1)
    return (float(best), best_ab) if np.isfinite(best) else (0.0, (0, 0))


def deramp(x: np.ndarray) -> np.ndarray:
    """Subtract a global degree-1 fit.

    Not equivalent to ``F_1``: that fits a separate line inside and outside
    the candidate interval, whereas this fits one line over the whole signal,
    so a real step tilts that line and is partly absorbed before ``F_0`` sees
    it. Cheaper, strictly weaker.
    """
    x = np.asarray(x, dtype=float)
    t = np.arange(x.size, dtype=float)
    good = np.isfinite(x)
    if good.sum() <= 2:
        return x
    return x - np.polyval(np.polyfit(t[good], x[good], 1), t)


def epidemic_score(x: np.ndarray, a: int, b: int) -> float:
    """Score an interval under the constant family.

    Used to put change-point baselines (which return boundaries, not scores)
    onto the same [0, 1] scale as ``F_0``, so their outputs are comparable.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0 or a > b:
        return float("-inf")
    sra = float(((x - x.mean()) ** 2).sum())
    if sra < 1e-18:
        return 0.0
    ins = x[a : b + 1]
    out = np.concatenate([x[:a], x[b + 1 :]])
    sse_in = float(((ins - ins.mean()) ** 2).sum()) if ins.size else 0.0
    sse_out = float(((out - out.mean()) ** 2).sum()) if out.size else 0.0
    return 1.0 - (sse_in + sse_out) / sra
