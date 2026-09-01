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


class _PolyMoments:
    """Precomputed moment prefix sums for degree-*d* least squares.

    Fitting a degree-d polynomial to a contiguous range needs only the moment
    sums over that range::

        M[k, l] = sum t^(k+l)      v[k] = sum t^k y      sy2 = sum y^2

    Prefix sums over ``t^p`` for p up to 2d and over ``t^p y`` for p up to d
    make every one of those an O(1) lookup, so a fit costs O(d^2) to assemble
    plus O(d^3) to solve -- independent of how many samples the range holds.
    Without this the cost is O(m) per fit and the scan degrades to O(n r m).

    Moments are additive over disjoint ranges, so the outside of an interval
    (two flanks) costs the same as the inside.

    ``t`` is normalised to [0, 1). Raw indices would put ``t^(2d)`` at 1920^6
    for a 1920-sample signal at d=3, which is representable but badly
    conditioned; normalising keeps every moment O(1) in magnitude. A small
    ridge on the diagonal covers the remaining ill-conditioning of the
    Hilbert-like moment matrix.
    """

    __slots__ = ("_sy2", "_tp", "_tpy", "degree", "n", "reg")

    def __init__(self, x: np.ndarray, degree: int, reg: float = 1e-10):
        y = np.asarray(x, dtype=float)
        n = y.size
        self.degree = int(degree)
        self.reg = float(reg)
        self.n = n
        t = np.arange(n, dtype=float) / max(n, 1)

        powers = [np.ones(n)]
        for _ in range(2 * self.degree):
            powers.append(powers[-1] * t)
        self._tp = [np.concatenate([[0.0], np.cumsum(p)]) for p in powers]
        self._tpy = [np.concatenate([[0.0], np.cumsum(powers[k] * y)])
                     for k in range(self.degree + 1)]
        self._sy2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def _blocks(self, ranges):
        d = self.degree
        m = np.empty((d + 1, d + 1))
        v = np.empty(d + 1)
        sy2 = 0.0
        count = 0
        raw_t = [0.0] * (2 * d + 1)
        raw_ty = [0.0] * (d + 1)
        for lo, hi in ranges:                      # hi exclusive
            if lo >= hi:
                continue
            count += hi - lo
            for p in range(2 * d + 1):
                raw_t[p] += self._tp[p][hi] - self._tp[p][lo]
            for k in range(d + 1):
                raw_ty[k] += self._tpy[k][hi] - self._tpy[k][lo]
            sy2 += self._sy2[hi] - self._sy2[lo]
        for k in range(d + 1):
            v[k] = raw_ty[k]
            for l in range(d + 1):
                m[k, l] = raw_t[k + l]
        return m, v, sy2, count

    def sse(self, ranges) -> float:
        """SSE of the least-squares fit over the union of *ranges*."""
        d = self.degree
        m, v, sy2, count = self._blocks(ranges)
        if count <= d + 1:
            return 0.0
        m.flat[:: d + 2] += self.reg
        try:
            alpha = np.linalg.solve(m, v)
        except np.linalg.LinAlgError:
            return float(sy2 - (v[0] ** 2 / count if count else 0.0))
        # At the least-squares solution M a = v, so a^T M a = a . v and
        # SSE = sum y^2 - 2 a.v + a^T M a collapses to sum y^2 - a.v.
        return max(0.0, float(sy2 - alpha @ v))


def scan_constant(x: np.ndarray, w: int, r: int | None = None,
                  cfg: ScanConfig | None = None):
    """Scan with ``F_0``.

    *w* is unused by the constant model itself; it only sets the default range
    cap ``r = 3w``, the longest interval considered. O(n r) with prefix sums.
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

    r = 3 * int(w) if r is None else int(r)
    lo_bound = cfg.buffer
    hi_bound = n - cfg.buffer
    w_min = max(1, int(cfg.min_width * n))
    w_max = min(r, int(cfg.max_width * n), hi_bound - lo_bound)

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


def scan_poly(x, w: int, r: int | None = None, degree: int = 1,
              cfg: ScanConfig | None = None, reg: float = 1e-10):
    """Scan with ``F_d``, a degree-*d* polynomial fitted inside and outside.

    *w* sets the default range cap ``r = 3w``. O(n r d^3) using
    :class:`_PolyMoments`; the per-interval cost does not depend on the
    interval length.
    """
    cfg = cfg or ScanConfig()
    y = np.asarray(x, dtype=float)
    n = y.size
    if n < 2 * (degree + 2):
        return 0.0, (0, 0)

    mom = _PolyMoments(y, degree, reg)
    sra = mom.sse([(0, n)])
    if sra <= 1e-18:
        return 0.0, (0, 0)

    r = 3 * int(w) if r is None else int(r)
    lo_b, hi_b = cfg.buffer, n - cfg.buffer
    w_min = max(degree + 2, int(cfg.min_width * n))
    w_max = min(r, int(cfg.max_width * n) or n, hi_b - lo_b)

    best, best_ab = -np.inf, (0, 0)
    for a in range(lo_b, hi_b - w_min + 1):
        for b in range(a + w_min, min(a + w_max, hi_b) + 1):
            if n - (b - a) <= degree + 1:
                continue
            s = 1.0 - (
                mom.sse([(a, b)]) + mom.sse([(0, a), (b, n)])
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
