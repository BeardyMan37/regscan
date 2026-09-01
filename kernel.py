"""Kernel function families.

``F_KR`` -- Nadaraya-Watson kernel regression, the proposed method. The fit at
index *i* restricted to an index set *S* is the kernel-weighted mean of the
observations in *S* near *i*::

    yhat_i = sum_{j in S, |i-j| <= r} k[|i-j|] y_j / sum_{j in S, |i-j| <= r} k[|i-j|]

Truncating at ``r = truncation * w`` is what makes the family tractable: a
Gaussian at 3w has decayed to e^-9, so dropping the tail costs nothing
measurable while the cost of a fit falls from O(n) to O(w).

The scan does not evaluate that formula per interval. It grows an interval one
sample at a time and updates the fit in O(r) per step, using the state in
:mod:`regscan.kernel_state`. That is the difference between a scan that is
usable on long signals and one that is not: at n = 800 the kernel family
completes in under half a second, roughly six times faster than the degree-1
polynomial family, despite fitting a far richer model.

``F_KRR`` -- kernel ridge regression. It solves an m x m linear system per
candidate interval, giving O(n^4 w) over a scan, which limits it to short
signals.
"""

from __future__ import annotations

import numpy as np

from .config import ScanConfig
from .kernel_state import (
    buf_add,
    buf_init,
    nin_din_add,
    nin_din_init_full,
    sse_out_add,
    sse_out_from_nin_din,
)
from .window import decimate, sr_factor


def truncated_kernel_vector(w: float, r: int, kind: str) -> np.ndarray:
    """One-sided kernel ``k[d]`` for ``d = 0 .. r``.

    Gaussian is ``exp(-d^2 / w^2)``, Laplace ``exp(-d / w)``. Only offsets up
    to *r* are represented; beyond that the weight is treated as exactly zero,
    which is what turns an O(n) fit into an O(w) one.
    """
    w = max(float(w), 1e-12)
    d = np.arange(int(r) + 1, dtype=float)
    if kind == "gaussian":
        return np.exp(-(d * d) / (w * w))
    if kind == "laplace":
        return np.exp(-d / w)
    raise ValueError(f"unknown kernel {kind!r}")


def accumulate(x: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All-points kernel numerator and denominator.

    ``num[i] = sum_j k[|i-j|] x[j]`` and ``den[i] = sum_j k[|i-j|]`` over
    ``|i-j| <= r``. Out-of-range neighbours contribute nothing, which is what
    the zero padding in the convolution gives.
    """
    sym = np.concatenate([k[:0:-1], k])
    num = np.convolve(np.asarray(x, dtype=float), sym, mode="same")
    den = np.convolve(np.ones(x.size), sym, mode="same")
    return num, den


def _keep_indices(n: int, buffer: int) -> np.ndarray:
    """Indices surviving the edge buffer.

    The buffer *excises* samples rather than merely forbidding interval
    placement there: the fit, ``SR_A`` and the outside residuals are all
    computed on the trimmed signal, so a buffered scan is a scan of a shorter
    signal.
    """
    if buffer <= 0:
        return np.arange(n, dtype=np.int64)
    if 2 * buffer >= n:
        return np.empty(0, dtype=np.int64)
    return np.arange(buffer, n - buffer, dtype=np.int64)


def _spans(nf: int, r: int, cfg: ScanConfig) -> tuple[int, int]:
    """Inclusive bounds on ``j - i`` for a candidate interval ``[i, j]``."""
    span_max = min(int(r), int(cfg.max_width * nf) or nf, nf - 1)
    span_min = max(1, int(cfg.min_width * nf) - 1)
    return span_min, span_max


def _fit_stats(y: np.ndarray, k: np.ndarray):
    """All-points kernel sums, the unrestricted fit, and ``SR_A``."""
    numer, denom = accumulate(y, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.where(denom > 1e-12, numer / denom, 0.0)
    return numer, denom, float(((y - pred) ** 2).sum())


def _search(y, numer, denom, ssra, k, span_min, span_max):
    """Best interval by growing each start one sample at a time.

    For a fixed start ``i`` the interval ``[i, j]`` is built once at ``j =
    i+1`` and extended from there. Each extension costs O(r):

    * ``nin``/``din`` gain the new point's kernel contribution;
    * ``buf``/``sse_in`` gain the point and adjust its inside neighbours;
    * ``sse_out`` is corrected for the point leaving the outside and for its
      outside neighbours.

    ``sse_out`` is adjusted rather than recomputed, so error accumulates along
    a run of extensions; it is recomputed exactly every ``refresh`` steps to
    bound the drift without giving back the speed.

    Returns ``(score, (i, j))`` in *y*'s own indices, or ``(0.0, (0, -1))``
    when no interval beats a null split.
    """
    nf = y.size
    refresh = max(1, span_max)
    cap = nf + 1
    buf_idxs = np.empty(cap, dtype=np.int64)
    buf_num = np.empty(cap, dtype=np.float64)
    buf_den = np.empty(cap, dtype=np.float64)
    nin = np.zeros(nf, dtype=np.float64)
    din = np.zeros(nf, dtype=np.float64)

    best_sc, best_win = 0.0, (0, -1)
    for i in range(nf - 1):
        started = False
        m = 0
        sse_in = sse_out = 0.0
        steps = 0
        for j in range(i + 1, min(i + span_max, nf - 1) + 1):
            if not started:
                pair = np.array([i, j], dtype=np.int64)
                nin_din_init_full(y, pair, k, nin, din)
                m, sse_in = buf_init(y, pair, k, buf_idxs, buf_num, buf_den)
                sse_out = sse_out_from_nin_din(
                    y, numer, denom, nin, din, buf_idxs, m)
                steps = 0
                started = True
            else:
                nin_din_add(y, nin, din, j, k)
                m, sse_in = buf_add(y, j, k, buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = sse_out_add(
                    y, numer, denom, nin, din, buf_idxs, m, j, k, sse_out)
                steps += 1
                if steps >= refresh:
                    sse_out = sse_out_from_nin_din(
                        y, numer, denom, nin, din, buf_idxs, m)
                    steps = 0

            if j - i < span_min:
                continue
            sc = 1.0 - (sse_in + sse_out) / ssra
            if sc > best_sc:
                best_sc, best_win = sc, (i, j)
    return best_sc, best_win


def _score_interval(y, numer, denom, ssra, k, a, b, nin, din,
                    buf_idxs, buf_num, buf_den) -> float:
    """Exact score of one interval, built from scratch. O(n + m*r)."""
    inside = np.arange(a, b + 1, dtype=np.int64)
    nin_din_init_full(y, inside, k, nin, din)
    m, sse_in = buf_init(y, inside, k, buf_idxs, buf_num, buf_den)
    sse_out = sse_out_from_nin_din(y, numer, denom, nin, din, buf_idxs, m)
    return 1.0 - (sse_in + sse_out) / ssra


def _refine(y, numer, denom, ssra, k, a_sr, b_sr, factor, span_min, span_max):
    """Recover exact endpoints from a decimated result.

    Decimation maps a block of *factor* original samples to one coarse sample,
    so a coarse interval ``[a_sr, b_sr]`` locates each endpoint only to within
    a block. Every original sample near those blocks is scored here at full
    resolution and the best pair wins.

    The search spans one block either side of each coarse endpoint, not just
    the block itself. Block means smooth an anomaly's edges, so when an edge
    falls near a block boundary the coarse pass picks the neighbouring block
    roughly as often as the right one; searching only inside it leaves that
    unrecoverable. Measured over 40 randomly placed anomalies, confining the
    search to the winning block reproduced the exact interval 20/40 times at
    factor 4 and 11/40 at factor 8, against 40/40 with the neighbours
    included. The extra samples cost nothing worth counting -- refinement is a
    few dozen interval scores set against a coarse scan of the whole signal.
    """
    nf = y.size
    a_lo = max(0, (a_sr - 1) * factor)
    a_hi = min((a_sr + 2) * factor - 1, nf - 1)
    b_lo = max(0, (b_sr - 1) * factor)
    b_hi = min((b_sr + 2) * factor - 1, nf - 1)

    cap = nf + 1
    buf_idxs = np.empty(cap, dtype=np.int64)
    buf_num = np.empty(cap, dtype=np.float64)
    buf_den = np.empty(cap, dtype=np.float64)
    nin = np.zeros(nf, dtype=np.float64)
    din = np.zeros(nf, dtype=np.float64)

    best_sc, best_ab = 0.0, (0, -1)
    for a in range(a_lo, a_hi + 1):
        for b in range(max(a + span_min, b_lo), b_hi + 1):
            if b - a > span_max or b >= nf:
                continue
            sc = _score_interval(y, numer, denom, ssra, k, a, b,
                                 nin, din, buf_idxs, buf_num, buf_den)
            if sc > best_sc:
                best_sc, best_ab = sc, (a, b)
    return best_sc, best_ab


def resolve_factor(n: int, cfg: ScanConfig) -> int:
    """The decimation factor this config asks for on a length-*n* signal."""
    if cfg.super_resolution == "auto":
        return sr_factor(n, base=cfg.sr_base, cap=cfg.sr_cap)
    return int(cfg.super_resolution)


def scan_nwkr(x, w: int, r: int | None = None, cfg: ScanConfig | None = None):
    """Scan with ``F_KR``.

    Parameters
    ----------
    x: the signal.
    w: kernel bandwidth, in samples of *x*.
    r: longest interval considered, in samples. Defaults to ``3 * w``.
    cfg: :class:`ScanConfig`; ``kernel``, ``truncation``, ``buffer``, the width
        bounds and the super-resolution settings all come from here.

    With ``cfg.super_resolution`` above 1 the signal is block-meaned by that
    factor and scanned coarsely, then the winning blocks are searched at full
    resolution by :func:`_refine`. Bandwidth and range scale with the factor,
    so ``w`` and ``r`` keep their meaning in samples of *x* either way.

    Returns ``(score, (a, b))`` in *x*'s own indices, ``b`` inclusive. The
    score is floored at 0: when no interval beats a null split the result is
    ``(0.0, (0, -1))``.
    """
    cfg = cfg or ScanConfig()
    y = np.asarray(x, dtype=np.float64)
    n = y.size
    keep = _keep_indices(n, cfg.buffer)
    if keep.size < 2:
        return 0.0, (0, -1)

    srow = np.ascontiguousarray(y[keep])
    nf = srow.size
    w = int(w)
    r = 3 * w if r is None else int(r)
    if w < 1 or r < 1:
        raise ValueError("w and r must be >= 1")

    k = truncated_kernel_vector(w, cfg.truncation * w, cfg.kernel)
    numer, denom, sra_raw = _fit_stats(srow, k)
    # A signal the kernel already fits exactly leaves nothing for an interval
    # to explain. Without this guard the ratio is 0/epsilon and every interval
    # scores 1.0, so a constant signal would report a perfect detection.
    if sra_raw <= 1e-18:
        return 0.0, (0, -1)
    ssra = max(sra_raw, 1e-12)

    span_min, span_max = _spans(nf, r, cfg)
    if span_max < span_min:
        return 0.0, (0, -1)

    factor = resolve_factor(nf, cfg)
    if factor > 1 and nf // factor >= 4:
        coarse = decimate(srow, factor)
        w_c = max(1, round(w / factor))
        r_c = max(1, round(r / factor))
        k_c = truncated_kernel_vector(w_c, cfg.truncation * w_c, cfg.kernel)
        num_c, den_c, sra_c = _fit_stats(coarse, k_c)
        if sra_c > 1e-18:
            smin_c, smax_c = _spans(coarse.size, r_c, cfg)
            if smax_c >= smin_c:
                _, (i_c, j_c) = _search(
                    coarse, num_c, den_c, max(sra_c, 1e-12), k_c, smin_c, smax_c)
                if j_c >= i_c:
                    sc, (a, b) = _refine(srow, numer, denom, ssra, k,
                                         i_c, j_c, factor, span_min, span_max)
                    if b < a:
                        return 0.0, (0, -1)
                    return float(sc), (int(keep[a]), int(keep[b]))
        return 0.0, (0, -1)

    sc, (a, b) = _search(srow, numer, denom, ssra, k, span_min, span_max)
    if b < a:
        return 0.0, (0, -1)
    return float(sc), (int(keep[a]), int(keep[b]))


def _krr_sse(y: np.ndarray, idx: np.ndarray, kind: str, scale: float,
             reg: float) -> float:
    """SSE of a kernel ridge fit on *idx*, evaluated on *idx*."""
    m = idx.size
    if m < 2:
        return 0.0
    d = idx[:, None] - idx[None, :]
    if kind == "gaussian":
        kk = np.exp(-((d / scale) ** 2))
    elif kind == "laplace":
        kk = np.exp(-np.abs(d) / scale)
    else:
        raise ValueError(f"unknown kernel {kind!r}")
    a = kk.copy()
    a.flat[:: m + 1] += reg
    try:
        alpha = np.linalg.solve(a, y[idx])
    except np.linalg.LinAlgError:
        return float(((y[idx] - y[idx].mean()) ** 2).sum())
    resid = y[idx] - kk @ alpha
    return float(resid @ resid)


def scan_krr(x, w: int, r: int | None = None, cfg: ScanConfig | None = None,
             reg: float = 1e-1, scale: float | None = None, stride: int = 1):
    """Scan with ``F_KRR``.

    Solves an ``m x m`` linear system for every candidate interval, so a full
    scan is O(n^4 w) -- orders of magnitude slower than the other families and
    impractical much beyond n of a few hundred.

    *stride* subsamples candidate endpoints. It defaults to 1 (exhaustive)
    because subsampling can step straight over the true interval and return a
    lower-scoring one instead, which looks like a detection failure rather
    than a search shortcut. Raise it only when you would rather have a fast
    approximate answer and know that is what you are getting.
    """
    cfg = cfg or ScanConfig()
    y = np.asarray(x, dtype=float)
    n = y.size
    if n < 8:
        return 0.0, (0, 0)
    scale = float(scale if scale is not None else max(w, 1))
    stride = max(1, int(stride))

    idx = np.arange(n)
    sra = _krr_sse(y, idx, cfg.kernel, scale, reg)
    if sra <= 1e-18:
        return 0.0, (0, 0)

    r = 3 * int(w) if r is None else int(r)
    lo_b, hi_b = cfg.buffer, n - 1 - cfg.buffer
    w_min = max(2, int(cfg.min_width * n))
    w_max = min(r, int(cfg.max_width * n) or n, hi_b - lo_b + 1)

    best, best_ab = -np.inf, (0, 0)
    for a in range(lo_b, hi_b - w_min + 2, stride):
        for b in range(a + w_min - 1, min(a + w_max - 1, hi_b) + 1, stride):
            inside = idx[a : b + 1]
            outside = np.concatenate([idx[:a], idx[b + 1 :]])
            if outside.size < 2:
                continue
            s = 1.0 - (
                _krr_sse(y, inside, cfg.kernel, scale, reg)
                + _krr_sse(y, outside, cfg.kernel, scale, reg)
            ) / sra
            if s > best:
                best, best_ab = s, (a, b)
    return (float(best), best_ab) if np.isfinite(best) else (0.0, (0, 0))
