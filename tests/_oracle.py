"""A deliberately naive kernel scan, used only to check the fast one."""

from __future__ import annotations

import numpy as np

from regscan.config import ScanConfig
from regscan.kernel import _keep_indices, _spans, accumulate, truncated_kernel_vector


def direct_nwkr(x, w: int, r: int | None = None, cfg: ScanConfig | None = None):
    """The same statistic, computed the slow and obvious way.

    Every index is refitted from scratch for every candidate interval, so
    there is no carried state to get wrong. This is the oracle the incremental
    scan is checked against.
    """
    cfg = cfg or ScanConfig()
    y = np.asarray(x, dtype=np.float64)
    n = y.size
    keep = _keep_indices(n, cfg.buffer)
    if keep.size < 2:
        return 0.0, (0, -1)

    srow = y[keep]
    nf = srow.size
    w = int(w)
    span_r = 3 * w if r is None else int(r)
    k = truncated_kernel_vector(w, cfg.truncation * w, cfg.kernel)
    trunc = k.size - 1

    numer_all, denom_all = accumulate(srow, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred_all = np.where(denom_all > 1e-12, numer_all / denom_all, 0.0)
    sra_raw = float(((srow - pred_all) ** 2).sum())
    # Match scan_nwkr: an exactly-fitted signal has nothing to explain.
    if sra_raw <= 1e-18:
        return 0.0, (0, -1)
    ssra = max(sra_raw, 1e-12)

    def sse(idxs: np.ndarray) -> float:
        total = 0.0
        for i in idxs:
            d = np.abs(idxs - i)
            near = d <= trunc
            wts = k[d[near]]
            den = wts.sum()
            pred = float(wts @ srow[idxs[near]] / den) if den > 1e-12 else 0.0
            total += (srow[i] - pred) ** 2
        return float(total)

    span_min, span_max = _spans(nf, span_r, cfg)
    all_idx = np.arange(nf, dtype=np.int64)
    best_sc, best_win = 0.0, (0, -1)
    for i in range(nf - 1):
        for j in range(i + span_min, min(i + span_max, nf - 1) + 1):
            inside = all_idx[i : j + 1]
            outside = np.concatenate([all_idx[:i], all_idx[j + 1 :]])
            sc = 1.0 - (sse(inside) + sse(outside)) / ssra
            if sc > best_sc:
                best_sc, best_win = sc, (i, j)

    if best_win[1] < best_win[0]:
        return 0.0, (0, -1)
    return float(best_sc), (int(keep[best_win[0]]), int(keep[best_win[1]]))
