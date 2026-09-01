"""Incremental state for the kernel scan.

A candidate interval grows one sample at a time. Rebuilding the fit at each
step would cost O(n); these routines update it in O(r) instead, where ``r`` is
the kernel truncation radius (``len(k) - 1``).

Three pieces of state are carried:

``nin`` / ``din``
    The kernel-weighted numerator and denominator that the *inside* points
    contribute to every index of the signal. Subtracting them from the
    all-points totals leaves exactly the outside points' contribution, so the
    outside fit at any index is one subtraction rather than a refit. This is
    what makes ``sse_out`` affordable.

``buf_idxs`` / ``buf_num`` / ``buf_den``
    The inside points in sorted order, each with its kernel sums restricted to
    the other inside points. Dividing gives the inside fit, and hence
    ``sse_in``.

``sse_out``
    Adjusted by ``sse_out_add`` / ``sse_out_remove`` when a point crosses the
    boundary. Only that point and its O(r) neighbours change, because the
    kernel is zero beyond ``r``; every other index keeps the residual it had.

Adding a point touches ``nin``/``din`` first, then ``buf``, then ``sse_out`` --
the ``sse_out`` update reads the already-updated ``nin``/``din`` and needs the
new point to be present in ``buf_idxs``, so the order is not interchangeable.

``sse_out`` is adjusted rather than recomputed, so floating-point error
accumulates across a run of updates. The caller recomputes it exactly from
:func:`sse_out_from_nin_din` on a fixed cadence to bound that drift.
"""

from __future__ import annotations

import numpy as np
from numba import njit as _njit


def njit(fn):
    return _njit(cache=True, fastmath=True)(fn)


@njit
def _is_inside(buf_idxs: np.ndarray, m: int, idx: int) -> bool:
    """Binary search: True if idx is in buf_idxs[0..m-1] (sorted)."""
    lo = 0; hi = m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < idx: lo = mid + 1
        else: hi = mid
    return lo < m and buf_idxs[lo] == idx


# ---------------------------------------------------------------------------
# nin/din management — O(r) per add/remove, O(n) for full init
# ---------------------------------------------------------------------------

@njit
def nin_din_init_full(
    x:        np.ndarray,
    idxs_in:  np.ndarray,
    k:        np.ndarray,
    nin:      np.ndarray,
    din:      np.ndarray,
) -> None:
    """Zero and rebuild nin/din from scratch. O(n + m*r). Called at run boundaries only."""
    n = x.shape[0]; r = k.shape[0] - 1
    for i in range(n):
        nin[i] = 0.0; din[i] = 0.0
    for p in range(idxs_in.shape[0]):
        j = int(idxs_in[p]); xj = x[j]
        i = j
        while i >= 0:
            d = j - i
            if d > r: break
            nin[i] += k[d] * xj; din[i] += k[d]; i -= 1
        i = j + 1
        while i < n:
            d = i - j
            if d > r: break
            nin[i] += k[d] * xj; din[i] += k[d]; i += 1


@njit
def nin_din_add(x: np.ndarray, nin: np.ndarray, din: np.ndarray, idx: int, k: np.ndarray) -> None:
    """Add idx's contribution to nin/din in-place. O(r)."""
    r = k.shape[0] - 1; n = nin.shape[0]; xj = x[idx]
    i = idx
    while i >= 0:
        d = idx - i
        if d > r: break
        nin[i] += k[d] * xj; din[i] += k[d]; i -= 1
    i = idx + 1
    while i < n:
        d = i - idx
        if d > r: break
        nin[i] += k[d] * xj; din[i] += k[d]; i += 1


@njit
def nin_din_remove(x: np.ndarray, nin: np.ndarray, din: np.ndarray, idx: int, k: np.ndarray) -> None:
    """Remove idx's contribution from nin/din in-place. O(r)."""
    r = k.shape[0] - 1; n = nin.shape[0]; xj = x[idx]
    i = idx
    while i >= 0:
        d = idx - i
        if d > r: break
        nin[i] -= k[d] * xj; din[i] -= k[d]; i -= 1
    i = idx + 1
    while i < n:
        d = i - idx
        if d > r: break
        nin[i] -= k[d] * xj; din[i] -= k[d]; i += 1


# ---------------------------------------------------------------------------
# sse_out from nin/din — O(n) exact recompute
# ---------------------------------------------------------------------------

@njit
def sse_out_from_nin_din(
    x: np.ndarray, numer_all: np.ndarray, denom_all: np.ndarray,
    nin: np.ndarray, din: np.ndarray,
    buf_idxs: np.ndarray, m: int,
) -> float:
    """Exact sse_out in O(n). Called at run boundaries and for periodic refresh."""
    n = x.shape[0]; eps = 1e-12; sse = 0.0; p = 0
    for i in range(n):
        while p < m and buf_idxs[p] < i: p += 1
        if p < m and buf_idxs[p] == i: continue
        den_out = denom_all[i] - din[i]
        num_out = numer_all[i] - nin[i]
        pred    = num_out / den_out if den_out > eps else 0.0
        sse    += (x[i] - pred) ** 2
    return sse


# ---------------------------------------------------------------------------
# sse_out incremental update — O(r), skipping inside points
# ---------------------------------------------------------------------------

@njit
def sse_out_add(
    x: np.ndarray, numer_all: np.ndarray, denom_all: np.ndarray,
    nin: np.ndarray, din: np.ndarray,
    buf_idxs: np.ndarray, m_new: int,
    new_idx: int, k: np.ndarray, sse_out: float,
) -> float:
    """
    Update sse_out after new_idx moved from outside to inside.
    nin/din must already be updated (nin_din_add called first).
    buf_idxs[0..m_new-1] must already contain new_idx.
    O(r).
    """
    r = k.shape[0] - 1; eps = 1e-12; n = nin.shape[0]; x_new = x[new_idx]

    # Remove new_idx's own outside contribution
    nin_old = nin[new_idx] - k[0] * x_new
    din_old = din[new_idx] - k[0]
    den_out_old = denom_all[new_idx] - din_old
    num_out_old = numer_all[new_idx] - nin_old
    pred_old = num_out_old / den_out_old if den_out_old > eps else 0.0
    sse_out -= (x_new - pred_old) ** 2

    # Update O(r) outside neighbours
    i = new_idx - 1
    while i >= 0:
        d = new_idx - i
        if d > r: break
        if not _is_inside(buf_idxs, m_new, i):
            w = k[d]
            nin_old_i   = nin[i] - w * x_new
            din_old_i   = din[i] - w
            den_out_old = denom_all[i] - din_old_i
            num_out_old = numer_all[i] - nin_old_i
            pred_old_i  = num_out_old / den_out_old if den_out_old > eps else 0.0
            den_out_new = denom_all[i] - din[i]
            num_out_new = numer_all[i] - nin[i]
            pred_new_i  = num_out_new / den_out_new if den_out_new > eps else 0.0
            sse_out    -= (x[i] - pred_old_i) ** 2
            sse_out    += (x[i] - pred_new_i) ** 2
        i -= 1

    i = new_idx + 1
    while i < n:
        d = i - new_idx
        if d > r: break
        if not _is_inside(buf_idxs, m_new, i):
            w = k[d]
            nin_old_i   = nin[i] - w * x_new
            din_old_i   = din[i] - w
            den_out_old = denom_all[i] - din_old_i
            num_out_old = numer_all[i] - nin_old_i
            pred_old_i  = num_out_old / den_out_old if den_out_old > eps else 0.0
            den_out_new = denom_all[i] - din[i]
            num_out_new = numer_all[i] - nin[i]
            pred_new_i  = num_out_new / den_out_new if den_out_new > eps else 0.0
            sse_out    -= (x[i] - pred_old_i) ** 2
            sse_out    += (x[i] - pred_new_i) ** 2
        i += 1

    return sse_out


@njit
def sse_out_remove(
    x: np.ndarray, numer_all: np.ndarray, denom_all: np.ndarray,
    nin: np.ndarray, din: np.ndarray,
    buf_idxs: np.ndarray, m_new: int,
    rem_idx: int, k: np.ndarray, sse_out: float,
) -> float:
    """
    Update sse_out after rem_idx moved from inside to outside.
    nin/din must already be updated (nin_din_remove called first).
    buf_idxs[0..m_new-1] must NOT contain rem_idx.
    O(r).
    """
    r = k.shape[0] - 1; eps = 1e-12; n = nin.shape[0]; x_rem = x[rem_idx]

    # Add rem_idx's new outside contribution
    den_out_new = denom_all[rem_idx] - din[rem_idx]
    num_out_new = numer_all[rem_idx] - nin[rem_idx]
    pred_new    = num_out_new / den_out_new if den_out_new > eps else 0.0
    sse_out    += (x_rem - pred_new) ** 2

    # Update O(r) outside neighbours
    i = rem_idx - 1
    while i >= 0:
        d = rem_idx - i
        if d > r: break
        if not _is_inside(buf_idxs, m_new, i):
            w = k[d]
            nin_old_i   = nin[i] + w * x_rem
            din_old_i   = din[i] + w
            den_out_old = denom_all[i] - din_old_i
            num_out_old = numer_all[i] - nin_old_i
            pred_old_i  = num_out_old / den_out_old if den_out_old > eps else 0.0
            den_out_new = denom_all[i] - din[i]
            num_out_new = numer_all[i] - nin[i]
            pred_new_i  = num_out_new / den_out_new if den_out_new > eps else 0.0
            sse_out    -= (x[i] - pred_old_i) ** 2
            sse_out    += (x[i] - pred_new_i) ** 2
        i -= 1

    i = rem_idx + 1
    while i < n:
        d = i - rem_idx
        if d > r: break
        if not _is_inside(buf_idxs, m_new, i):
            w = k[d]
            nin_old_i   = nin[i] + w * x_rem
            din_old_i   = din[i] + w
            den_out_old = denom_all[i] - din_old_i
            num_out_old = numer_all[i] - nin_old_i
            pred_old_i  = num_out_old / den_out_old if den_out_old > eps else 0.0
            den_out_new = denom_all[i] - din[i]
            num_out_new = numer_all[i] - nin[i]
            pred_new_i  = num_out_new / den_out_new if den_out_new > eps else 0.0
            sse_out    -= (x[i] - pred_old_i) ** 2
            sse_out    += (x[i] - pred_new_i) ** 2
        i += 1

    return sse_out


# ---------------------------------------------------------------------------
# Inside buf/sse_in management — O(r) per add/remove
# ---------------------------------------------------------------------------

@njit
def buf_init(
    x: np.ndarray, idxs_in: np.ndarray, k: np.ndarray,
    buf_idxs: np.ndarray, buf_num: np.ndarray, buf_den: np.ndarray,
) -> tuple[int, float]:
    """
    Init buf/sse_in only (does NOT touch nin/din).
    O(m0 * r). Called at the first j of each outer step.
    """
    m0 = idxs_in.shape[0]; r = k.shape[0] - 1; eps = 1e-12
    for t in range(m0):
        buf_idxs[t] = idxs_in[t]
    for u in range(m0):
        i = int(buf_idxs[u]); s_num = 0.0; s_den = 0.0
        v = u
        while v >= 0:
            d = i - int(buf_idxs[v])
            if d > r: break
            s_num += k[d] * x[int(buf_idxs[v])]; s_den += k[d]; v -= 1
        v = u + 1
        while v < m0:
            d = int(buf_idxs[v]) - i
            if d > r: break
            s_num += k[d] * x[int(buf_idxs[v])]; s_den += k[d]; v += 1
        buf_num[u] = s_num; buf_den[u] = s_den
    sse_in = 0.0
    for u in range(m0):
        d = buf_den[u]; pred = buf_num[u] / d if d > eps else 0.0
        sse_in += (x[int(buf_idxs[u])] - pred) ** 2
    return m0, float(sse_in)


@njit
def buf_add(
    x: np.ndarray, new_idx: int, k: np.ndarray,
    buf_idxs: np.ndarray, buf_num: np.ndarray, buf_den: np.ndarray,
    m: int, sse_in: float,
) -> tuple[int, float]:
    """Add new_idx to buf/sse_in in-place. O(r). Does NOT touch nin/din."""
    r = k.shape[0] - 1; eps = 1e-12; x_new = x[new_idx]
    lo = 0; hi = m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < new_idx: lo = mid + 1
        else: hi = mid
    ins = lo
    for t in range(m - 1, ins - 1, -1):
        buf_idxs[t+1] = buf_idxs[t]; buf_num[t+1] = buf_num[t]; buf_den[t+1] = buf_den[t]
    buf_idxs[ins] = new_idx; m_new = m + 1

    v = ins - 1
    while v >= 0:
        d = new_idx - int(buf_idxs[v])
        if d > r: break
        w = k[d]
        od = buf_den[v]; op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] += w * x_new; buf_den[v] += w
        nd = buf_den[v]; np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2
        v -= 1
    v = ins + 1
    while v < m_new:
        d = int(buf_idxs[v]) - new_idx
        if d > r: break
        w = k[d]
        od = buf_den[v]; op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] += w * x_new; buf_den[v] += w
        nd = buf_den[v]; np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2
        v += 1

    s_num = k[0] * x_new; s_den = k[0]
    v = ins - 1
    while v >= 0:
        d = new_idx - int(buf_idxs[v])
        if d > r: break
        s_num += k[d] * x[int(buf_idxs[v])]; s_den += k[d]; v -= 1
    v = ins + 1
    while v < m_new:
        d = int(buf_idxs[v]) - new_idx
        if d > r: break
        s_num += k[d] * x[int(buf_idxs[v])]; s_den += k[d]; v += 1
    buf_num[ins] = s_num; buf_den[ins] = s_den
    pred_new = s_num / s_den if s_den > eps else 0.0
    sse_in += (x_new - pred_new) ** 2
    return m_new, float(sse_in)


@njit
def buf_remove(
    x: np.ndarray, rem_idx: int, k: np.ndarray,
    buf_idxs: np.ndarray, buf_num: np.ndarray, buf_den: np.ndarray,
    m: int, sse_in: float,
) -> tuple[int, float]:
    """Remove rem_idx from buf/sse_in in-place. O(r). Does NOT touch nin/din."""
    r = k.shape[0] - 1; eps = 1e-12; x_rem = x[rem_idx]
    lo = 0; hi = m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < rem_idx: lo = mid + 1
        else: hi = mid
    pos = lo
    if pos >= m or buf_idxs[pos] != rem_idx:
        return m, sse_in

    od = buf_den[pos]; op = buf_num[pos] / od if od > eps else 0.0
    sse_in -= (x_rem - op) ** 2

    v = pos - 1
    while v >= 0:
        d = rem_idx - int(buf_idxs[v])
        if d > r: break
        w = k[d]
        od = buf_den[v]; op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] -= w * x_rem; buf_den[v] -= w
        nd = buf_den[v]; np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2
        v -= 1
    v = pos + 1
    while v < m:
        d = int(buf_idxs[v]) - rem_idx
        if d > r: break
        w = k[d]
        od = buf_den[v]; op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] -= w * x_rem; buf_den[v] -= w
        nd = buf_den[v]; np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2
        v += 1

    for t in range(pos, m - 1):
        buf_idxs[t] = buf_idxs[t+1]; buf_num[t] = buf_num[t+1]; buf_den[t] = buf_den[t+1]
    return m - 1, float(sse_in)
