"""Default window width.

The window width ``w`` sets the scale of the structure the scan can represent:
the kernel bandwidth for ``F_KR``, and the range cap ``r = 3w`` bounding how
long a candidate interval may be.

Pass ``w`` explicitly whenever you know the scale of the structure you are
looking for. The default exists so the scan runs without one, not because it
is the right answer for any particular signal.
"""

from __future__ import annotations

import math

import numpy as np

#: The default window is this fraction of the signal length.
DEFAULT_W_FRACTION = 1 / 16

#: A window narrower than this cannot support a meaningful local fit.
MIN_W = 3

#: Signal length below which no decimation is applied.
SR_BASE = 450


def default_width(n: int) -> int:
    """``max(3, n // 16)`` -- the width used when the caller gives none."""
    if n <= 0:
        raise ValueError("n must be positive")
    return max(MIN_W, int(n * DEFAULT_W_FRACTION))


def range_cap(w: int) -> int:
    """Longest candidate interval considered, ``r = 3w``."""
    w = int(w)
    if w < 1:
        raise ValueError("w must be >= 1")
    return 3 * w


def sr_factor(n: int, base: int = SR_BASE, ratio: int = 2, step: int = 2,
              cap: int | None = None) -> int:
    """Decimation factor for a signal of length *n*.

    The factor is ``step ** ceil(log_ratio(n / base))``, so it is 1 while the
    signal fits in *base* samples and then grows geometrically: with the
    defaults, 1 up to 449 samples, 2 to 899, 4 to 1799, and so on. Scanning
    cost falls roughly with the square of the factor, since both the number of
    starts and the number of lengths shrink.

    *cap* bounds the factor, which is worth setting when the feature you are
    looking for is narrow: an interval must survive decimation to be found at
    all, so a factor approaching the feature width will lose it.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if base < 1 or ratio < 2 or step < 2:
        raise ValueError("base >= 1, ratio >= 2 and step >= 2 required")
    s = math.ceil((n + 1) / base)
    k = math.ceil(math.log(s, ratio)) if s > 1 else 0
    f = step ** k
    return min(f, cap) if cap is not None else f


def decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean *x* by *factor*, dropping any incomplete trailing block."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    n_blk = x.size // factor
    if n_blk == 0:
        return np.empty(0, dtype=float)
    return x[: n_blk * factor].reshape(n_blk, factor).mean(axis=1)


def window_params(n: int, w: int | None = None, r: int | None = None
                  ) -> tuple[int, int, int]:
    """Return ``(w, r, sr)`` for a signal of length *n*."""
    w = default_width(n) if w is None else int(w)
    r = range_cap(w) if r is None else int(r)
    return w, r, sr_factor(n)
