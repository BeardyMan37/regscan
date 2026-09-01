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

#: The default window is this fraction of the signal length.
DEFAULT_W_FRACTION = 1 / 16

#: A window narrower than this cannot support a meaningful local fit.
MIN_W = 3

#: Signal length per super-resolution level.
SR_BLOCK = 450


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


def sr_factor(n: int) -> int:
    """Super-resolution factor: the smallest power of two >= ``n / 450``.

    Long signals are decimated by this factor before scanning and the winning
    window refined exactly afterwards.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return max(1, 2 ** math.ceil(math.log2(max(1, math.ceil((n + 1) / SR_BLOCK)))))


def window_params(n: int, w: int | None = None) -> tuple[int, int, int]:
    """Return ``(w, r, sr)`` for a signal of length *n*."""
    w = default_width(n) if w is None else int(w)
    return w, range_cap(w), sr_factor(n)
