"""Public entry points."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .config import ScanConfig
from .registry import get
from .result import ScanResult
from .window import default_width


def scan(x, *, method: str = "nwkr_gaussian", w: int | None = None,
         r: int | None = None, config: ScanConfig | None = None) -> ScanResult:
    """Scan one signal for the interval that best explains it.

    Parameters
    ----------
    x: the signal.
    method: a name from :func:`regscan.available`.
    w: window width -- the kernel bandwidth for ``F_KR``, and the scale of
       structure any family's fit can follow. Defaults to ``max(3, n // 16)``.
       Pass it explicitly when you know the scale of the feature you are
       after; the default is a fallback, not a recommendation.
    r: the longest interval the scan will consider, in samples. Defaults to
       ``3 * w``. An anomaly wider than *r* cannot be returned, so raise it
       when looking for broad features; lower it to cut cost, which falls
       linearly in *r*.
    config: a :class:`ScanConfig`; defaults are used when omitted.
    """
    cfg = config or ScanConfig()
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if not np.isfinite(x).any():
        raise ValueError("x is entirely non-finite")
    if w is None:
        w = default_width(x.size)
    elif int(w) < 1:
        raise ValueError(f"w must be >= 1, got {w}")
    if r is not None and int(r) < 1:
        raise ValueError(f"r must be >= 1, got {r}")
    fn = get(method)
    score, (a, b) = fn(x, w, r, cfg=cfg)
    return ScanResult(float(score), int(a), int(b), method, int(x.size))


def scan_many(signals: Iterable[Sequence[float]], *,
              method: str = "nwkr_gaussian", **kw) -> list[ScanResult]:
    """Scan each signal in turn. Signals may differ in length."""
    return [scan(s, method=method, **kw) for s in signals]
