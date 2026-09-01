"""Regression-based scan statistics for interval anomalies in 1-D signals.

Notation: ``n`` is the signal length, ``w`` the window width (kernel
bandwidth for ``F_KR``), ``r = 3w`` the longest interval considered, and ``d``
the polynomial degree for ``F_d``. A candidate interval is ``[a, b]``,
inclusive.

Reference: Rakib et al., "Efficient Regression Models for Scan Statistics",
arXiv:2608.22201 (2026). https://arxiv.org/abs/2608.22201

Quick start
-----------
>>> import numpy as np, regscan
>>> x = np.random.default_rng(0).normal(0, 1, 400)
>>> x[150:180] -= 4.0
>>> res = regscan.scan(x, method="poly_deg1")
>>> res.a, res.b, round(res.score, 2)          # doctest: +SKIP
(150, 179, 0.72)
"""

from .api import scan, scan_many
from .config import ScanConfig
from .families import deramp, epidemic_score
from .registry import available, describe
from .result import ScanResult
from .window import default_width, range_cap, sr_factor, window_params

__version__ = "0.1.0"
__all__ = [
    "ScanConfig",
    "ScanResult",
    "__version__",
    "available",
    "default_width",
    "deramp",
    "describe",
    "epidemic_score",
    "range_cap",
    "scan",
    "scan_many",
    "sr_factor",
    "window_params",
]
