"""Scan configuration.

Kernel choice and scan bounds are values, not module-level globals. A global
mutated by a setter is fine in a script and wrong in a library: two callers in
one process, or two methods scanned in sequence inside one worker, silently
share state and contaminate each other's results.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

KernelKind = Literal["gaussian", "laplace"]

#: Kernels are truncated at ``TRUNCATION * r`` where r is the bandwidth, so a
#: Gaussian is evaluated over roughly +-3 sigma.
TRUNCATION = 3


@dataclass(frozen=True, slots=True)
class ScanConfig:
    """Immutable parameters for one scan.

    Parameters
    ----------
    kernel:
        ``"gaussian"`` uses ``exp(-(i-j)^2 / w^2)``, ``"laplace"`` uses
        ``exp(-|i-j| / w)``. Only used by the kernel family.
    buffer:
        Channels excluded at each end of the signal. Candidate intervals are
        never placed there. ``0`` means the whole signal is searchable.

        Note this *excises* the samples rather than merely forbidding an
        interval from starting there: the fit, ``SR_A`` and the outside
        residuals are all computed on the trimmed signal.

        A non-zero buffer suppresses detections at the ends of the signal, so
        when comparing families give them all the same value -- a buffer on
        one and not another is not a like-for-like comparison.
    min_width, max_width:
        Bounds on the candidate interval length, as a fraction of the signal.
        Intervals approaching half the signal stop discriminating (inside and
        outside become comparable), so capping is usually wise.
    truncation:
        Kernel support in units of bandwidth: weights beyond ``truncation * w``
        are treated as zero. At 3 a Gaussian has decayed to e^-9 there, so the
        discarded tail is far below numerical noise.
    super_resolution:
        ``1`` (the default) scans every sample. A larger integer block-means
        the signal by that factor, scans the shorter version, then searches
        the original samples around the winning blocks to recover the
        endpoints. ``"auto"`` picks the factor from the signal length via
        :func:`regscan.sr_factor`.

        This trades exactness for speed, and the saving is large: a factor of
        4 ran roughly 39x faster on a 1600-sample signal and returned the same
        interval. But the coarse pass locates each endpoint only to within a
        block, and a wrong block is a wrong answer, so the default is exact.
        Compare the two on your own data before enabling it.
    sr_base:
        Signal length below which ``"auto"`` chooses a factor of 1.
    sr_cap:
        Upper bound on the automatically chosen factor.
    """

    kernel: KernelKind = "gaussian"
    buffer: int = 0
    min_width: float = 0.0
    max_width: float = 1.0
    truncation: int = TRUNCATION
    super_resolution: int | str = 1
    sr_base: int = 450
    sr_cap: int | None = None

    def __post_init__(self) -> None:
        if self.kernel not in ("gaussian", "laplace"):
            raise ValueError(f"kernel must be gaussian or laplace, got {self.kernel!r}")
        if self.buffer < 0:
            raise ValueError("buffer must be >= 0")
        if not 0.0 <= self.min_width <= self.max_width <= 1.0:
            raise ValueError("need 0 <= min_width <= max_width <= 1")
        if self.truncation < 1:
            raise ValueError("truncation must be >= 1")
        if self.super_resolution != "auto" and (
            not isinstance(self.super_resolution, int) or self.super_resolution < 1
        ):
            raise ValueError('super_resolution must be a positive int or "auto"')

    def evolve(self, **kw) -> ScanConfig:
        """Return a copy with some fields changed."""
        return replace(self, **kw)
