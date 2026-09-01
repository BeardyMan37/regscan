"""Scan result."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ScanResult:
    """One scan outcome.

    Attributes
    ----------
    score: statistic in [0, 1]; higher means the interval explains more.
    a, b: inclusive channel indices of the winning interval.
    method: the method name that produced it.
    n: signal length, so ``width`` is meaningful without the signal.
    """

    score: float
    a: int
    b: int
    method: str
    n: int

    @property
    def width(self) -> int:
        return self.b - self.a + 1

    @property
    def width_frac(self) -> float:
        return self.width / self.n if self.n else 0.0

    def at_edge(self, frac: float = 0.05) -> bool:
        """True when either endpoint falls in the outer *frac* of the signal."""
        return self.a < self.n * frac or self.b > self.n * (1 - frac)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(width=self.width, width_frac=self.width_frac,
                 at_edge=self.at_edge())
        return d
