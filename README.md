# regscan

Regression-based scan statistics for detecting interval anomalies in smoothly
varying 1-D signals.

Given a signal and a function family `F`, the score of an interval `I = [a, b]` is

```
S(I) = 1 - (SR_I + SR_O) / SR_A
```

where `SR_A`, `SR_I` and `SR_O` are sums of squared residuals from fitting `F`
to the whole signal, to the inside of `I`, and to the outside. `S` is near 0
when splitting explains nothing and approaches 1 when it explains everything.
The scan returns the highest-scoring interval.

## Notation

| symbol | meaning |
|---|---|
| `n` | length of the signal, in samples |
| `w` | window width — the kernel bandwidth for `F_KR`, and the scale of local structure the fit can follow |
| `r` | range cap: the longest candidate interval considered, `r = 3w` |
| `d` | polynomial degree for the `F_d` family (`poly_deg1` is `d = 1`) |
| `a`, `b` | inclusive start and end indices of a candidate interval |
| `I` | the candidate interval `[a, b]` |

`n` is fixed by the data. `w` is the one parameter worth thinking about, and
`r` follows from it. Complexities below are for a full scan over all candidate
intervals.

| method | family | model | cost |
|---|---|---|---|
| `mean` | `F_0` | constant | O(nr) |
| `poly_deg1` … `poly_deg3` | `F_d` | degree-`d` polynomial | O(nr d³) |
| `nwkr_gaussian`, `nwkr_laplace` | `F_KR` | Nadaraya-Watson kernel regression | O(nrw) |
| `krr_gaussian`, `krr_laplace` | `F_KRR` | kernel ridge regression | O(n⁴w) |

Complexities are for the algorithm; see **Performance** below for what this
implementation actually achieves and where it falls short.

`F_KR` is the method the package exists for. A weak family such as `F_0` or
`F_1` cannot represent a curved background, so it reduces residual by
splitting the interval wherever the curvature is worst — flagging smooth
structure as an anomaly. The kernel fit tracks that structure, so it enters
`SR_A`, `SR_I` and `SR_O` alike and cancels out of the score.

## Install

```bash
pip install regscan
```

numpy and numba. The kernel scan is a tight scalar loop — the shape numba
compiles well and numpy vectorises badly — so it is JIT-compiled. The first
call in a process pays a one-off compile; results are cached on disk after
that.

## Use

```python
import numpy as np, regscan

t = np.linspace(0, 1, 300)
x = 4.7 + 0.10 * t + 0.05 * np.random.default_rng(0).normal(0, 1, 300)
x[140:170] -= 0.25                      # the anomaly

res = regscan.scan(x, method="nwkr_gaussian", w=12)
res.score, res.a, res.b                 # 0.248, 140, 169
res.width_frac, res.at_edge()           # 0.10, False
```

On the same signal with no anomaly planted, `nwkr_gaussian` scores 0.036 while
`mean` scores 0.185 — the weak family is reacting to the ramp.

## Choosing `w` and `r`

`w` sets the scale of structure the fit can follow. Too small and the kernel
reproduces the anomaly itself, so it cancels out of the score; too large and
the fit cannot follow the background, which is the failure mode of the weak
families.

`r` is the longest interval the scan will consider. An anomaly wider than `r`
cannot be returned at all, and cost falls linearly as `r` does, so it is the
knob to reach for when you know roughly how wide the feature is.

```python
regscan.scan(x, method="nwkr_gaussian", w=12, r=90)
```

Omitted, `w` defaults to `max(3, n // 16)` and `r` to `3 * w`. Those are
fallbacks so the scan runs unattended, not recommendations.

Sweeping `w` and checking whether `(a, b)` stays put is a cheap way to tell a
resolved feature from an artefact of the bandwidth: a real interval holds
steady, while one that tracks `w` is measuring the kernel.

## Super-resolution

`F_KR` can block-mean the signal, scan the shorter version, then search the
original samples around the winning blocks to recover exact endpoints:

```python
from regscan import ScanConfig

regscan.scan(x, method="nwkr_gaussian", w=100,
             config=ScanConfig(super_resolution=4))     # or "auto"
```

At n = 1600 with w = 100:

| factor | time | interval |
|---|---|---|
| 1 (exact) | 3.11 s | (700, 819) |
| 2 | 0.44 s | (700, 819) |
| 4 | 0.08 s | (700, 819) |
| 8 | 0.02 s | (704, 815) |

A factor of 4 is **39× faster** and returns the same answer. A factor of 8
does not, and that is the trade: the coarse pass locates each endpoint only to
within a block of `factor` samples, and a wrong block is a wrong answer.

Refinement searches one block either side of each coarse endpoint, which
matters more than it sounds — block means smooth an anomaly's edges, so the
coarse pass picks a neighbouring block often enough that confining the search
to the winning block alone reproduced the exact interval only 20/40 times at
factor 4. Including the neighbours makes it 40/40.

It remains an approximation. Verify against `super_resolution=1` on a sample
of your data before trusting it wholesale.

`"auto"` picks the factor from the length: 1 below 450 samples, then doubling
at 900, 1800 and so on. `sr_cap` bounds it, which matters when the feature is
narrow — an interval has to survive decimation to be found.

The default is `1`, exact.

## Configuration

```python
from regscan import ScanConfig

cfg = ScanConfig(
    kernel="laplace",     # gaussian | laplace
    buffer=24,            # exclude this many samples at each end
    min_width=0.01,       # bounds on interval length, as a fraction of n
    max_width=0.25,
)
regscan.scan(x, method="nwkr_gaussian", w=16, config=cfg)
```

`ScanConfig` is immutable and passed explicitly; nothing lives in module
globals, so scanning several methods in one process cannot leak state between
them.

`buffer` matters when comparing families. An interval can only be placed in
`[buffer, n-buffer)`, so a non-zero buffer suppresses detections at the ends of
the signal — give every family the same value or the comparison is not like for
like.

`max_width` is worth capping. As an interval approaches `n/2`, inside and
outside become comparable and the statistic stops discriminating; the
maximiser then drifts to whatever split best absorbs slow curvature.

## Performance

Measured on this implementation, one full scan, `w = n // 16`:

One full scan, `w = n // 16`, after JIT warm-up:

| n | `mean` | `poly_deg1` | `nwkr_gaussian` |
|---|---|---|---|
| 100 | 3 ms | 44 ms | 4 ms |
| 200 | 12 ms | 181 ms | 18 ms |
| 400 | 51 ms | 754 ms | 83 ms |
| 800 | 187 ms | 2.7 s | 0.45 s |
| 1600 | 738 ms | 11.0 s | 3.2 s |

`F_KR` is **6× faster than `F_1`** at n = 800 despite fitting a far richer
model, which is the practical case for it: the polynomial family pays O(d³) per
candidate interval, while the kernel family pays O(r) to extend one.

Each family reaches its complexity by carrying state rather than refitting.
`mean` scores an interval from prefix sums of `y` and `y²` in O(1).
`poly_deg1` uses prefix sums of the moments `t^p` and `t^p y`, so a fit costs
O(d³) to solve regardless of interval length. `F_KR` grows an interval one
sample at a time, updating in O(r): the inside buffer and `sse_in`, the
`nin`/`din` arrays holding the inside points' kernel contribution to every
index, and `sse_out` obtained from them by subtracting from the all-points
totals. `sse_out` is adjusted rather than recomputed, so it is refreshed
exactly on a fixed cadence to keep floating-point error from accumulating.

For longer signals, super-resolution (above) cuts this substantially again.

## Scores are comparable within a family, not across families

Each family divides by its own `SR_A`, and a kernel fit has a smaller `SR_A`
than a constant fit before any interval is chosen. The same interval therefore
scores differently under `F_0` and `F_KR`. Compare families by rank, by whether
they agree on the interval, or by the contrast between anomalous and clean
signals — not by absolute score.

## Citing

Rakib et al., *Efficient Regression Models for Scan Statistics*,
[arXiv:2608.22201](https://arxiv.org/abs/2608.22201) (2026).

```bibtex
@misc{rakib2026efficientregressionmodelsscan,
      title={Efficient Regression Models for Scan Statistics},
      author={Gazi Abdur Rakib and Tristan Ashton and Ryan A. Loomis and Brian S. Mason and Eric J. Murphy and Ci Xue and Jeff M. Phillips},
      year={2026},
      eprint={2608.22201},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2608.22201},
}
```
