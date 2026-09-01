# Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## Adding a family

Register it in `regscan/registry.py` with a *factory*, not an import:

```python
register("my_method", factory=lambda: _my_scan, family="F_X", doc="...")
```

Nothing is imported until the method is first requested, which keeps
`import regscan` cheap. `tests/test_registry.py::test_import_is_light` asserts
that importing the package pulls in no heavy modules; keep it passing.

Every scan function has the signature:

```python
fn(x: np.ndarray, w: int, cfg: ScanConfig | None = None) -> tuple[float, tuple[int, int]]
```

returning `(score, (a, b))` with `b` inclusive.

## Testing a fast implementation

`scan_nwkr` only recomputes indices within `T = truncation * w` of an interval
boundary, on the argument that every other index has the same fit restricted
or not. `tests/test_kernel.py::test_matches_naive_implementation` checks that
against a deliberately naive scan that refits everything, for both kernels and
several bandwidths, to machine precision. Any faster implementation — such as
the paper's incremental O(r) window updates — must pass the same test.

## Optimisations still open

`mean`, `poly_deg*` and `nwkr_*` are all at their intended complexity:

- constant family — prefix sums of `y` and `y²`, O(1) per interval;
- polynomial family — prefix sums of the moments `t^p` and `t^p y`, O(d³) per
  interval regardless of its length;
- kernel family — incremental `nin`/`din`, inside buffer and `sse_out`, O(r)
  to extend an interval by one sample.

Super-resolution sits on top: `decimate` block-means the signal, and the
coarse result is refined by scoring the original samples around the winning
blocks. The refinement window deliberately spans one block either side of each
coarse endpoint — narrowing it to the winning block halves the rate of exact
agreement, and does so silently, returning a plausible interval a few samples
out rather than an error. `test_refinement_searches_beyond_the_winning_block`
pins that directly.

## The kernel scan's correctness contract

`scan_nwkr` carries four pieces of mutable state across thousands of O(r)
updates. A mistake there does not crash — it drifts. So it is pinned against
`tests/_oracle.py::direct_nwkr`, which refits every index for every interval
and carries no state at all:

- `test_incremental_matches_reference` covers both kernels, three bandwidths
  and two buffer settings, to 1e-11 relative;
- `test_no_drift_over_a_long_run` uses large `w`, which puts many incremental
  updates between exact refreshes — where accumulated error would appear.

Any change to `kernel_state.py` or to the search loop must keep both passing.
Note that `sse_out` is refreshed exactly every `REFRESH = range_cap` steps; if
you change that cadence, the drift test is what tells you whether you can.

## Scope

This package holds the regression scan statistic families only: `F_0`, `F_d`,
`F_KR`, `F_KRR`. Baseline detectors, benchmarking harnesses and
application-specific preprocessing belong elsewhere, so that a scan statistic
never drags ruptures, stumpy or torch into someone's environment.
