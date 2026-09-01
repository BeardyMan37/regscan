import numpy as np
import pytest
from _oracle import direct_nwkr

import regscan
from regscan import ScanConfig
from regscan.kernel import accumulate, scan_nwkr, truncated_kernel_vector


def _configs():
    for kind in ("gaussian", "laplace"):
        for w in (3, 5, 8):
            for buf in (0, 5):
                yield kind, w, buf


@pytest.mark.parametrize("kind,w,buf", list(_configs()))
def test_incremental_matches_direct(kind, w, buf):
    """The contract for the incremental scan.

    scan_nwkr carries nin/din/buf/sse_out across thousands of O(r) updates.
    A bug there surfaces as numerical drift rather than a crash, so it is
    pinned against a scan that refits everything and carries no state.
    """
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 70) + 0.02 * np.arange(70)
    y[28:40] -= 3.0
    cfg = ScanConfig(kernel=kind, buffer=buf)
    fast = scan_nwkr(y, w, cfg=cfg)
    ref = direct_nwkr(y, w, cfg=cfg)
    assert fast[1] == ref[1]
    assert fast[0] == pytest.approx(ref[0], rel=1e-11, abs=1e-13)


def test_no_drift_over_a_long_run():
    """sse_out is updated incrementally and refreshed periodically.

    A large w means many updates between refreshes, which is where accumulated
    floating-point error would show if the refresh cadence were wrong.
    """
    rng = np.random.default_rng(9)
    y = rng.normal(0, 1, 160) + np.sin(np.arange(160) / 20)
    y[60:95] -= 2.0
    for w in (10, 20):
        cfg = ScanConfig(kernel="gaussian")
        fast = scan_nwkr(y, w, cfg=cfg)
        ref = direct_nwkr(y, w, cfg=cfg)
        assert fast[1] == ref[1]
        assert fast[0] == pytest.approx(ref[0], rel=1e-10, abs=1e-12)


def test_null_split_returns_the_empty_window():
    """Scores are floored at 0, as in the reference implementation."""
    s, ab = scan_nwkr(np.ones(80), 6)
    assert s == 0.0 and ab == (0, -1)


def test_accumulate_matches_the_state_module():
    from regscan.kernel_state import nin_din_init_full

    x = np.random.default_rng(0).normal(0, 1, 40)
    k = truncated_kernel_vector(4, 12, "gaussian")
    num, den = accumulate(x, k)
    nin = np.zeros(40)
    din = np.zeros(40)
    nin_din_init_full(x, np.arange(40, dtype=np.int64), k, nin, din)
    assert np.allclose(num, nin) and np.allclose(den, din)


def test_truncated_kernel_vector():
    k = truncated_kernel_vector(4, 12, "gaussian")
    assert k.size == 13
    assert k[0] == pytest.approx(1.0)
    assert k[12] == pytest.approx(np.exp(-9.0))
    kl = truncated_kernel_vector(4, 12, "laplace")
    assert kl[12] == pytest.approx(np.exp(-3.0))
    assert (np.diff(k) <= 0).all(), "weights must decay with distance"


def test_zero_bandwidth_is_guarded_vector():
    k = truncated_kernel_vector(0.0, 5, "gaussian")
    assert np.isfinite(k).all()





def _bandpass_like(sigma=0.02, step=0.0, n=300, seed=3):
    """A gentle ramp plus noise -- the shape real bandpass spectra have."""
    t = np.linspace(0, 1, n)
    y = 4.7 + 0.10 * t + 0.04 * np.sin(4 * t)
    y = y + sigma * np.random.default_rng(seed).normal(0, 1, n)
    if step:
        y[140:170] -= step
    return y


def test_nwkr_is_not_fooled_by_a_ramp_that_fools_the_mean():
    """The central claim: a smooth background is fitted, not flagged."""
    y = _bandpass_like()
    kr = regscan.scan(y, method="nwkr_gaussian", w=12).score
    mn = regscan.scan(y, method="mean", w=12).score
    assert kr < 0.10, f"F_KR should see almost nothing here, got {kr:.3f}"
    assert mn > 3 * kr, "F_0 is expected to be fooled by the ramp"


def test_nwkr_separates_signal_from_null_far_better_than_the_mean():
    """Absolute scores are not comparable across families; contrast is."""
    null = _bandpass_like()
    sig = _bandpass_like(step=0.25)
    kr = [regscan.scan(v, method="nwkr_gaussian", w=12).score for v in (null, sig)]
    mn = [regscan.scan(v, method="mean", w=12).score for v in (null, sig)]
    assert sig[140] < null[140]
    assert kr[1] / kr[0] > 3 * (mn[1] / mn[0]), (
        f"F_KR contrast {kr[1] / kr[0]:.1f}x should beat F_0 {mn[1] / mn[0]:.1f}x"
    )


def test_nwkr_localises_a_step_on_a_ramp_exactly():
    r = regscan.scan(_bandpass_like(step=0.25), method="nwkr_gaussian", w=12)
    assert r.score > 0.3
    assert (r.a, r.b) == (140, 169)


def test_kernel_in_name_overrides_config():
    y = np.random.default_rng(5).normal(0, 1, 80); y[30:40] -= 3
    a = regscan.scan(y, method="nwkr_laplace", w=5,
                     config=ScanConfig(kernel="gaussian")).score
    b = regscan.scan(y, method="nwkr_laplace", w=5,
                     config=ScanConfig(kernel="laplace")).score
    assert a == pytest.approx(b), "method name must win over config kernel"


def test_buffer_is_honoured_by_nwkr():
    y = np.random.default_rng(2).normal(0, 1, 200); y[3:20] -= 6
    free = regscan.scan(y, method="nwkr_gaussian", w=8)
    buf = regscan.scan(y, method="nwkr_gaussian", w=8,
                       config=ScanConfig(buffer=40))
    assert free.a < 40 <= buf.a


def test_krr_runs_and_overlaps_the_truth():
    """KRR searches exhaustively by default, so the exact interval is expected."""
    y = np.random.default_rng(4).normal(0, 0.3, 120)
    y[50:70] -= 3
    r = regscan.scan(y, method="krr_gaussian", w=10)
    overlap = max(0, min(r.b, 69) - max(r.a, 50) + 1)
    assert r.score > 0.1
    assert overlap > 0, f"({r.a},{r.b}) misses the planted interval entirely"





# --------------------------------------------------------------------------
# super-resolution
# --------------------------------------------------------------------------
def _sr_signal(n=600, a=180, width=60, seed=0):
    rng = np.random.default_rng(seed)
    y = 4.7 + 0.10 * np.linspace(0, 1, n) + 0.05 * rng.standard_normal(n)
    y[a : a + width] -= 0.4
    return y


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_super_resolution_reproduces_the_exact_interval(factor):
    """With one block of refinement slack the coarse pass loses nothing."""
    for seed in range(8):
        y = _sr_signal(seed=seed, a=100 + 30 * seed)
        exact = scan_nwkr(y, 30, cfg=ScanConfig(super_resolution=1))
        approx = scan_nwkr(y, 30, cfg=ScanConfig(super_resolution=factor))
        assert approx[1] == exact[1], f"seed={seed} factor={factor}"
        assert approx[0] == pytest.approx(exact[0], rel=1e-9)


def test_refinement_searches_beyond_the_winning_block():
    """Refinement must reach into the neighbouring blocks, not just its own.

    Confining it to the winning block loses the exact interval about half the
    time, and silently -- the result is a plausible interval a few samples
    out, not an error -- so this pins the behaviour directly rather than
    trusting a knob.
    """
    from regscan.kernel import _fit_stats, _refine, _spans

    factor = 4
    y = _sr_signal(a=99, width=60)          # start deliberately off-block
    k = truncated_kernel_vector(30, 90, "gaussian")
    numer, denom, sra = _fit_stats(y, k)
    span_min, span_max = _spans(y.size, 90, ScanConfig())

    # The coarse pass would report block 25 (samples 100..103); the true start
    # at 99 lives in block 24 and is only reachable if neighbours are searched.
    _, (a, _b) = _refine(y, numer, denom, sra, k, 25, 39, factor,
                         span_min, span_max)
    assert a < 100, f"refinement never looked before sample 100 (got a={a})"


def test_auto_factor_follows_length():
    from regscan.kernel import resolve_factor

    cfg = ScanConfig(super_resolution="auto")
    assert resolve_factor(300, cfg) == 1
    assert resolve_factor(600, cfg) == 2
    assert resolve_factor(1000, cfg) == 4
    assert resolve_factor(1000, ScanConfig(super_resolution="auto", sr_cap=2)) == 2


def test_exact_is_the_default():
    """A scan is exact unless super-resolution is asked for explicitly."""
    from regscan.kernel import resolve_factor

    assert resolve_factor(300, ScanConfig()) == 1
    assert resolve_factor(10_000, ScanConfig()) == 1


def test_auto_matches_an_exact_scan():
    """"auto" is approximate, so it is pinned against the exact scan."""
    for seed in range(6):
        y = _sr_signal(n=1000, a=150 + 90 * seed, width=70, seed=seed)
        auto = scan_nwkr(y, 40, cfg=ScanConfig(super_resolution="auto"))
        exact = scan_nwkr(y, 40, cfg=ScanConfig(super_resolution=1))
        assert auto[1] == exact[1], f"seed={seed}"
        assert auto[0] == pytest.approx(exact[0], rel=1e-9)


def test_r_bounds_the_interval():
    y = _sr_signal(a=200, width=120)
    wide = scan_nwkr(y, 20, r=200)
    narrow = scan_nwkr(y, 20, r=40)
    assert wide[1][1] - wide[1][0] + 1 > 40
    assert narrow[1][1] - narrow[1][0] + 1 <= 41


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="super_resolution"):
        ScanConfig(super_resolution=0)
    with pytest.raises(ValueError, match="truncation"):
        ScanConfig(truncation=0)
