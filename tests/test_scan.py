import numpy as np
import pytest

import regscan
from regscan import ScanConfig


def signal(n=400, a=150, b=180, depth=4.0, slope=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n) + slope * np.arange(n)
    x[a:b] -= depth
    return x

def test_finds_planted_interval():
    x = signal()
    r = regscan.scan(x, method="mean", w=40)
    assert r.score > 0.3
    assert abs(r.a - 150) < 15 and abs(r.b - 179) < 15

def test_poly_handles_a_ramp_that_defeats_mean():
    x = signal(slope=0.05)
    m = regscan.scan(x, method="mean", w=40)
    p = regscan.scan(x, method="poly_deg1", w=40)
    assert abs(p.a - 150) < 20, "F_1 should still localise under a ramp"
    assert p.score != m.score

def test_score_is_scale_and_shift_invariant():
    x = signal()
    base = regscan.scan(x, method="mean", w=40)
    for y in (3.0 * x, x + 100.0, 3.0 * x + 100.0):
        r = regscan.scan(y, method="mean", w=40)
        assert (r.a, r.b) == (base.a, base.b)
        assert r.score == pytest.approx(base.score, rel=1e-9)

def test_flat_signal_scores_zero():
    assert regscan.scan(np.ones(200), method="mean", w=20).score == 0.0

def test_buffer_excludes_the_edges():
    x = signal(a=5, b=25, depth=8.0)
    free = regscan.scan(x, method="mean", w=40)
    buf = regscan.scan(x, method="mean", w=40,
                       config=ScanConfig(buffer=int(0.2 * x.size)))
    assert free.a < 40
    assert buf.a >= int(0.2 * x.size), "buffer must forbid edge placements"

def test_width_bounds_respected():
    x = signal()
    r = regscan.scan(x, method="mean", w=200,
                     config=ScanConfig(min_width=0.02, max_width=0.10))
    assert 0.02 <= r.width_frac <= 0.105

def test_result_helpers():
    r = regscan.scan(signal(a=2, b=20, depth=9.0), method="mean", w=30)
    assert r.at_edge() and r.width == r.b - r.a + 1
    assert set(r.as_dict()) >= {"score", "a", "b", "width_frac", "at_edge"}

def test_epidemic_score_matches_mean_family():
    x = signal()
    r = regscan.scan(x, method="mean", w=40)
    assert regscan.epidemic_score(x, r.a, r.b) == pytest.approx(r.score, rel=1e-9)

def test_deramp_removes_a_line():
    t = np.arange(200, dtype=float)
    assert abs(regscan.deramp(3 + 0.7 * t).max()) < 1e-9

def test_default_w_is_used_when_omitted():
    x = signal(n=480, a=200, b=230, depth=6.0)
    assert regscan.scan(x, method="mean").score == regscan.scan(
        x, method="mean", w=30).score


def test_errors():
    with pytest.raises(ValueError, match="1-D"):
        regscan.scan(np.zeros((4, 4)), method="mean", w=2)
    with pytest.raises(KeyError, match="unknown method"):
        regscan.scan(np.zeros(50), method="nope", w=5)
    with pytest.raises(ValueError, match="w must be"):
        regscan.scan(np.zeros(50), method="mean", w=0)
