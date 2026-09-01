import pytest

from regscan.window import default_width, range_cap, sr_factor, window_params


@pytest.mark.parametrize("n,expected", [
    (16, 3),        # floor applies
    (48, 3),        # n//16 == 3, floor and formula agree
    (100, 6),
    (480, 30),
    (1920, 120),
])
def test_default_width(n, expected):
    assert default_width(n) == expected


def test_floor_holds_for_short_signals():
    assert all(default_width(n) == 3 for n in range(1, 48))


def test_range_cap():
    assert range_cap(16) == 48
    with pytest.raises(ValueError):
        range_cap(0)


def test_sr_factor():
    assert sr_factor(449) == 1
    assert sr_factor(450) == 2
    assert sr_factor(899) == 2
    assert sr_factor(1920) == 8


def test_window_params_agree_with_the_pieces():
    w, r, sr = window_params(480)
    assert w == default_width(480) and r == range_cap(w) and sr == sr_factor(480)
    assert window_params(480, w=16)[0] == 16


def test_bad_lengths_rejected():
    for fn in (default_width, sr_factor):
        with pytest.raises(ValueError, match="positive"):
            fn(0)
