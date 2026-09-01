import subprocess
import sys

from regscan.registry import available, get


def test_core_methods_present():
    assert {"mean", "poly_deg1", "poly_deg2"} <= set(available())

def test_missing_extra_gives_actionable_error():
    from regscan import registry
    registry.register("fake", factory=lambda: (_ for _ in ()).throw(
        ImportError("No module named 'ruptures'")), extras="baselines")
    try:
        get("fake")
    except ImportError as e:
        assert "regscan[baselines]" in str(e)
    else:
        raise AssertionError("should have raised")

def test_import_is_light():
    """``import regscan`` must not trigger the numba compile.

    numba is a hard dependency, but importing it costs real time and it is
    only needed by the kernel family. The registry defers it until a kernel
    method is first requested, so a caller who only wants F_0 never pays.
    """
    code = ("import sys, regscan; "
            "bad=[m for m in ('torch','numba','stumpy','ruptures') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, check=False)
    assert r.returncode == 0, f"heavy modules imported at import time: {r.stdout}"
