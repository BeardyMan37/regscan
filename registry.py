"""Method registry with lazy imports.

Every method is registered by name with a *factory*, not an imported
function. Nothing heavy is imported until a method is actually requested.

This matters: the research module called ``_warmup_stumpy()`` at import time,
which JIT-compiled a numba kernel on every ``import``. In a script that costs
you once; in a library it costs every process that imports the package, and
under a spawn-based process pool it costs every worker even when STUMPY is
not among the selected methods.
"""

from __future__ import annotations

from typing import Callable, Dict

_REGISTRY: Dict[str, dict] = {}


def register(name: str, *, factory: Callable[[], Callable], extras: str | None = None,
             family: str = "", doc: str = "") -> None:
    """Register *name*; *factory* is called once, on first use."""
    _REGISTRY[name] = dict(factory=factory, extras=extras, family=family,
                           doc=doc, _fn=None)


def available() -> list[str]:
    """All registered names, whether or not their dependencies are installed."""
    return sorted(_REGISTRY)


def describe() -> Dict[str, dict]:
    return {k: {i: j for i, j in v.items() if not i.startswith("_")}
            for k, v in _REGISTRY.items()}


def get(name: str) -> Callable:
    """Resolve a method name to a callable, importing its deps on first use."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {available()}")
    entry = _REGISTRY[name]
    if entry["_fn"] is None:
        try:
            entry["_fn"] = entry["factory"]()
        except ImportError as exc:
            extra = entry["extras"]
            hint = (f"  pip install 'regression-scan-stats[{extra}]'"
                    if extra else "")
            raise ImportError(
                f"method {name!r} needs an optional dependency: {exc}\n{hint}"
            ) from exc
    return entry["_fn"]


def _poly(degree: int):
    from functools import partial

    from . import families

    return partial(families.scan_poly, degree=degree)


def _kernel_scan(name: str, kind: str):
    """NWKR / KRR bound to a kernel, overriding whatever the config carries.

    A method named ``nwkr_laplace`` must use a Laplace kernel even if the
    caller passes ``ScanConfig(kernel="gaussian")``, so the name wins.
    """
    from functools import partial

    from . import kernel as K

    fn = K.scan_nwkr if name == "nwkr" else K.scan_krr

    def run(x, w, cfg=None):
        from .config import ScanConfig

        cfg = (cfg or ScanConfig()).evolve(kernel=kind)
        return fn(x, w, cfg=cfg)

    run.__name__ = f"{name}_{kind}"
    return run


def _core():
    from . import families

    register("mean", factory=lambda: families.scan_constant, family="F_0",
             doc="constant family; the classic Gaussian scan statistic")
    register("poly_deg1", factory=lambda: _poly(1), family="F_1",
             doc="degree-1 polynomial family")
    register("poly_deg2", factory=lambda: _poly(2), family="F_2",
             doc="degree-2 polynomial family")
    register("poly_deg3", factory=lambda: _poly(3), family="F_3",
             doc="degree-3 polynomial family")
    register("nwkr_gaussian", factory=lambda: _kernel_scan("nwkr", "gaussian"),
             family="F_KR", doc="Nadaraya-Watson, Gaussian kernel (proposed)")
    register("nwkr_laplace", factory=lambda: _kernel_scan("nwkr", "laplace"),
             family="F_KR", doc="Nadaraya-Watson, Laplace kernel")
    register("krr_gaussian", factory=lambda: _kernel_scan("krr", "gaussian"),
             family="F_KRR", doc="kernel ridge regression, Gaussian kernel; slow")
    register("krr_laplace", factory=lambda: _kernel_scan("krr", "laplace"),
             family="F_KRR", doc="kernel ridge regression, Laplace kernel; slow")


_core()
