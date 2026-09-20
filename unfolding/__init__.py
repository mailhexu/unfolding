from .lcao_unfolder import HamiltonIOModel, LCAOUnfolder, LCAOWeights
from .mapping import RelabelMap, RelabelMapError
from .phonon_unfolder import phonon_unfolder
from .unfolder import Unfolder

__all__ = [
    "HamiltonIOModel",
    "LCAOUnfolder",
    "LCAOWeights",
    "RelabelMap",
    "RelabelMapError",
    "phonon_unfolder",
    "Unfolder",
    "phonopy_unfold",
]

# Symbols whose adapter modules need optional backends: name -> (module, extra).
# Note: DDB_unfolder/nc_unfolder are intentionally NOT re-exported here -- the
# function names would collide with the submodule names and make package
# attributes order-dependent. Use `from unfolding.DDB_unfolder import DDB_unfolder`.
_LAZY_EXPORTS = {
    "phonopy_unfold": ("unfolding.phonopy_unfolder", "phonopy"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, extra = _LAZY_EXPORTS[name]
        try:
            module = __import__(module_name, fromlist=[name])
        except ModuleNotFoundError as exc:
            if exc.name != extra:
                raise  # a different dependency is missing: surface the real error
            msg = (
                f"{extra} is required for {name}. "
                f"Install it with: pip install unfolding[{extra}]"
            )
            raise ImportError(msg) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
