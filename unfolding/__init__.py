from .lcao_unfolder import HamiltonIOModel, LCAOUnfolder, LCAOWeights
from .wfsx_unfolder import WFSXUnfolder, WFSXWeights
from .mapping import RelabelMap, RelabelMapError
from .phonon_unfolder import phonon_unfolder
from .unfolder import Unfolder

__all__ = [
    "HamiltonIOModel",
    "LCAOUnfolder",
    "LCAOWeights",
    "WFSXUnfolder",
    "WFSXWeights",
    "RelabelMap",
    "RelabelMapError",
    "phonon_unfolder",
    "Unfolder",
    "phonopy_unfold",
    "unfold_siesta",
]

# Symbols whose adapter modules need optional backends: name -> (module, extra).
# Note: DDB_unfolder/nc_unfolder are intentionally NOT re-exported here -- the
# function names would collide with the submodule names and make package
# attributes order-dependent. Use `from unfolding.DDB_unfolder import DDB_unfolder`.
_LAZY_EXPORTS = {
    "phonopy_unfold": ("unfolding.phonopy_unfolder", "phonopy"),
    # unfolding.siesta_unfold imports only numpy at module level; the
    # HamiltonIO dependency is checked lazily inside unfold_siesta when
    # an fdf is parsed (a pre-parsed `model=` needs no HamiltonIO).
    "unfold_siesta": ("unfolding.siesta_unfold", None),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, extra = _LAZY_EXPORTS[name]
        try:
            module = __import__(module_name, fromlist=[name])
        except ModuleNotFoundError as exc:
            if extra is None or exc.name != extra:
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
