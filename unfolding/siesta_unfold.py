"""One-call SIESTA unfolding adapter (story 008).

Staged API::

    SislParser(fdf) -> HamiltonIOModel -> RelabelMap -> LCAOUnfolder -> plot

``unfold_siesta`` glues the stages together: parse a SIESTA calculation
through HamiltonIO, build the supercell-to-normal-cell relabel map
against the primitive cell, compute the spectral weights on a k-path,
and draw weight-coded bands with the same ``plot_band_weight``
ergonomics as the phonon adapter.

The HamiltonIO import happens lazily inside :func:`unfold_siesta`; when
only the weights are wanted, pass an already-parsed HamiltonIO-compatible
``model`` object (or a HamiltonIOModel) and no HamiltonIO import is
attempted. Collinear spin-polarized calculations expose ``up``/``down``
channels on the parsed model; select one with ``spin``.
"""
from __future__ import annotations

import os

import numpy as np


def _select_channel(model, spin):
    """Return the spin channel of a parsed HamiltonIO model.

    ``SislParser.get_model()`` returns either a single model
    (unpolarized/non-collinear), an object with ``up``/``down``
    attributes, or a ``(model_up, model_down)`` tuple for a collinear
    calculation. Non-collinear/unpolarized models are returned
    unchanged.
    """
    if isinstance(model, (tuple, list)):
        idx = {"up": 0, "down": 1}[spin]
        if len(model) <= idx:
            raise ValueError(
                f"parsed model has {len(model)} spin channel(s); "
                f"cannot select spin={spin!r}"
            )
        return model[idx]
    if spin in ("up", "down") and hasattr(model, spin):
        return getattr(model, spin)
    if spin in ("up", "down", "both"):
        return model
    raise ValueError(f"unknown spin selection: {spin!r}")


def _load_channel_model(fdf, spin):
    """Parse a SIESTA fdf through HamiltonIO's SislParser.

    Raises FileNotFoundError when the fdf is missing and an actionable
    ImportError when HamiltonIO is not installed.
    """
    if not os.path.isfile(fdf):
        raise FileNotFoundError(f"SIESTA fdf not found: {fdf!r}")
    try:
        from HamiltonIO.siesta import SislParser
    except (ImportError, ModuleNotFoundError) as exc:
        name = getattr(exc, "name", None) or ""
        if "HamiltonIO" not in name and "sisl" not in name and name != "":
            raise
        raise ImportError(
            "HamiltonIO is required to parse SIESTA calculations. "
            "Install it with: pip install HamiltonIO sisl"
        ) from exc
    model = SislParser(fdf).get_model()
    return _select_channel(model, spin)


def _as_axis_arrays(res, xqpts):
    """Reshape LCAOWeights into the per-band lists plot_band_weight takes."""
    x = np.asarray(xqpts, dtype=float)
    nbands = res.weights.shape[1]
    kslist = [list(x) for _ in range(nbands)]
    ekslist = [list(res.eigenvalues[:, ib]) for ib in range(nbands)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nbands)]
    return kslist, ekslist, wkslist


def _derive_prim_counts(sc_atoms, orb_dict_sc, prim_atoms):
    """Default primitive orbital counts: species-level counts from the
    parsed model's ``orb_dict``. Returns None when unambiguous."""
    if orb_dict_sc is None:
        return None
    counts = _as_counts(orb_dict_sc, len(sc_atoms))
    sym_sc = sc_atoms.get_chemical_symbols()
    species_counts = {}
    for s, c in zip(sym_sc, counts):
        if species_counts.setdefault(s, c) != c:
            return None  # species carries mixed counts: user must specify
    sym_prim = prim_atoms.get_chemical_symbols()
    missing = sorted({s for s in sym_prim if s not in species_counts})
    if missing:
        raise ValueError(
            f"no orbital counts for species {missing} in the parsed model; "
            "pass orb_counts_prim explicitly"
        )
    return [species_counts[s] for s in sym_prim]


def _as_counts(orb_dict, n):
    """Normalize a HamiltonIO ``{index: [names]}`` dict to per-atom ints."""
    out = [len(orb_dict[i]) for i in range(n)]
    if any(c == 0 for c in out):
        raise ValueError("every atom needs at least one orbital in orb_dict")
    return out


def unfold_siesta(
    fdf=None,
    model=None,
    prim_atoms=None,
    unfold_sc_mat=None,
    spin="up",
    kpts=None,
    knames=None,
    xqpts=None,
    Xqpts=None,
    tol_r=0.04,
    orb_counts_prim=None,
    efermi=0.0,
    axis=None,
    output=None,
    style="alpha",
    color="blue",
    yrange=None,
    ylabel="Energy (eV)",
    ypad=1.5,
    width=2,
    title=None,
):
    """Unfold a SIESTA supercell calculation onto a primitive k-path.

    Parameters
    ----------
    fdf : str, optional
        SIESTA fdf file; parsed through HamiltonIO's SislParser. Either
        ``fdf`` or a pre-parsed ``model`` must be given.
    model : object, optional
        Already-parsed HamiltonIO-compatible model (or
        :class:`~unfolding.lcao_unfolder.HamiltonIOModel` channel);
        skips the HamiltonIO import entirely.
    prim_atoms : ase.Atoms
        Primitive cell used for the relabel map.
    unfold_sc_mat : (3, 3) int array
        Supercell matrix in primitive-lattice units (row convention).
    spin : {"up", "down"}
        Collinear channel selection (ignored for unpolarized models).
    kpts, knames, xqpts, Xqpts
        k-path exactly as in :func:`unfolding.phonopy_unfolder.phonopy_unfold`.
    tol_r : float
        Cartesian atom-matching tolerance (Angstrom) for the relabel map.
    orb_counts_prim : sequence of int or dict, optional
        Per-atom orbital counts of the primitive cell (HamiltonIO
        ``orb_dict`` form). Supercell counts come from the parsed
        model's ``orb_dict``.
    efermi : float
        Fermi level (passed to the plotter).
    axis : matplotlib Axes, optional
        Draw into an existing axes instead of creating one.
    output : str, optional
        Save the figure to this path.
    ypad : float
        Energy margin added below/above the band range (eV).

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the weight-coded unfolded bands.
    """
    if fdf is None and model is None:
        raise ValueError("provide either fdf or a parsed model")
    if model is None:
        model = _load_channel_model(fdf, spin)
    else:
        model = _select_channel(model, spin)
    if prim_atoms is None:
        raise ValueError("prim_atoms (the primitive cell) is required")
    if unfold_sc_mat is None:
        raise ValueError("unfold_sc_mat is required")
    if kpts is None:
        raise ValueError("kpts (the primitive-cell k-path) is required")

    from ase.io import read as _ase_read

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    if isinstance(prim_atoms, (str, os.PathLike)):
        prim_atoms = _ase_read(prim_atoms)

    if isinstance(model, HamiltonIOModel):
        adapted = model
        raw_orb_dict = getattr(adapted._model, "orb_dict", None)
    else:
        adapted = HamiltonIOModel(model)
        raw_orb_dict = getattr(model, "orb_dict", None)

    if orb_counts_prim is None:
        orb_counts_prim = _derive_prim_counts(
            adapted.atoms, raw_orb_dict, prim_atoms
        )

    rm = RelabelMap.from_atoms(
        adapted.atoms, prim_atoms, unfold_sc_mat, tol_r=tol_r,
        orb_counts_sc=raw_orb_dict, orb_counts_prim=orb_counts_prim,
    )
    unf = LCAOUnfolder(adapted, rm)
    res = unf.compute(kpts)

    from unfolding.plotphon import plot_band_weight

    if xqpts is None:
        xqpts = np.arange(len(kpts))
    kslist, ekslist, wkslist = _as_axis_arrays(res, xqpts)
    ax = plot_band_weight(
        kslist,
        ekslist,
        wkslist,
        efermi=efermi,
        yrange=yrange,
        output=output,
        style=style,
        color=color,
        axis=axis,
        width=width,
        xticks=[knames, Xqpts] if knames is not None and Xqpts is not None else None,
        title=title,
        ylabel=ylabel,
        ypad=ypad,
    )
    if output is not None:
        ax.figure.savefig(output)
    return ax
