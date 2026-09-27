"""One-call OpenMX supercell unfolding onto a primitive k-path.

Thin adapter over :class:`~unfolding.lcao_unfolder.LCAOUnfolder` for
models parsed from OpenMX ``.scfout`` files by HamiltonIO's
``OpenmxParser``. Mirrors :func:`unfolding.siesta_unfold.unfold_siesta`.
The HamiltonIO import happens lazily here as well; passing a
pre-parsed ``model`` skips it entirely.
"""

from __future__ import annotations

import os

import numpy as np


def _orb_counts_from_model(model, n_atoms):
    """Per-atom orbital counts of an OpenmxWrapper model."""
    data = model._model.data
    counts = [int(c) for c in data.total_numorbs]
    if len(counts) != n_atoms:
        raise ValueError("orbital counts do not match the atom count")
    return counts


def unfold_openmx(
    scfout=None,
    model=None,
    prim_atoms=None,
    unfold_sc_mat=None,
    spin=None,
    kpts=None,
    knames=None,
    xqpts=None,
    Xqpts=None,
    tol_r=0.04,
    orb_counts_prim=None,
    match_species=True,
    method="ideal",
    atol_orth=1e-6,
    atol_imag=1e-5,
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
    """Unfold an OpenMX supercell calculation onto a primitive k-path.

    Parameters
    ----------
    scfout : str, optional
        OpenMX ``.scfout`` file; parsed through HamiltonIO's
        ``OpenmxParser``. Either ``scfout`` or a pre-parsed ``model``
        must be given.
    model : object, optional
        Already-parsed HamiltonIO-compatible OpenMX model.
    prim_atoms : ase.Atoms
        Primitive cell used for the relabel map.
    unfold_sc_mat : (3, 3) int array
        Supercell matrix in primitive-lattice units (row convention).
    spin : None or {0, 1}
        Channel selection for collinear runs; leave ``None`` for
        unpolarized data.
    kpts, knames, xqpts, Xqpts
        k-path exactly as in :func:`unfold_siesta`.
    tol_r : float
        Cartesian atom-matching tolerance (Angstrom).
    orb_counts_prim : sequence of int, optional
        Per-atom primitive-cell orbital counts. Defaults to the
        supercell counts (equal-basis species), as for
        Si7.0-s2p2d1/P7.0-s2p2d1.
    match_species : bool
        Pass ``False`` for substitutional defects (the dopant atom maps
        onto the host site it replaces).
    method : {"ideal", "ring"}
        Weight definition (see ``LCAOUnfolder.compute``).
    efermi : float
        Fermi level (eV); the plotted eigenvalues are shifted by
        ``-efermi`` so that E_F is at zero. Defaults to the OpenMX
        ``ChemP`` when a ``scfout`` is parsed here.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the weight-coded unfolded bands.
    """
    if scfout is None and model is None:
        raise ValueError("provide either scfout or a parsed model")
    if prim_atoms is None:
        raise ValueError("prim_atoms (the primitive cell) is required")
    if unfold_sc_mat is None:
        raise ValueError("unfold_sc_mat is required")
    if kpts is None:
        raise ValueError("kpts (the primitive-cell k-path) is required")

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    if model is None:
        from HamiltonIO.openmx import OpenmxParser

        parser = OpenmxParser(scfout)
        model = parser.get_model(spin=spin)
        if efermi == 0.0 and parser.efermi != 0.0:
            efermi = parser.efermi

    if isinstance(model, HamiltonIOModel):
        adapted = model
    else:
        adapted = HamiltonIOModel(model)

    if orb_counts_prim is None:
        # species-level counts from the supercell, mapped onto the
        # primitive symbols (equal-basis host/dopant species share counts)
        sym_sc = adapted.atoms.get_chemical_symbols()
        sym_prim = prim_atoms.get_chemical_symbols()
        counts_sc = _orb_counts_from_model(adapted, len(sym_sc))
        by_symbol = dict(zip(sym_sc, counts_sc))
        orb_counts_prim = [by_symbol.get(s, counts_sc[0]) for s in sym_prim]

    rm = RelabelMap.from_atoms(
        adapted.atoms,
        prim_atoms,
        unfold_sc_mat,
        tol_r=tol_r,
        orb_counts_sc=_orb_counts_from_model(adapted, len(adapted.atoms)),
        orb_counts_prim=orb_counts_prim,
        match_species=match_species,
    )
    unf = LCAOUnfolder(adapted, rm)
    res = unf.compute(
        kpts, method=method, atol_orth=atol_orth, atol_imag=atol_imag
    )
    # Reference energies to the Fermi level (house style: E_F = 0).
    res.eigenvalues = res.eigenvalues - efermi

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
        xticks=(
            [knames, Xqpts]
            if knames is not None and Xqpts is not None
            else None
        ),
        title=title,
        ylabel=ylabel,
        ypad=ypad,
    )
    if output is not None:
        ax.figure.savefig(output)
    return ax


def _as_axis_arrays(res, xqpts):
    nb = res.weights.shape[1]
    kslist = [list(xqpts) for _ in range(nb)]
    ekslist = [list(res.eigenvalues[:, ib]) for ib in range(nb)]
    wkslist = [
        list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)
    ]
    return kslist, ekslist, wkslist
