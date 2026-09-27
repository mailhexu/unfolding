"""ABINIT planewave unfolding adapter; WFK parsing lives in HamiltonIO."""
from __future__ import annotations

from HamiltonIO.abinit import HARTREE_TO_EV, WFKData, read_wfk

from .pw_unfolder import PWEigenData, PWUnfolder, PWWeights


def unfold_abinit(
    wfk=None,
    unfold_sc_mat=None,
    kpts=None,
    *,
    data: WFKData | None = None,
    knames=None,
    xqpts=None,
    Xqpts=None,
    spin: int = 0,
    average_degenerate: float | None = None,
    resolve_degenerate: float | None = None,
    fermi_shift: bool = True,
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
    """Unfold an ABINIT WFK on a primitive-cell k-path and return Axes.

    Parameters
    ----------
    wfk : path-like or WFKData, optional
        Supercell WFK path, or a pre-parsed record. Supply exactly one of
        wfk and data.
    unfold_sc_mat : (3, 3) integer array
        Row-convention supercell matrix satisfying A_sc = M @ A_prim.
    kpts : (nk, 3) array-like
        Primitive-cell fractional k-path to unfold.
    data : WFKData, optional
        Pre-parsed WFK record; avoids filesystem and netCDF I/O.
    knames, xqpts, Xqpts : sequence, optional
        Plot labels, path coordinates, and tick positions.
    spin : int
        Collinear WFK spin-channel index.
    average_degenerate : float, optional
        Band-energy grouping tolerance in eV.
    resolve_degenerate : float, optional
        Band-energy tolerance in eV for eigen-assigning weights inside
        near-degenerate groups. Degenerate bands may be stored as arbitrary
        unitary mixtures of their fold sectors, which splits the per-band
        weights randomly between k-points and renders as dotted or broken
        weight-coded lines; resolving restores gauge-invariant branch
        weights (0/1 for pristine sectors) and continuous lines.
    fermi_shift : bool
        Subtract the WFK Fermi header and draw zero as the Fermi level.
        Requires that the WFK carries a fermi_energy header.
    axis : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    output : path-like, optional
        Figure filename to save after drawing.
    style, color, width, yrange, ylabel, ypad, title
        Plotting options forwarded to PWWeights.plot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing weight-coded unfolded bands.
    """
    if isinstance(wfk, WFKData):
        if data is not None:
            raise ValueError("provide WFKData through either wfk or data, not both")
        data = wfk
    elif data is None:
        if wfk is None:
            raise ValueError("provide a WFK path or WFKData")
        data = read_wfk(wfk)
    elif wfk is not None:
        raise ValueError("provide either wfk or data, not both")

    if not isinstance(data, WFKData):
        raise TypeError("data must be WFKData")
    if data.usepaw:
        raise ValueError("PAW WFK requires unfold_abinit_paw with matching PAW XML datasets")
    if unfold_sc_mat is None:
        raise ValueError("unfold_sc_mat is required")
    if kpts is None:
        raise ValueError("kpts (the primitive-cell k-path) is required")
    if fermi_shift and data.fermi_energy is None:
        raise ValueError(
            "WFK has no fermi_energy header; pass fermi_shift=False "
            "to plot unshifted energies"
        )

    eigendata = PWEigenData(data.kpoints, data.gvecs, data.coefficients, data.eigenvalues)
    result = PWUnfolder(eigendata, unfold_sc_mat).compute(
        kpts, spin=spin,
        resolve_degenerate=(
            resolve_degenerate / HARTREE_TO_EV
            if resolve_degenerate is not None else None
        ),
    )
    shift = data.fermi_energy * HARTREE_TO_EV if fermi_shift else 0.0
    result_ev = PWWeights(
        result.kpoints,
        result.eigenvalues * HARTREE_TO_EV - shift,
        result.weights,
        result.fold_kpoints,
        result.sc_kpoints,
    )
    if average_degenerate is not None:
        result_ev = result_ev.average_degenerate(float(average_degenerate))
    ax = result_ev.plot(
        xqpts=xqpts,
        style=style,
        axis=axis,
        ylabel=ylabel,
        efermi=0.0 if fermi_shift else None,
        yrange=yrange,
        color=color,
        width=width,
        xticks=[knames, Xqpts] if knames is not None and Xqpts is not None else None,
        title=title,
        ypad=ypad,
    )
    if output is not None:
        ax.figure.savefig(output)
    return ax
