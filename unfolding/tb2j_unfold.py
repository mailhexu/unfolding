"""TB2J adapter: one-call magnon downfolding from TB2J results (story
023, PRD Rev 3 FR-020, ADR-012/015).

Takes a ``TB2J.pickle`` (or a configured TB2J ``Magnon``), evaluates the
magnon eigendata at the supercell momenta of a primitive-cell q-path,
converts TB2J's Cholesky-frame wavefunctions to the canonical BdG
amplitudes sealed by ``derivations/magnon_weight.py`` (``Psi = K @ X``,
Euclidean-normalized rows), and unfolds through
:class:`unfolding.magnon_unfolder.MagnonUnfolder`.

Reference convention (ADR-015): the default is the collinear
two-sublattice frame -- ``Q = 0``, quantization axis ``uz = [[0, 0, 1]]``,
rotation axis ``n = [1, 0, 0]`` and the magnetic moments read from the
pickle. This matches the TB2J example CLI defaults; spiral (``Q != 0``)
references are not supported in v1 (they change the eigenvector phase
convention between cells and are not covered by the sealed weight).
"""
from __future__ import annotations

import numpy as np

from .magnon_unfolder import MagnonEigenData, MagnonUnfolder

__all__ = ["unfold_tb2j", "magnon_eigendata_from_tb2j"]


def _canonical_amplitudes(H, X):
    """(nk, nmode, 2N) TB2J Cholesky-frame modes -> canonical, normalized.

    Psi = K @ X with K = chol(H) (mirroring TB2J's fallback shift when H
    is not positive definite); rows are Euclidean-normalized so both BdG
    sectors of a clean fold state carry the same fold character exactly.
    """
    nk, nmode, dim = X.shape
    n = dim // 2
    psi = np.empty_like(X)
    for ik in range(nk):
        Hk = H[ik]
        try:
            K = np.linalg.cholesky(Hk)
        except np.linalg.LinAlgError:
            shift = np.min(np.linalg.eigvalsh(Hk))
            K = np.linalg.cholesky(Hk - (shift - 1e-9) * np.eye(dim))
        cand = (K @ X[ik].T).T
        psi[ik] = cand / np.linalg.norm(cand, axis=1)[:, None]
    return psi


def magnon_eigendata_from_tb2j(magnon, kpoints_sc) -> MagnonEigenData:
    """Evaluate a configured TB2J ``Magnon`` at supercell momenta.

    ``kpoints_sc`` are supercell-fractional momenta (one per requested
    primitive q, ``K = frac(q @ M.T)``). Returns the engine eigendata
    with canonical amplitudes.
    """
    kpoints_sc = np.asarray(kpoints_sc, dtype=float)
    es = magnon.get_magnon_eigenstates(kpoints_sc, include_wavefunctions=True)
    H = magnon.Hq(kpoints_sc)
    psi = _canonical_amplitudes(H, es.wavefunctions)
    # Magnon.positions are cartesian (Angstrom); the engine wants
    # supercell fractional coordinates
    cell = np.asarray(magnon.cell, dtype=float)
    positions = np.asarray(magnon.positions, dtype=float) @ np.linalg.inv(cell).T
    return MagnonEigenData(
        kpoints=kpoints_sc,
        energies=es.energies,
        wavefunctions=psi,
        positions=positions,
    )


def unfold_tb2j(
    source,
    unfold_sc_mat,
    qpts,
    knames=None,
    xqpts=None,
    Xqpts=None,
    spin_conf=None,
    degen_tolerance=1e-5,
    axis=None,
    output=None,
    style="alpha",
    color="blue",
    width=2,
    title=None,
    ylabel="Energy (meV)",
):
    """Downfold a TB2J magnon band structure onto a primitive q-path.

    Parameters
    ----------
    source : path-like or ``TB2J.magnon.Magnon``
        A ``TB2J_results`` directory containing ``TB2J.pickle``, or a
        pre-configured ``Magnon`` (its reference settings are used as-is).
    unfold_sc_mat : (3, 3) integer array
        Row convention ``A_sc = M @ A_prim`` mapping the downfold target
        (primitive) cell to the TB2J cell.
    qpts : (nk, 3) array-like
        Primitive-cell fractional q-path.
    knames, xqpts, Xqpts : sequence, optional
        Plot labels, path coordinates, tick positions.
    spin_conf : (nspin, 3) array-like, optional
        Override the collinear magnetic moments (e.g. ``[[0, 0, 3],
        [0, 0, -3]]``); by default the moments from the pickle are used.
    degen_tolerance : float, optional
        Energy tolerance for the degenerate-group presentation (eV);
        1e-5 eV by default -- symmetry degeneracies (the Q-periodic
        G-AFM fold pairs) hold to the DFT extraction noise of ~1e-7 eV,
        far below any physical magnon splitting.
    axis, output, style, color, width, title, ylabel
        Plotting options forwarded to ``MagnonWeights.plot``; energies
        are converted to meV for plotting.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - plotting is core
        raise ImportError("matplotlib is required for unfold_tb2j") from exc

    if hasattr(source, "get_magnon_eigenstates"):
        magnon = source
    else:
        from TB2J.magnon.magnon3 import Magnon

        magnon = Magnon.from_TB2J_results(path=str(source))

    qpts = np.asarray(qpts, dtype=float)
    M = np.asarray(unfold_sc_mat, dtype=int)
    if spin_conf is not None:
        magnon.set_reference(
            Q=(0, 0, 0),
            uz=np.array([[0.0, 0.0, 1.0]]),
            n=np.array([1.0, 0.0, 0.0]),
            magmoms=np.asarray(spin_conf, dtype=float),
        )

    kpoints_sc = np.mod(qpts @ M.T, 1.0)
    eigendata = magnon_eigendata_from_tb2j(magnon, kpoints_sc)
    unf = MagnonUnfolder(eigendata, M)
    res = unf.compute(qpts, resolve_degenerate=degen_tolerance)

    if xqpts is None:
        xqpts = np.arange(len(qpts), dtype=float)
    scale = 1000.0  # eV -> meV for magnon plotting
    from .magnon_unfolder import MagnonWeights

    res_mev = MagnonWeights(
        res.kpoints,
        res.energies * scale,
        res.weights,
        res.sc_kpoints,
    )
    ax = res_mev.plot(
        xqpts=xqpts,
        xticks=(knames, Xqpts) if knames is not None else None,
        style=style,
        color=color,
        width=width,
        ylabel=ylabel,
        title=title,
        axis=axis,
    )
    if output is not None:
        ax.figure.savefig(output, dpi=200, bbox_inches="tight")
    return ax
