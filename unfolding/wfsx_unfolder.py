"""WFSX-driven spectral unfolding: eigendata from a SIESTA wavefunction file.

Instead of diagonalizing the supercell Hamiltonian, this unfolder takes
eigenvalues and eigenvector coefficients from a SIESTA WFSX file and
evaluates the same dual-basis weight formula as
:class:`unfolding.lcao_unfolder.LCAOUnfolder` (Lee et al., PRB 87, 085322
(2013), Eq. 25). All overlap/Gram math is delegated to an internal
:class:`LCAOUnfolder` built on the *same* supercell model (its SR tables
feed the bra overlap and the Gram; the model is never diagonalized here).

Gauge adjudication (the core of this module)
--------------------------------------------
SIESTA stores WFSX coefficients in the *orbital-position gauge*; the HSX
overlap machinery works in convention 2. The conversion is per-orbital

    c_conv2[s, a] = c_sia[s, a] * exp(+2 pi i K . tau_s)

with ``K = k_prim @ scmat`` the supercell-fractional momentum of the
state and ``tau_s`` the supercell-fractional position of the atom
carrying orbital ``s`` (PAO shells sit on atoms; ``tau`` from
``atoms.get_scaled_positions()`` tiled by per-atom orbital counts). The
``+`` sign was adjudicated empirically on the committed
``si_sc_path.selected.WFSX`` fixture: with it, degenerate-group weights
match the HSX diagonalization to ~1e-7 at generic k, while omitting the
factor or flipping the sign deviates by 1e-2 .. 16
(``tests/test_wfsx_unfolder.py::test_gauge_conversion_is_pinned``). At
Gamma every choice gives the trivial factor 1, so a Gamma-only oracle
cannot fix the sign. Per-band overall phases are irrelevant to the
weights (``conj(c) G^-1 c`` is invariant), which is why the conversion
is purely the relative per-orbital pattern.

Eigenvalues are passed through SIESTA-native: WFSX energies are
``e_abs + E_F`` with ``E_F`` of the writing run (e.g. -3.00800582 eV for
the path fixture), i.e. Fermi-shifted as stored; subtract the header
Fermi of the WFSX's own run to recover absolute eigenvalues.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lcao_unfolder import LCAOUnfolder, LCAOWeights


@dataclass
class WFSXWeights:
    """Spectral weights fed by WFSX eigendata.

    Attributes
    ----------
    kpoints : (N, 3) float array
        Primitive-cell k-points, exactly as passed to :meth:`WFSXUnfolder.compute`.
    eigenvalues : (N, n_bands) float array
        SIESTA-native band energies (Fermi-shifted as stored in the WFSX).
    weights : (N, n_bands) float array
        Unfolding weight of each WFSX band at its k-point.
    """

    kpoints: np.ndarray
    eigenvalues: np.ndarray
    weights: np.ndarray

    def average_degenerate(self, tol_e: float) -> "WFSXWeights":
        """Sum weights within (near-)degenerate eigenspaces.

        Same grouping as :meth:`unfolding.lcao_unfolder.LCAOWeights.average_degenerate`,
        reused verbatim on these arrays.
        """
        grouped = LCAOWeights(self.kpoints, self.eigenvalues, self.weights)
        grouped = grouped.average_degenerate(tol_e)
        return WFSXWeights(grouped.kpoints, grouped.eigenvalues, grouped.weights)


class WFSXUnfolder:
    """Unfolding weights from SIESTA WFSX wavefunctions.

    Parameters
    ----------
    wfsx : object
        Duck-typed WFSX data (HamiltonIO ``SiestaWFSXParser(...).read()``):
        ``.kpoints (nk, 3)`` supercell fractional, ``.eigenvalues (nk,
        n_bands)``, ``.coefficients (nk, n_bands, n_orb)`` complex in
        SIESTA's stored gauge. ``n_orb`` must equal the supercell
        orbital count of ``relabel`` (scalar, non-spinor data).
    hs_model : object
        HamiltonIO view of the same supercell (``HamiltonIOModel``);
        only its SR tables are used -- never diagonalized.
    relabel : RelabelMap
        Supercell -> primitive-cell orbital relabel map.
    scmat : (3, 3) int array
        Supercell matrix, row convention ``conv = scmat @ prim``; the
        supercell momentum of a primitive k is ``K = k @ scmat``.
    """

    def __init__(self, wfsx, hs_model, relabel, scmat):
        self._wfsx = wfsx
        self._scmat = np.asarray(scmat, dtype=int)
        self._lcao = LCAOUnfolder(hs_model, relabel)
        self._n_orb_sc = len(relabel.orb_to_m)
        coefficients = np.asarray(wfsx.coefficients)
        if coefficients.shape[2] != self._n_orb_sc:
            raise ValueError(
                f"wfsx coefficients carry {coefficients.shape[2]} orbitals per "
                f"band but the relabel map has {self._n_orb_sc}"
            )
        tau = np.asarray(
            hs_model.atoms.get_scaled_positions(wrap=True), dtype=float
        )
        self._tau = np.repeat(
            tau, [int(c) for c in relabel.orb_counts_sc], axis=0
        )

    # -- gauge --------------------------------------------------------------

    def _gauge_phase(self, K, sign: float = 1.0):
        """Per-orbital conversion factor ``exp(sign * 2 pi i K . tau_s)``.

        ``sign=+1`` maps SIESTA's stored gauge to convention 2 (adjudicated);
        ``sign=0`` / ``sign=-1`` are the omission / flipped-sign controls
        pinned by ``tests/test_wfsx_unfolder.py``.
        """
        return np.exp(
            sign * 2j * np.pi * (self._tau @ np.asarray(K, dtype=float))
        )

    # -- internals ----------------------------------------------------------

    def _match_entry(self, K, k):
        """Index of the WFSX entry at supercell-fractional ``K`` (atol 1e-6).

        1e-6 absorbs the 6-decimal rounding of WaveFuncKPoints blocks in
        committed fdf recipes (residuals up to ~5e-7).
        """
        stored = np.asarray(self._wfsx.kpoints, dtype=float)
        dist = np.abs(stored - np.asarray(K, dtype=float)).max(axis=1)
        j = int(np.argmin(dist))
        if dist[j] > 1e-6:
            raise KeyError(
                f"supercell momentum K = {np.round(K, 8).tolist()} (primitive "
                f"k = {np.round(np.asarray(k, dtype=float), 8).tolist()}) "
                f"matches no WFSX entry within atol=1e-6 (nearest residual "
                f"{dist[j]:.3e})"
            )
        return j

    def _weights_for_coefficients(self, k, C, method: str):
        """Dual-basis weights of the (n_orb, n_bands) coefficient block ``C``.

        Reuses the LCAO internals: the ideal (open-lattice) bra overlap and
        primitive Gram for ``method="ideal"``, the torus bra overlap and the
        self sector-Gram block for ``method="ring"``. Exact only at Gamma for
        the ring branch on multi-atom cells (same restriction as the LCAO
        reference; the generic-k multi-atom sector gauge is an open item).
        """
        if method == "ideal":
            A = self._lcao._ideal_bra_overlap(k) @ C
            G = self._lcao.ideal_gram(k)
        elif method == "ring":
            A = self._lcao._bra_overlap(k) @ C
            G = self._lcao._sector_gram_block(k, k)
        else:
            raise ValueError(f"unknown weight method: {method!r}")
        X = np.conj(A) * np.linalg.solve(G, A)
        return np.real(np.sum(X, axis=0))

    # -- public API ---------------------------------------------------------

    def compute(self, kpoints, method: str = "ideal") -> WFSXWeights:
        """Weights of every WFSX band at each primitive k-point.

        For each ``k`` the supercell momentum ``K = k @ scmat`` locates the
        matching WFSX entry (atol 1e-6; a missing point raises
        :class:`KeyError` listing the unmatched k), the stored coefficients
        are converted to convention 2 (see the gauge note above), and the
        LCAO dual-basis weight is evaluated. Eigenvalues are returned
        SIESTA-native (Fermi-shifted as stored).
        """
        if method not in ("ideal", "ring"):
            raise ValueError(f"unknown weight method: {method!r}")
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        n_bands = np.asarray(self._wfsx.eigenvalues).shape[1]
        all_e = np.zeros((len(kpoints), n_bands))
        all_w = np.zeros((len(kpoints), n_bands))
        for ik, k in enumerate(kpoints):
            K = k @ self._scmat
            j = self._match_entry(K, k)
            C = np.asarray(self._wfsx.coefficients[j]).T.astype(complex)
            C = C * self._gauge_phase(K)[:, None]
            all_w[ik] = self._weights_for_coefficients(k, C, method)
            all_e[ik] = np.asarray(self._wfsx.eigenvalues)[j]
        return WFSXWeights(kpoints, all_e, all_w)
