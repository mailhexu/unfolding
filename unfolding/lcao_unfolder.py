"""LCAO spectral weights for supercell bands on a primitive k-path.

Implements the weight formula of Lee et al., Phys. Rev. B 87, 085322
(2013) (the Eq. 9 -> 25 chain) for LCAO supercell calculations:

    W(K, J; k) = A^dag G_sec(k)^{-1} A,      A_m = sum_s C_s <k m|s>

with the AO-overlap vectors (HamiltonIO convention 2, ``e^{2 pi i}``)

    <k m|s> = sum_c e^{-2 pi i k.o_c} sum_T e^{-2 pi i K.T}
              SR[T][rep(m, c), s] ,        K = scmat^T k ,

and the primitive-cell overlap Bloch matrix

    S_p(k)[n, m] = sum_c e^{+2 pi i k.o_c} sum_T e^{+2 pi i K.T}
                   SR[T][rep(n, 0), rep(m, c)] ,

where ``rep``/``offsets`` come from the RelabelMap and ``T`` runs over
the supercell-lattice keys of the real-space tables. For
translation-invariant overlaps the dual Gram <k n|S|k' m> is exactly
block-diagonal in k (the cell-origin sum cancels k' != k), so the
per-k dual weight A^dag S_p(k)^{-1} A is the exact Eq. 25 evaluation.
Everything is built from the real-space supercell tables ``SR`` only --
no HamiltonIO import happens here. Any model exposing
``(atoms, HR, SR, hs_and_eigen(k))`` works, including a single
collinear spin channel of a SislParser model (story 008 wires those);
the model object is never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass
class LCAOWeights:
    """Spectral weights on a primitive k-path.

    Attributes
    ----------
    kpoints : (N, 3) float array
        Primitive-cell k-points (fractional, convention 2).
    eigenvalues : (N, n_bands) float array
        Supercell band energies at the folded momentum of each k-point.
    weights : (N, n_bands) float array
        Unfolding weight of each supercell band at its k-point.
    """

    kpoints: np.ndarray
    eigenvalues: np.ndarray
    weights: np.ndarray

    def average_degenerate(self, tol_e: float) -> "LCAOWeights":
        """Sum weights within (near-)degenerate eigenspaces.

        Bands whose eigenvalues at a given k-point lie within ``tol_e``
        of each other are collapsed to one entry carrying the sum of
        their weights (the band-averaged spectral weight of the
        eigenspace).
        """
        kpoints = self.kpoints
        weights = np.zeros_like(self.weights)
        eig = np.zeros_like(self.eigenvalues)
        for ik in range(len(kpoints)):
            e = self.eigenvalues[ik]
            w = self.weights[ik]
            out_w, out_e = [], []
            start = 0
            for i in range(1, len(e) + 1):
                if i == len(e) or e[i] - e[i - 1] > tol_e:
                    out_w.append(w[start:i].sum())
                    out_e.append(e[start:i].mean())
                    start = i
            weights[ik, : len(out_w)] = out_w
            eig[ik, : len(out_e)] = out_e
        n_max = int((np.abs(weights) > 0).sum(axis=1).max())
        return LCAOWeights(kpoints, eig[:, :n_max], weights[:, :n_max])


class LCAOUnfolder:
    """Unfolding weights for an LCAO supercell model.

    Parameters
    ----------
    model : object
        Thin wrapper exposing ``HR``/``SR`` (dicts
        ``{R_tuple: (n_orb_sc, n_orb_sc) array}`` over supercell-lattice
        translations, HamiltonIO convention) and
        ``hs_and_eigen(k) -> (H, S, eigenvalues)`` for a supercell
        fractional k-point. Read-only; never mutated.
    relabel_map : RelabelMap
        Supercell -> normal-cell relabel map (story 006).
    scmat : (3, 3) int array
        Supercell matrix in primitive-lattice units (row-vector
        convention, as passed to ``RelabelMap.from_atoms``).
    """

    def __init__(self, model, relabel_map, scmat):
        self._model = model
        self._rm = relabel_map
        self._scmat = np.asarray(scmat, dtype=int)
        self._offsets = relabel_map.offsets
        self._rep = relabel_map.rep_orbital
        self._n_orb_prim = self._rep.shape[0]
        self._n_cells = len(self._offsets)
        self._n_orb_sc = len(relabel_map.orb_to_m)

    # -- overlap vectors ---------------------------------------------------

    def _O_matrix(self, k, K):
        """``O[k][m, s] = <k m|s>`` (n_orb_prim, n_orb_sc)."""
        rm = self._rm
        phases_c = np.exp(-2j * np.pi * (self._offsets @ k))
        O = np.zeros((self._n_orb_prim, self._n_orb_sc), dtype=complex)
        for T, block in self._model.SR.items():
            Tv = np.atleast_1d(np.asarray(T, dtype=float))
            if Tv.ndim == 0:
                Tv = Tv.reshape(1)
            if Tv.shape[0] < 3:  # scalar/per-axis keys: pad along first axes
                Tv = np.concatenate([Tv, np.zeros(3 - Tv.shape[0])])
            # image r = offsets[c] - scmat@T  =>  phase carries +K.T
            # (SR[T][p, q] = <p|(T, q)>: the ket sits at +T, so the bra
            # image that pairs with it lies at -T)
            ph = np.exp(+2j * np.pi * (K @ Tv))
            for c in range(self._n_cells):
                O += (phases_c[c] * ph) * block[self._rep[:, c], :]
        return O

    def _fold(self, k):
        return self._scmat.T @ np.asarray(k, dtype=float)

    # -- public API ---------------------------------------------------------

    def _S_prim(self, k, K):
        """Primitive-cell overlap Bloch matrix S_p(k)[n, m] from SR."""
        S_p = np.zeros((self._n_orb_prim, self._n_orb_prim), dtype=complex)
        phases_c = np.exp(2j * np.pi * (self._offsets @ k))
        rep0 = self._rep[:, 0]
        for T, block in self._model.SR.items():
            Tv = np.atleast_1d(np.asarray(T, dtype=float))
            if Tv.shape[0] < 3:
                Tv = np.concatenate([Tv, np.zeros(3 - Tv.shape[0])])
            ph = np.exp(2j * np.pi * (K @ Tv))
            for c in range(self._n_cells):
                S_p += (phases_c[c] * ph) * block[
                    np.ix_(rep0, self._rep[:, c])
                ]
        return S_p

    def compute(self, kpoints, atol_imag: float = 1e-8, atol_orth: float = 1e-8):
        """Weights of every supercell band at each primitive k-point.

        Returns an :class:`LCAOWeights`; the weights of one supercell
        band over the complete unfolding k-grid sum to 1 (Parseval,
        story 005).
        """
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        all_w = np.zeros((len(kpoints), self._n_orb_sc))
        all_e = np.zeros((len(kpoints), self._n_orb_sc))
        for ik, k in enumerate(kpoints):
            K = self._fold(k)
            H, S, _ = self._model.hs_and_eigen(K)
            # scipy eigenvector S-orthonormality (story verification item)
            eps, C = eigh(H, S)
            assert np.allclose(np.conj(C).T @ S @ C, np.eye(len(S)), atol=atol_orth)

            A = (self._O_matrix(k, K) @ C) / np.sqrt(self._n_cells)
            S_p = self._S_prim(k, K)
            # symmetrize away rounding noise before inversion
            S_p = 0.5 * (S_p + np.conj(S_p).T)
            x = np.linalg.solve(S_p, A)
            w = np.real(np.sum(np.conj(A) * x, axis=0))
            assert np.abs(np.imag(np.sum(np.conj(A) * x, axis=0))).max() < atol_imag
            all_w[ik] = w
            all_e[ik] = eps
        return LCAOWeights(kpoints, all_e, all_w)
