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
        max_groups = 0
        for ik in range(len(kpoints)):
            e = self.eigenvalues[ik]
            cnt, start = 0, 0
            for i in range(1, len(e) + 1):
                if i == len(e) or e[i] - e[i - 1] > tol_e:
                    cnt += 1
                    start = i
            max_groups = max(max_groups, cnt)
        return LCAOWeights(kpoints, eig[:, :max_groups], weights[:, :max_groups])


def _key3(T):
    """Normalize an SR key to a 3-tuple (scalar -> (T, 0, 0))."""
    if isinstance(T, (int, float, np.integer)):
        return (int(T), 0, 0)
    T = tuple(int(v) for v in T)
    return T + (0,) * (3 - len(T))


class HamiltonIOModel:
    """Backend-neutral view of a HamiltonIO model (thin adapter).

    Translates the HamiltonIO surface -- ``Rlist`` (supercell-lattice
    translation tuples) with ``SR``/``HR`` as ``(nR, n, n)`` arrays, and
    ``HS_and_eigen(kpts) -> (H, S, eigenvalues, eigenvectors)`` -- into
    the dict/list interface :class:`LCAOUnfolder` consumes, without
    importing HamiltonIO. The wrapped model is read-only.

    For collinear spin-polarized calculations a SislParser model exposes
    one channel per spin; construct one ``LCAOUnfolder`` per channel
    from the same ``RelabelMap``.
    """

    def __init__(self, model):
        self._model = model
        self._SR = self._as_dict(getattr(model, "SR", None), getattr(model, "Rlist", None))
        self._HR = self._as_dict(getattr(model, "HR", None), getattr(model, "Rlist", None))

    @staticmethod
    def _as_dict(mat, rlist):
        if mat is None:
            return None
        if isinstance(mat, dict):
            return mat
        arr = np.asarray(mat)
        return {_key3(T): arr[i] for i, T in enumerate(rlist)}

    @property
    def SR(self):
        return self._SR

    @property
    def HR(self):
        return self._HR

    @property
    def atoms(self):
        return self._model.atoms

    def hs_and_eigen(self, k):
        """Return (H, S) at supercell fractional k, folding the batched
        four-value HamiltonIO ``HS_and_eigen`` result."""
        out = self._model.HS_and_eigen(np.atleast_2d(np.asarray(k, dtype=float)))
        H, S = out[0], out[1]
        if np.ndim(H) == 3:  # HamiltonIO returns a leading k-point axis
            H, S = H[0], S[0]
        return H, S


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
    """

    def __init__(self, model, relabel_map):
        self._model = model
        self._rm = relabel_map
        self._scmat = np.asarray(relabel_map.scmat, dtype=int)
        self._offsets = relabel_map.offsets
        self._rep = relabel_map.rep_orbital
        self._n_orb_prim = self._rep.shape[0]
        self._n_cells = len(self._offsets)
        self._n_orb_sc = len(relabel_map.orb_to_m)
        self._S_AO = self._build_S_AO()

    # -- geometry -----------------------------------------------------------

    def _build_S_AO(self):
        """Position-resolved supercell AO overlap assembled from ``SR``.

        ``S_AO[p, q] = SR[T][p, q]`` where ``T`` is the supercell
        translation taking orbital ``q``'s cell onto orbital ``p``'s
        cell (empty pairs contribute zero). Carries any defect's
        position dependence, unlike a single ``SR`` block.
        """
        n_orb = self._n_orb_sc
        S_AO = np.zeros((n_orb, n_orb), dtype=complex)
        inv_scmat = np.linalg.inv(self._scmat.astype(float))
        sr_by_key = {_key3(T): (np.asarray(_key3(T), dtype=float), b)
                     for T, b in self._model.SR.items()}
        for c in range(self._n_cells):
            for cp in range(self._n_cells):
                T0 = np.asarray(self._rm.scmat_keys[(c, cp)], dtype=float)
                idx_c = self._rep[:, c]
                idx_cp = self._rep[:, cp]
                # every SR translation congruent to T0 modulo the
                # supercell lattice contributes (wrapped torus images)
                for (Tv, block) in sr_by_key.values():
                    h = (Tv - T0) @ inv_scmat
                    if not np.allclose(h, np.round(h), atol=1e-6):
                        continue
                    S_AO[np.ix_(idx_c, idx_cp)] += block[np.ix_(idx_c, idx_cp)]
        return S_AO

    # -- overlap vectors ---------------------------------------------------

    def _fold(self, k):
        return np.asarray(k, dtype=float) @ self._scmat.T

    # -- public API ---------------------------------------------------------

    def _sector(self, k):
        """Primitive k's folding to this k's supercell momentum.

        Members are k_h = (g_k + h) @ scmat^{-T} for integer h over a
        box (g_k = k @ scmat^T); k-points differing by a primitive
        reciprocal lattice vector are identified, leaving exactly
        n_cells distinct members. The box grows until the reciprocal
        quotient is complete (skew supercell matrices).
        """
        g_k = self._fold(k)
        invT = np.linalg.inv(self._scmat.T)
        span = max(1, int(np.abs(self._scmat).max()))
        seen = {}
        while len(seen) < self._n_cells:
            seen = {}
            for hx in range(span):
                for hy in range(span):
                    for hz in range(span):
                        cand = (g_k + np.array([hx, hy, hz], dtype=float)) @ invT
                        key = tuple(np.round(cand % 1.0, 8) % 1.0)
                        if key not in seen:
                            seen[key] = cand
            if len(seen) < self._n_cells:
                span *= 2
            if span > 4096:
                break
        members = list(seen.values())[: self._n_cells]
        assert len(members) == self._n_cells, (len(members), self._n_cells)
        if not any(np.allclose(cand, k, atol=1e-8) for cand in members):
            members = [np.asarray(k, dtype=float)] + members[1:]
        return members

    def _sector_gram_block(self, ki, kj):
        """``<k_i m|S_AO|k_j m'>/N`` in the unwrapped-pair gauge.

        The ket phase is anchored at the bra cell plus the actual pair
        displacement ``delta(c, c')`` carried by ``S_AO``, so for a
        translation-invariant (pristine) overlap the cell-origin sum
        kills every ``k_i != k_j`` block exactly and the self block is
        the primitive overlap matrix ``S_p(k)``.
        """
        n = self._n_orb_prim
        Gb = np.zeros((n, n), dtype=complex)
        for c in range(self._n_cells):
            for cp in range(self._n_cells):
                T = np.asarray(self._rm.scmat_keys[(c, cp)], dtype=float)
                # lattice-coordinate displacement (fractional k x lattice
                # integer dot product: the Bloch phase convention)
                delta = self._offsets[cp] - self._offsets[c] + T @ self._scmat
                ph = np.exp(2j * np.pi * (
                    np.asarray(kj, dtype=float) @ (self._offsets[c] + delta)
                    - np.asarray(ki, dtype=float) @ self._offsets[c]
                ))
                Gb += ph * self._S_AO[np.ix_(self._rep[:, c], self._rep[:, cp])]
        return Gb / self._n_cells

    def _bra_overlap(self, k):
        """Plain AO overlaps ``A[m, s] = <k m|s>/...`` from ``S_AO``."""
        A = np.zeros((self._n_orb_prim, self._n_orb_sc), dtype=complex)
        for c in range(self._n_cells):
            ph = np.exp(-2j * np.pi * (
                np.asarray(k, dtype=float) @ self._offsets[c]
            ))
            A += ph * self._S_AO[self._rep[:, c], :]
        return A / np.sqrt(self._n_cells)

    def compute(self, kpoints, atol_imag: float = 1e-8, atol_orth: float = 1e-8):
        """Weights of every supercell band at each primitive k-point.

        Dual-basis spectral weight on the supercell torus (paper
        Eq. 25; sealed against the story-4/5 dual-basis derivations):

        1. the sector of primitive k's folding to the same supercell
           momentum K;
        2. the S-metric Gram of the sector Bloch kets,
           ``G[(i m),(j m')] = <k_i m|S_AO|k_j m'>/N``;
        3. the plain AO overlaps ``A[(i m), band] = <k_i m|psi_band>``;
        4. the self-k rows of ``x = G^-1 A`` give the weights.

        The position-resolved ``S_AO`` carries any defect's position
        dependence. Returns an :class:`LCAOWeights`; the weights of one
        supercell band over the complete unfolding k-grid sum to 1
        (Parseval, story 005).
        """
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        n = self._n_orb_prim
        all_w = np.zeros((len(kpoints), self._n_orb_sc))
        all_e = np.zeros((len(kpoints), self._n_orb_sc))
        for ik, k in enumerate(kpoints):
            K = self._fold(k)
            out = self._model.hs_and_eigen(K)
            H, S = out[0], out[1]
            # scipy eigenvector S-orthonormality (story verification item)
            eps, C = eigh(H, S)
            assert np.allclose(np.conj(C).T @ S @ C, np.eye(len(S)), atol=atol_orth)

            members = self._sector(k)
            A_mats = [self._bra_overlap(ki) @ C for ki in members]
            n_mem = len(members)
            G = np.zeros((n_mem * n, n_mem * n), dtype=complex)
            for i, ki in enumerate(members):
                for j, kj in enumerate(members):
                    G[i * n:(i + 1) * n, j * n:(j + 1) * n] = (
                        self._sector_gram_block(ki, kj)
                    )
            A_stack = np.vstack(A_mats)
            x = np.linalg.solve(G, A_stack)
            i_self = next(
                i for i, cand in enumerate(members)
                if np.allclose(cand, k, atol=1e-8)
            )
            rows = slice(i_self * n, (i_self + 1) * n)
            X = np.conj(A_stack[rows]) * x[rows]
            w = np.real(np.sum(X, axis=0))
            assert np.abs(np.imag(np.sum(X, axis=0))).max() < atol_imag
            all_w[ik] = w
            all_e[ik] = eps
        return LCAOWeights(kpoints, all_e, all_w)
