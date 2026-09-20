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
        self._SR = self._as_dict(getattr(model, "SR"), getattr(model, "Rlist"))
        self._HR = self._as_dict(getattr(model, "HR", None), getattr(model, "Rlist"))

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
        """Return (H, S) at supercell fractional k, folding the four-value
        HamiltonIO ``HS_and_eigen`` result."""
        out = self._model.HS_and_eigen(np.atleast_2d(np.asarray(k, dtype=float))[0])
        return out[0], out[1]


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

    def _sector(self, k):
        """Primitive k's folding to this k's supercell momentum.

        Members are k_h = (g_k + h) @ scmat^{-T} for integer h over a
        box (g_k = k @ scmat^T); k-points differing by a primitive
        reciprocal lattice vector are identified, leaving exactly
        n_cells distinct members.
        """
        g_k = self._fold(k)
        invT = np.linalg.inv(self._scmat.T)
        n = int(np.abs(self._scmat).max()) + 1
        seen = {}
        for hx in range(n):
            for hy in range(n):
                for hz in range(n):
                    cand = (g_k + np.array([hx, hy, hz], dtype=float)) @ invT
                    key = tuple(np.round(cand % 1.0, 8) % 1.0)
                    if key not in seen:
                        seen[key] = cand
        members = list(seen.values())
        assert len(members) == self._n_cells, (len(members), self._n_cells)
        if not any(np.allclose(cand, k, atol=1e-8) for cand in members):
            members = [np.asarray(k, dtype=float)] + members[1:]
        return members

    def _build_S_AO(self):
        """Position-resolved supercell AO overlap assembled from ``SR``.

        ``S_AO[p, q] = SR[T][p, q]`` where ``T`` is the supercell
        translation taking orbital ``q``'s cell onto orbital ``p``'s
        cell (empty pairs contribute zero). Carries any defect's
        position dependence, unlike a single ``SR`` block.
        """
        n_orb = self._n_orb_sc
        S_AO = np.zeros((n_orb, n_orb), dtype=complex)
        sr_by_key = {_key3(T): b for T, b in self._model.SR.items()}
        for c in range(self._n_cells):
            for cp in range(self._n_cells):
                T = self._rm.scmat_keys[(c, cp)]
                block = sr_by_key.get(_key3(T))
                if block is None:
                    continue
                idx_c = self._rep[:, c]
                idx_cp = self._rep[:, cp]
                S_AO[np.ix_(idx_c, idx_cp)] += block[np.ix_(idx_c, idx_cp)]
        return S_AO

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
        return np.asarray(k, dtype=float) @ self._scmat.T

    # -- public API ---------------------------------------------------------

    def _S_prim(self, k, K):
        """Primitive-cell overlap Bloch matrix S_p(k)[n, m] from SR."""
        S_p = np.zeros((self._n_orb_prim, self._n_orb_prim), dtype=complex)
        phases_c = np.exp(2j * np.pi * ((self._offsets - self._offsets[0]) @ k))
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
            out = self._model.hs_and_eigen(K)
            H, S = out[0], out[1]
            # scipy eigenvector S-orthonormality (story verification item)
            eps, C = eigh(H, S)
            assert np.allclose(np.conj(C).T @ S @ C, np.eye(len(S)), atol=atol_orth)

            # sector Gram <k n|S|k' m> as the direct phased double sum
            # over the position-resolved S_AO (cross-k blocks kept: exact
            # for pristine and defect overlaps alike)
            members = self._sector(k)
            n = self._n_orb_prim
            O_mats = [self._O_matrix(cand, self._fold(cand)) / np.sqrt(self._n_cells)
                      for cand in members]
            A = np.vstack(O_mats) @ C
            G = np.zeros((len(members) * n, len(members) * n), dtype=complex)
            for ig, kg in enumerate(members):
                for jg, k2 in enumerate(members):
                    for a in range(n):
                        for b in range(n):
                            acc = 0
                            for c in range(self._n_cells):
                                for cp in range(self._n_cells):
                                    ph = np.exp(-2j * np.pi * (
                                        kg @ self._offsets[c]
                                        - k2 @ self._offsets[cp]))
                                    acc += ph * self._S_AO[
                                        self._rep[a, c], self._rep[b, cp]]
                            G[ig * n + a, jg * n + b] = acc / self._n_cells
            x = np.linalg.solve(G, A)
            i_self = next(
                i for i, cand in enumerate(members)
                if np.allclose(cand, k, atol=1e-8)
            )
            rows = slice(i_self * n, (i_self + 1) * n)
            w = np.real(np.sum(np.conj(A[rows]) * x[rows], axis=0))
            assert np.abs(np.imag(np.sum(np.conj(A[rows]) * x[rows], axis=0))).max() < atol_imag
            all_w[ik] = w
            all_e[ik] = eps
        return LCAOWeights(kpoints, all_e, all_w)
