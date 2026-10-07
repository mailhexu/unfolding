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


_QOFF = 1 << 25  # offset keeping quantized deltas non-negative (|delta| < 32)


def _pack_hi(hi, qx):
    """Pack (small id, quantized delta component) into one int64 key."""
    return (np.asarray(hi, dtype=np.int64) << 26) + (np.asarray(qx, dtype=np.int64) + _QOFF)


def _pack_lo(qy, qz):
    """Pack two quantized delta components into one int64 key."""
    return ((np.asarray(qy, dtype=np.int64) + _QOFF) << 26) + (np.asarray(qz, dtype=np.int64) + _QOFF)


def _group_mean(ids, dq, vals):
    """Mean of ``vals`` over duplicate ``(ids, dq)`` keys (sort-based).

    ``ids`` is a per-row small integer id (orbital pair), ``dq`` the
    (rows, 3) int64 quantized displacement. Returns ``(uk, means)`` with
    ``uk`` columns ``(id, qx, qy, qz)`` (components de-quantized only by
    the caller's scale). Uses the packed two-int64 fast path when every
    component fits the 26-bit field; falls back to a column lexsort for
    arbitrarily large displacements.
    """
    if np.abs(dq).max() < _QOFF:
        kA = _pack_hi(ids, dq[:, 0])
        kB = _pack_lo(dq[:, 1], dq[:, 2])
        order = np.lexsort((kB, kA))
        ka, kb = kA[order], kB[order]
        key1, key2 = ka, kb
        unpack = True
    else:
        order = np.lexsort((dq[:, 2], dq[:, 1], dq[:, 0], ids))
        key1 = np.asarray(ids)[order]
        key2 = dq[order, 0]
        ka = np.stack([key1, dq[order, 0], dq[order, 1], dq[order, 2]], axis=1)
        unpack = False
    v = vals[order]
    if unpack:
        new = np.empty(len(ka), dtype=bool)
        new[0] = True
        np.logical_or(ka[1:] != ka[:-1], kb[1:] != kb[:-1], out=new[1:])
        starts = np.flatnonzero(new)
        mask = (1 << 26) - 1
        uk = np.stack([ka[starts] >> 26, (ka[starts] & mask) - _QOFF,
                       (kb[starts] >> 26) - _QOFF, (kb[starts] & mask) - _QOFF],
                      axis=1)
    else:
        d0, d1, d2 = dq[order, 0], dq[order, 1], dq[order, 2]
        new = np.empty(len(key1), dtype=bool)
        new[0] = True
        np.logical_or.reduce([key1[1:] != key1[:-1], d0[1:] != d0[:-1],
                              d1[1:] != d1[:-1], d2[1:] != d2[:-1]],
                             out=new[1:])
        starts = np.flatnonzero(new)
        uk = np.stack([key1[starts], d0[starts], d1[starts], d2[starts]], axis=1)
    cnt = np.diff(np.append(starts, len(v)))
    sr = np.add.reduceat(v.real, starts)
    si = np.add.reduceat(v.imag, starts)
    return uk, (sr + 1j * si) / cnt


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
        # k-independent caches (built lazily; structure-only constants)
        self._invT = np.linalg.inv(self._scmat.T)
        self._sector_span = None
        self._ring_tables = None   # (blocks (N,N,n,n), deltas (N,N,3))
        self._bra_cells = None     # (N, n, n_orb_sc) S_AO rows per cell

    # -- geometry -----------------------------------------------------------

    def _build_S_AO(self):
        """Position-resolved supercell AO overlap assembled from ``SR``.

        ``S_AO[p, q]`` is the plain sum of every stored ``SR[T]`` element
        for the orbital pair ``(p, q)``. Each pair appears at most once
        per shell, so on single-shell archives this is just the ``R=0``
        block, while multi-shell (k-grid) archives contribute the full
        set of image elements the pair realizes. Carries any defect's
        position dependence, unlike a single ``SR`` block.
        """
        n_orb = self._n_orb_sc
        S_AO = np.zeros((n_orb, n_orb), dtype=complex)
        for block in self._model.SR.values():
            S_AO += np.asarray(block)
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
        invT = self._invT
        span = self._sector_span or max(1, int(np.abs(self._scmat).max()))
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
        self._sector_span = span
        members = list(seen.values())[: self._n_cells]
        assert len(members) == self._n_cells, (len(members), self._n_cells)
        if not any(np.allclose(cand, k, atol=1e-8) for cand in members):
            members = [np.asarray(k, dtype=float)] + members[1:]
        return members

    def _sector_gram_block(self, ki, kj):
        """``<k_i m|S_AO|k_j m'>/N`` in the cell-offset gauge (story 4/5).

        The ket phase is anchored at the bra cell and the pair
        displacement carried by ``S_AO``; with the exact-integer
        supercell wraps of ``scmat_keys`` the displacement reduces to
        the intra-cell coordinates and the bra-phase sum kills every
        ``k_i != k_j`` block for a translation-invariant overlap.
        Multi-atom generic-k gauges (intra-cell coordinates in the
        phases) are an open derivation item (story 010).
        """
        blocks, deltas = self._get_ring_tables()
        ki = np.asarray(ki, dtype=float)
        kj = np.asarray(kj, dtype=float)
        off = np.asarray(self._offsets, dtype=float)
        arg = (np.einsum('x,cpx->cp', kj, deltas)
               + np.einsum('x,cx->c', kj - ki, off)[:, None])
        ph = np.exp(2j * np.pi * arg)
        Gb = np.einsum('cp,cpmn->mn', ph, blocks)
        return Gb / self._n_cells

    def _get_ring_tables(self):
        """k-independent sector-Gram tables, built once.

        ``blocks[c, cp, m, m'] = S_AO[rep[m, c], rep[m', cp']]`` and
        ``deltas[c, cp] = offsets[cp] - offsets[c] + T(c, cp) @ scmat``
        (the per-pair displacement of ``_sector_gram_block``); both are
        pure structure quantities consumed by the ring weight path.
        """
        if self._ring_tables is None:
            N, n = self._n_cells, self._n_orb_prim
            # assemble the (N, N, n, n) block stack in place: one dense
            # n_orb_sc^2 allocation, no intermediate gather or transpose copy
            blocks = np.empty((N, N, n, n), dtype=self._S_AO.dtype)
            for c in range(N):
                rows = self._rep[:, c]
                for cp in range(N):
                    blocks[c, cp] = self._S_AO[np.ix_(rows, self._rep[:, cp])]
            off = np.asarray(self._offsets, dtype=float)
            scmat = self._scmat.astype(float)
            deltas = np.empty((N, N, 3))
            for (c, cp), T in self._rm.scmat_keys.items():
                deltas[c, cp] = off[cp] - off[c] + np.asarray(T, dtype=float) @ scmat
            self._ring_tables = (blocks, deltas)
        return self._ring_tables

    def _sector_gram(self, members):
        """Sector Gram for all member pairs in one pass per k-point.

        Identical math to :meth:`_sector_gram_block` over every
        ``(k_i, k_j)`` pair, assembled as phase-tensor contractions of
        the cached ring tables instead of ``n_cells**4`` Python
        iterations.
        """
        blocks, deltas = self._get_ring_tables()
        Km = np.asarray(members, dtype=float)                  # (M, 3)
        off = np.asarray(self._offsets, dtype=float)           # (N, 3)
        u = np.exp(2j * np.pi * np.einsum('jx,cpx->jcp', Km, deltas))
        W = np.einsum('jcp,cpmn->jcmn', u, blocks)             # (M, N, n, n)
        off_proj = np.einsum('jx,cx->jc', Km, off)             # (M, N)
        v = np.exp(2j * np.pi * (off_proj[None, :, :] - off_proj[:, None, :]))
        G = np.einsum('ijc,jcmn->imjn', v, W) / self._n_cells  # (M, n, M, n)
        M, n = len(members), self._n_orb_prim
        return G.reshape(M * n, M * n)

    def _bra_overlap(self, k):
        """Plain AO overlaps ``A[m, s] = <k m|s>/...`` from ``S_AO``."""
        if self._bra_cells is None:
            rep = self._rep
            self._bra_cells = np.stack(
                [self._S_AO[rep[:, c], :] for c in range(self._n_cells)]
            )                                                  # (N, n, n_sc)
        ph = np.exp(-2j * np.pi * (
            np.asarray(self._offsets, dtype=float) @ np.asarray(k, dtype=float)
        ))
        A = np.einsum('c,csm->sm', ph, self._bra_cells)
        return A / np.sqrt(self._n_cells)

    def ideal_gram(self, k):
        """Primitive (Popescu-Zunger / Lee) overlap matrix ``S_p(k)``.

        Infinite-lattice (BvK-converged) construction: every stored
        ``SR[T]`` shell block is folded onto the primitive orbital
        level at its physical displacement
        ``delta = r0' + T @ scmat - r0`` and accumulated with the Bloch
        phase ``exp(2i pi k.delta)``::

            S_p(k)[m, m'] = sum_delta exp(2i pi k.delta) S_prim(delta)

        This is the Gram of the *ideal* primitive Bloch-sum family,
        whose dual defines the standard band-unfolding weight. It
        equals the torus (ring) sector Gram's self block at commensurate
        momenta; at generic momenta the two definitions differ (the
        ring samples each image class once on the torus, the ideal sum
        uses the full shell set).

        Requires the shell keys to cover ``delta`` and ``-delta``
        symmetrically (true for SIESTA R-shell lists and the toy
        builders); on asymmetric shells the result degrades gracefully
        to the covered range.
        """
        k = np.asarray(k, dtype=float)
        n = self._n_orb_prim
        deltas, vals, km1, km2 = self._ideal_gram_terms()
        Gp = np.zeros(n * n, dtype=complex)
        ph = vals * np.exp(2j * np.pi * (deltas @ k))
        Gp.real = np.bincount(km1 * n + km2, weights=ph.real, minlength=n * n)
        Gp.imag = np.bincount(km1 * n + km2, weights=ph.imag, minlength=n * n)
        return Gp.reshape(n, n)

    def _ideal_gram_terms(self):
        """Cache ``(deltas, vals, m1, m2)`` for :meth:`ideal_gram`.

        Each entry realizes one unique ``(m, m', delta)`` with the mean of
        the stored ``SR`` blocks that realize it; ``vals`` carries that
        mean so the per-k evaluation is a single phase-weighted scatter.
        """
        cache = getattr(self, "_ideal_gram_cache", None)
        if cache is not None:
            return cache
        n_sc = self._n_orb_sc
        m_idx = np.asarray(self._rm.orb_to_m)
        r0 = self._rm.orb_to_r0.astype(float)
        scmat = self._scmat.astype(float)
        intra = r0[None, :] - r0[:, None]                  # (s, s', 3)
        m1_g = np.broadcast_to(m_idx[:, None], (n_sc, n_sc)).reshape(-1)
        m2_g = np.broadcast_to(m_idx[None, :], (n_sc, n_sc)).reshape(-1)
        pair = m1_g * n_sc + m2_g
        ids, dqs, vals = [], [], []
        for T, block in self._model.SR.items():
            Tv = np.asarray(_key3(T), dtype=float)
            delta = intra + Tv @ scmat                     # (s, s', 3)
            dqs.append(np.round(delta * 1e6).astype(np.int64).reshape(-1, 3))
            ids.append(pair)
            vals.append(np.asarray(block).reshape(-1))
        uk, vmean = _group_mean(np.concatenate(ids), np.concatenate(dqs),
                                np.concatenate(vals))
        km1, km2 = np.divmod(uk[:, 0], n_sc)
        deltas = uk[:, 1:4] / 1e6
        self._ideal_gram_cache = (deltas, vmean, km1, km2)
        return self._ideal_gram_cache

    def _ideal_bra_overlap(self, k):
        """Open-lattice AO overlaps ``A[m, s] = <k m|s>`` for the ideal
        weight: the bra sum runs over cells with Bloch phases and the
        overlap values are the *open-lattice* elements at the physical
        displacement ``r0(s) - r0(m cell) + T @ scmat`` (torus wraps do
        not fold the displacement). Multiple stored realizations of the
        same displacement are averaged, matching :meth:`ideal_gram`.
        """
        k = np.asarray(k, dtype=float)
        n = self._n_orb_prim
        n_sc = self._n_orb_sc
        m_arr, s_arr, rbra, vals = self._ideal_bra_terms()
        ph = vals * np.exp(-2j * np.pi * (rbra @ k))
        A = np.zeros(n * n_sc, dtype=complex)
        idx = m_arr * n_sc + s_arr
        A.real = np.bincount(idx, weights=ph.real, minlength=n * n_sc)
        A.imag = np.bincount(idx, weights=ph.imag, minlength=n * n_sc)
        return A.reshape(n, n_sc) / np.sqrt(self._n_cells)

    def _ideal_bra_terms(self):
        """Cache ``(m, s, R_bra, vals)`` for :meth:`_ideal_bra_overlap`."""
        cache = getattr(self, "_ideal_bra_cache", None)
        if cache is not None:
            return cache
        n = self._n_orb_prim
        n_sc = self._n_orb_sc
        r0 = self._rm.orb_to_r0.astype(float)
        scmat = self._scmat.astype(float)
        rep = self._rep
        m_g = np.broadcast_to(np.arange(n)[:, None], (n, n_sc)).reshape(-1)
        s_g = np.broadcast_to(np.arange(n_sc)[None, :], (n, n_sc)).reshape(-1)
        pair = m_g * n_sc + s_g
        ids, dqs, vals = [], [], []
        for c in range(self._n_cells):
            src = rep[:, c]  # SC orbitals of (m, c), m = 0..n-1
            for T, block in self._model.SR.items():
                Tv = np.asarray(_key3(T), dtype=float)
                delta = r0[None, :] - r0[src, None] + Tv @ scmat
                # element <m c|s> = block[src[m], s] at displacement delta[m, s];
                # the bra's total lattice position is R = r0[s] - delta
                blk = np.asarray(block)[src, :]
                dqs.append(np.round(delta * 1e6).astype(np.int64).reshape(-1, 3))
                ids.append(pair)
                vals.append(blk.reshape(-1))
        uk, vmean = _group_mean(np.concatenate(ids), np.concatenate(dqs),
                                np.concatenate(vals))
        m_arr, s_arr = np.divmod(uk[:, 0], n_sc)
        # R_bra is reconstructed exactly from the unique key
        rbra = r0[s_arr] - uk[:, 1:4] / 1e6
        self._ideal_bra_cache = (m_arr, s_arr, rbra, vmean)
        return self._ideal_bra_cache

    def compute(self, kpoints, method: str = "ring",
                atol_imag: float = 1e-8, atol_orth: float | None = None):
        """Weights of every supercell band at each primitive k-point.

        Dual-basis spectral weight (paper Eq. 25; sealed against the
        story-4/5 dual-basis derivations). Two weight definitions are
        available:

        ``method="ring"`` (default)
            Exact supercell-torus projection: the sector of primitive
            k's folding to the same supercell momentum K, the S-metric
            Gram of the sector Bloch kets
            (``G[(i m),(j m')] = <k_i m|S_AO|k_j m'>/N``), the plain AO
            overlaps, and the self-k rows of ``x = G^-1 A``. This is
            the identity sealed by the story-4/5 derivations: exact on
            the BvK ring, Parseval-exact over the folded grid for
            pristine *and* defect supercells.

        ``method="ideal"``
            Standard band-unfolding weight (Popescu-Zunger / Lee):
            ``w = c^dagger S_p(k)^-1 c`` with the open-lattice Bloch
            overlaps (:meth:`_ideal_bra_overlap`) and the open-lattice
            primitive Gram :meth:`ideal_gram`, both built from the
            stored shell set with torus wraps left unfolded. For a
            pristine supercell this weight is exactly 0 or 1 at every
            momentum - each supercell band is a folded primitive band -
            matching the plane-wave / phonopy unfolding convention at
            generic momenta. Requires shells covering the image set of
            each displacement class: single-shell (Gamma-only) archives
            give data-limited ideal weights (only the commensurate grid
            is sealed there).

        The position-resolved ``S_AO`` carries any defect's position
        dependence. Returns an :class:`LCAOWeights`.
        """
        if method not in ("ring", "ideal"):
            raise ValueError(f"unknown weight method: {method!r}")
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        n = self._n_orb_prim
        all_w = np.zeros((len(kpoints), self._n_orb_sc))
        all_e = np.zeros((len(kpoints), self._n_orb_sc))
        for ik, k in enumerate(kpoints):
            K = self._fold(k)
            out = self._model.hs_and_eigen(K)
            H, S = out[0], out[1]
            # scipy eigenvector S-orthonormality (story verification item);
            # opt-in: pass atol_orth to enable (costs an O(n^3) matmul per k)
            eps, C = eigh(H, S)
            if atol_orth is not None:
                assert np.allclose(np.conj(C).T @ S @ C, np.eye(len(S)), atol=atol_orth)

            if method == "ideal":
                A = self._ideal_bra_overlap(k) @ C
                Gp = self.ideal_gram(k)
                X = np.conj(A) * np.linalg.solve(Gp, A)
                w = np.real(np.sum(X, axis=0))
                assert np.abs(np.imag(np.sum(X, axis=0))).max() < atol_imag
                all_w[ik] = w
                all_e[ik] = eps
                continue

            members = self._sector(k)
            A_stack = np.vstack([self._bra_overlap(ki) @ C for ki in members])
            G = self._sector_gram(members)
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
