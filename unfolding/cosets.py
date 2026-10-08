"""Shared k-point folding services for all unfolding engines.

Single source of truth (story 046, ADR-018) for the machinery every
engine needs: the row-convention k <-> K mapping
(``K = k @ M.T``, ``A_sc = M @ A_prim``), the reciprocal coset
representatives of ``Z^3 / M^T Z^3``, the per-k folding sector members
used by dual-basis engines, and stored-momentum matching with an
informative error. Pure integer/lookup math — no phase or gauge
content lives here.

The ``fold_kpoints`` semantics are the validated ``PWUnfolder``
implementation (box-grown ``solve(M, n) mod 1`` enumeration with a
1e12-quantized stable key, zero vector first).
"""
from itertools import product

import numpy as np

__all__ = ["fold", "unfold", "fold_kpoints", "sector", "match_stored_k"]


def fold(k, M):
    """Primitive k -> supercell momentum (row convention ``K = k @ M.T``)."""
    return np.asarray(k, dtype=float) @ np.asarray(M, dtype=float).T


def unfold(K, M):
    """Supercell momentum -> primitive k (inverse row convention)."""
    M = np.asarray(M, dtype=float)
    return np.asarray(K, dtype=float) @ np.linalg.inv(M).T


def fold_kpoints(unfold_sc_mat) -> np.ndarray:
    """Primitive fractional coset representatives of ``Z^3 / M^T Z^3``.

    The enumeration uses a bounded integer box and a stable rational
    grid key; the zero vector sorts first. Physical path-sector
    matching itself uses a ``tol_k`` tolerance (see
    :func:`match_stored_k`).
    """
    matrix = np.asarray(unfold_sc_mat, dtype=int)
    if matrix.shape != (3, 3):
        raise ValueError("unfold_sc_mat must have shape (3, 3)")
    nfold = int(abs(round(np.linalg.det(matrix))))
    if nfold == 0:
        raise ValueError("unfold_sc_mat must be nonsingular")

    # Every quotient class has a short integer representative. Grow a
    # small positive box until all |det M| sectors have been found.
    reps = {}
    for bound in range(0, max(nfold, int(np.abs(matrix).max())) + 2):
        for n in product(range(bound + 1), repeat=3):
            frac = np.mod(np.linalg.solve(matrix, np.asarray(n, dtype=float)), 1.0)
            key = tuple(np.rint(frac * 1e12).astype(np.int64) % 10**12)
            reps.setdefault(key, frac)
        if len(reps) == nfold:
            break
    if len(reps) != nfold:
        raise RuntimeError("could not enumerate all reciprocal fold sectors")

    zero = (0, 0, 0)
    values = list(reps.values())
    values.sort(key=lambda value: (not np.allclose(value, 0.0), *value.tolist()))
    assert tuple(np.rint(values[0] * 1e12).astype(np.int64) % 10**12) == zero
    return np.asarray(values)


def sector(k, scmat, n_cells, invT=None, span=None):
    """Primitive k's folding sector members (``LCAOUnfolder._sector``).

    Members are ``k_h = (g_k + h) @ scmat^{-T}`` for integer ``h`` over
    a box; k-points differing by a primitive reciprocal lattice vector
    are identified, leaving exactly ``n_cells`` distinct members. The
    requested k itself is guaranteed to be one of them. Member order is
    first-occurrence box-scan order.

    ``invT`` (the cached ``inv(scmat.T)``) and a previously successful
    ``span`` may be passed to avoid recomputation across calls.
    """
    k = np.asarray(k, dtype=float)
    scmat_i = np.asarray(scmat, dtype=int)
    g_k = fold(k, scmat_i)
    if invT is None:
        invT = np.linalg.inv(np.asarray(scmat, dtype=float).T)
    if span is None:
        span = max(1, int(np.abs(scmat_i).max()))
    seen = {}
    while len(seen) < n_cells:
        seen = {}
        for hx in range(span):
            for hy in range(span):
                for hz in range(span):
                    cand = (g_k + np.array([hx, hy, hz], dtype=float)) @ invT
                    key = tuple(np.round(cand % 1.0, 8) % 1.0)
                    if key not in seen:
                        seen[key] = cand
        if len(seen) < n_cells:
            span *= 2
        if span > 4096:
            break
    members = list(seen.values())[:n_cells]
    assert len(members) == n_cells, (len(members), n_cells)
    if not any(np.allclose(cand, k, atol=1e-8) for cand in members):
        members = [k] + members[1:]
    return members, span


def match_stored_k(stored_kpoints, kpoint, unfold_sc_mat, tol_k):
    """Index of the stored supercell momentum matching ``fold(k, M)``.

    Periodic-delta distance (each component wrapped to [-1/2, 1/2)),
    L2 norm, nearest match; raises ``ValueError`` naming the target,
    tolerance, and nearest stored candidate when nothing is within
    ``tol_k``.
    """
    stored = np.asarray(stored_kpoints, dtype=float)
    target = np.mod(fold(kpoint, unfold_sc_mat), 1.0)
    delta = np.mod(stored - target + 0.5, 1.0) - 0.5
    distances = np.linalg.norm(delta, axis=1)
    ik = int(np.argmin(distances))
    if distances[ik] > tol_k:
        raise ValueError(
            "target SC k-point "
            f"{target.tolist()} has no stored match within tol_k={tol_k:g}; "
            f"nearest stored {stored[ik].tolist()} "
            f"(periodic distance {distances[ik]:.3g})"
        )
    return ik
