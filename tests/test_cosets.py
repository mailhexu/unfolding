"""Unit tests for the shared k-point folding services (story 046).

Seeded from the shared-core prototype checks: skew supercell matrices,
zero-first coset order, fold/unfold roundtrip, sector membership and
member-count, matching success/failure paths, and delegate parity for
the engine class methods that now forward to :mod:`unfolding.cosets`.
"""
from itertools import product

import numpy as np
import pytest

from unfolding import cosets
from unfolding.pw_unfolder import PWUnfolder
from unfolding.magnon_unfolder import MagnonUnfolder

MATS = [
    np.diag([2, 2, 2]),
    np.diag([3, 2, 1]),
    np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]]),   # skew
]


@pytest.mark.parametrize("M", MATS)
def test_fold_kpoints_count_and_zero_first(M):
    folds = cosets.fold_kpoints(M)
    nfold = int(abs(round(np.linalg.det(M))))
    assert len(folds) == nfold
    assert np.allclose(folds[0], 0.0)
    # coset representatives satisfy t @ M.T = integer (they fold to Gamma)
    for t in folds:
        assert np.allclose(np.mod(cosets.fold(t, M), 1.0), 0.0, atol=1e-9) \
            or np.allclose(np.abs(np.mod(cosets.fold(t, M), 1.0) - 1.0), 0.0, atol=1e-9)
    keys = {tuple(np.round(np.mod(f, 1.0), 9)) for f in folds}
    assert len(keys) == nfold


@pytest.mark.parametrize("M", MATS)
def test_engine_fold_kpoints_delegates(M):
    assert np.array_equal(PWUnfolder.fold_kpoints(M), cosets.fold_kpoints(M))
    assert np.array_equal(MagnonUnfolder.fold_kpoints(M), cosets.fold_kpoints(M))


@pytest.mark.parametrize("M", MATS)
def test_fold_unfold_roundtrip(M):
    rng = np.random.default_rng(0)
    for _ in range(5):
        k = rng.random(3)
        assert np.allclose(cosets.unfold(cosets.fold(k, M), M), k, atol=1e-12)


@pytest.mark.parametrize("M", MATS)
def test_sector_members(M):
    rng = np.random.default_rng(1)
    n_cells = int(abs(round(np.linalg.det(M))))
    k = rng.random(3)
    members, span = cosets.sector(k, M, n_cells)
    assert len(members) == n_cells
    assert span >= 1
    # the requested k is a member; all members fold to the same momentum
    assert any(np.allclose(m, k, atol=1e-8) for m in members)
    K = cosets.fold(k, M)
    for m in members:
        delta = np.mod(cosets.fold(m, M) - K + 0.5, 1.0) - 0.5
        assert np.allclose(delta, 0.0, atol=1e-8)


def test_match_stored_k_roundtrip_and_error():
    stored = np.mod(np.array([[i, j, l] for i in range(2)
                              for j in range(2) for l in range(2)]) / 2.0, 1.0)
    M = np.diag([2, 2, 2])
    # k = (0.5002, 0, 1) folds to ~0.0004 -> nearest stored is zero row
    ik = cosets.match_stored_k(stored, np.array([0.5002, 0.0, 1.0]), M, tol_k=1e-2)
    # k=(0.25,0,0) folds to (0.5,0,0) -> stored row 4
    assert cosets.match_stored_k(stored, np.array([0.25, 0.0, 0.0]), M,
                                 tol_k=1e-12) == 4
    with pytest.raises(ValueError, match=r"target SC k-point.*nearest stored"):
        cosets.match_stored_k(stored, np.array([0.125, 0.0, 0.0]), M, tol_k=1e-4)


def test_match_stored_k_periodic_wrap():
    # stored at 0.99 should match target 0.01 across the zone boundary
    stored = np.array([[0.99, 0.0, 0.0]])
    ik = cosets.match_stored_k(stored, np.array([0.0025, 0.0, 0.0]),
                               np.diag([2, 2, 2]), tol_k=0.02)
    assert ik == 0


def test_fold_kpoints_validation():
    with pytest.raises(ValueError, match="shape"):
        cosets.fold_kpoints(np.diag([2, 2]))
    with pytest.raises(ValueError, match="nonsingular"):
        cosets.fold_kpoints(np.zeros((3, 3), dtype=int))
