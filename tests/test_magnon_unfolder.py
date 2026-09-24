"""Behavioral tests for the backend-free magnon unfolding engine (story 022)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from derivations.magnon_weight import (
    NCELL,
    heisenberg_chain_bdg,
    positive_modes,
)
from unfolding import MagnonEigenData, MagnonUnfolder, MagnonWeights

DATA = Path(__file__).parent / "data" / "magnon_synth" / "magnon_synth_fixture.npz"
M = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 1]])
NMAG = 2 * NCELL
POSITIONS = np.array([[j / NMAG, 0.0, 0.0] for j in range(NMAG)])


def _eigendata(js, K):
    energies, modes = positive_modes(heisenberg_chain_bdg(NMAG, js, K))
    return MagnonEigenData(
        kpoints=np.array([K], dtype=float),
        energies=energies[None, :],
        wavefunctions=modes[None, :, :],
        positions=POSITIONS,
    )


def test_pristine_weights_binary_and_sum_rule():
    """Pristine chain: every mode carries w=1 on its on-shell fold."""
    for Kx in (0.0, 0.12, 0.3, 0.5, 0.77):
        K = np.array([Kx, 0.0, 0.0])
        unf = MagnonUnfolder(_eigendata(np.ones(NMAG), K), M)
        folds = MagnonUnfolder.fold_kpoints(M)
        qfolds = np.mod(
            np.linalg.solve(M.T.astype(float), (K + np.eye(3)[0] * 0).T).T, 1.0
        )
        # request exactly the folds of this K: q = M^-T (K + n)
        labels = np.array([[j, 0, 0] for j in range(3)])
        qfolds = np.mod(
            np.linalg.solve(M.T.astype(float), (K + labels).T.astype(float)).T, 1.0
        )
        plain = unf.compute(qfolds)
        res = unf.compute(qfolds, resolve_degenerate=1e-9)
        # invariant sum rule: plain per-mode weights sum to 1 over the
        # folds sharing a stored K
        assert np.allclose(plain.weights.sum(axis=0), 1.0, atol=1e-10)
        # resolved presentation: binary per fold (row); every fold
        # carries exactly its doublet (group-total assignment)
        assert res.weights.shape == (3, NMAG)
        for row in range(res.weights.shape[0]):
            w = np.sort(res.weights[row])[::-1]
            assert w[0] > 1 - 1e-8
            assert w[1] > 1 - 1e-8
            assert w[2:].max() < 1e-8


def test_defect_bonds_keep_sum_rule_and_go_fractional():
    js = np.ones(NMAG)
    js[1] = 0.5
    K = np.array([0.2, 0.0, 0.0])
    unf = MagnonUnfolder(_eigendata(js, K), M)
    labels = np.array([[j, 0, 0] for j in range(3)])
    qfolds = np.mod(
        np.linalg.solve(M.T.astype(float), (K + labels).T.astype(float)).T, 1.0
    )
    res = unf.compute(qfolds)
    assert np.allclose(res.weights.sum(axis=0), 1.0, atol=1e-10)
    # a weakened bond mixes folds: some mode must leave the binary set
    w = res.weights
    frac = ((w > 0.05) & (w < 0.95)).any()
    assert frac


def test_resolve_degenerate_handles_cross_fold_groups():
    """K = 0: the 4-fold group spans two folds; assignment stays binary."""
    K = np.zeros(3)
    unf = MagnonUnfolder(_eigendata(np.ones(NMAG), K), M)
    labels = np.array([[j, 0, 0] for j in range(3)])
    qfolds = np.mod(
        np.linalg.solve(M.T.astype(float), (K + labels).T.astype(float)).T, 1.0
    )
    plain = unf.compute(qfolds)
    resolved = unf.compute(qfolds, resolve_degenerate=1e-9)
    assert np.allclose(plain.weights.sum(axis=0), 1.0, atol=1e-10)
    # group-total presentation: every fold carries its doublet exactly
    assert np.allclose(resolved.weights.sum(axis=1), 2.0, atol=1e-8)
    for row in range(resolved.weights.shape[0]):
        w = np.sort(resolved.weights[row])[::-1]
        assert w[:2].min() > 1 - 1e-8
        assert w[2:].max() < 1e-8


def test_fixture_roundtrip_weights():
    with np.load(DATA) as raw:
        modes = raw["modes"]
        energies = raw["energies"]
    data = MagnonEigenData(
        kpoints=np.zeros((1, 3)),
        energies=energies[None, :],
        wavefunctions=modes[None, :, :],
        positions=POSITIONS,
    )
    unf = MagnonUnfolder(data, M)
    q = np.array([[0.0, 0.0, 0.0], [1 / 3, 0.0, 0.0], [2 / 3, 0.0, 0.0]])
    res = unf.compute(q, resolve_degenerate=1e-9)
    assert np.allclose(res.weights.sum(axis=1), 2.0, atol=1e-8)
    for row in range(res.weights.shape[0]):
        assert (res.weights[row] > 1 - 1e-8).sum() == 2


def test_positions_must_form_a_supercell_of_the_primitive_basis():
    # each primitive sublattice must appear det(M) = 3 times; a two-site
    # set with distinct tau_prim cannot be a 3x supercell of anything
    bad = MagnonEigenData(
        kpoints=np.zeros((1, 3)),
        energies=np.zeros((1, 2)),
        wavefunctions=np.zeros((1, 2, 4)),
        positions=np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="supercell of the"):
        MagnonUnfolder(bad, M)


def test_resolve_degenerate_rejects_negative_tolerance():
    unf = MagnonUnfolder(_eigendata(np.ones(NMAG), np.zeros(3)), M)
    with pytest.raises(ValueError, match="resolve_degenerate"):
        unf.compute(np.zeros((1, 3)), resolve_degenerate=-1.0)


def test_plot_smoke():
    unf = MagnonUnfolder(_eigendata(np.ones(NMAG), np.zeros(3)), M)
    q = np.array([[0.0, 0.0, 0.0], [1 / 3, 0.0, 0.0]])
    res = unf.compute(q)
    ax = res.plot(ylabel="Energy (meV)")
    assert ax is not None


def test_nonsymmetric_unfold_matrix():
    """Non-symmetric M: positions transform with M, momenta with M.T.

    A shear in the dummy y/z directions leaves the chain physics
    untouched but exercises the asymmetric branch (regression guard for
    the positions @ M.T transpose bug).
    """
    Mshear = np.array([[3, 0, 0], [0, 1, 1], [0, 0, 1]])  # det 3, non-symmetric
    K = np.array([0.14, 0.0, 0.0])
    unf = MagnonUnfolder(_eigendata(np.ones(NMAG), K), Mshear)
    labels = np.array([[j, 0, 0] for j in range(3)])
    qfolds = np.mod(
        np.linalg.solve(Mshear.T.astype(float), (K + labels).T.astype(float)).T, 1.0
    )
    plain = unf.compute(qfolds)
    assert np.allclose(plain.weights.sum(axis=0), 1.0, atol=1e-10)
    res = unf.compute(qfolds, resolve_degenerate=1e-9)
    for row in range(res.weights.shape[0]):
        w = np.sort(res.weights[row])[::-1]
        assert w[:2].min() > 1 - 1e-8
        assert w[2:].max() < 1e-8
