"""Behavioral tests for the backend-free planewave unfolding core (story 017)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from unfolding.pw_unfolder import PWEigenData, PWUnfolder

DATA = Path(__file__).parent / "data" / "pw_synth" / "pw_synth_fixture.npz"


def _fixture_data():
    with np.load(DATA) as raw:
        M = raw["M"]
        K = raw["K"]
        gvecs = raw["gvecs"]
        cg = raw["cg"]
        w_ref = raw["W_ref"]
        reps = raw["reps"]
    # one stored SC k; scalar channel and one spinor component
    coeff = cg[None, :, None, :]
    eig = np.arange(cg.shape[0], dtype=float)[None, :]
    return (
        PWEigenData(
            kpoints=np.array([K]),
            gvecs=(gvecs,),
            coefficients=(coeff,),
            eigenvalues=(eig,),
        ),
        M,
        reps,
        w_ref,
    )


def test_core_symbols_are_exported_from_package():
    import unfolding

    assert unfolding.PWEigenData is PWEigenData
    assert unfolding.PWUnfolder is PWUnfolder



def _single_k_data(K, gvecs, coeff, eig):
    return PWEigenData(
        kpoints=np.array([K], dtype=float),
        gvecs=(np.asarray(gvecs, dtype=int),),
        coefficients=(np.asarray(coeff, dtype=complex),),
        eigenvalues=(np.asarray(eig, dtype=float),),
    )


def test_weights_match_sympy_fixture_and_parseval():
    data, M, folds, expected = _fixture_data()

    result = PWUnfolder(data, M).compute(folds)

    # Rows are the primitive fold images supplied to compute; transpose the
    # fixture's (band, sector) reference into the consumer result shape.
    assert np.allclose(result.weights, expected.T, atol=1e-12)
    assert np.allclose(result.weights.sum(axis=0), 1.0, atol=1e-12)
    assert np.allclose(result.fold_kpoints, folds)
    assert np.allclose(result.sc_kpoints, 0.0)


def test_forward_folded_state_has_binary_weight():
    # Independent forward map: choose the second fold kappa, map primitive
    # G_p=0 to its supercell G_s, then ask the core to bin it back.
    M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    kappa = np.array([0.5, 0.5, 0.0])
    g_sc = np.rint(M.T @ kappa).astype(int)
    data = _single_k_data(
        K=(0.0, 0.0, 0.0),
        gvecs=[g_sc],
        coeff=np.array([[[[1.0 + 0.0j]]]]),
        eig=np.array([[0.0]]),
    )
    folds = PWUnfolder.fold_kpoints(M)

    result = PWUnfolder(data, M).compute(folds)
    target = np.argmin(np.linalg.norm(folds - kappa, axis=1))
    assert np.allclose(result.weights[:, 0], np.eye(len(folds))[target])


def test_spinor_components_are_summed():
    data, M, folds, _ = _fixture_data()
    scalar = PWUnfolder(data, M).compute(folds)
    coeff = data.coefficients[0]
    spinor = np.concatenate(
        [coeff / np.sqrt(2.0), 1j * coeff / np.sqrt(2.0)], axis=2
    )
    spinor_data = PWEigenData(
        kpoints=data.kpoints,
        gvecs=data.gvecs,
        coefficients=(spinor,),
        eigenvalues=data.eigenvalues,
    )

    got = PWUnfolder(spinor_data, M).compute(folds)
    assert np.allclose(got.weights, scalar.weights, atol=1e-12)


def test_row_convention_with_nonsymmetric_supercell_matrix():
    M = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    k_prim = np.array([0.2, 0.3, 0.0])
    K_sc = np.mod(k_prim @ M.T, 1.0)
    data = _single_k_data(
        K=K_sc,
        gvecs=[[0, 0, 0]],
        coeff=np.array([[[[1.0 + 0.0j]]]]),
        eig=np.array([[0.0]]),
    )

    result = PWUnfolder(data, M).compute([k_prim])
    assert np.allclose(result.sc_kpoints[0], K_sc)
    assert np.allclose(result.weights, 1.0)


def test_missing_supercell_kpoint_names_target_and_nearest_point():
    data, M, _, _ = _fixture_data()

    with pytest.raises(ValueError, match=r"target SC k-point.*nearest stored"):
        PWUnfolder(data, M).compute([[0.123, 0.0, 0.0]])


def test_average_degenerate_collapses_groups():
    M = np.eye(3, dtype=int)
    data = _single_k_data(
        K=(0.0, 0.0, 0.0),
        gvecs=[[0, 0, 0]],
        coeff=np.array([[[[1.0 + 0.0j]], [[1.0 + 0.0j]], [[1.0 + 0.0j]]]]),
        eig=np.array([[0.0, 1e-8, 1.0]]),
    )
    grouped = PWUnfolder(data, M).compute([[0.0, 0.0, 0.0]]).average_degenerate(1e-6)

    assert grouped.eigenvalues.shape == (1, 2)
    assert np.allclose(grouped.eigenvalues[0], [5e-9, 1.0])
    assert np.allclose(grouped.weights[0], [2.0, 1.0])


def test_variable_npw_is_preserved_per_stored_kpoint():
    data = PWEigenData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        gvecs=(np.array([[0, 0, 0]]), np.array([[0, 0, 0], [1, 0, 0]])),
        coefficients=(
            np.array([[[[1.0 + 0.0j]]]]),
            np.array([[[[1.0 / np.sqrt(2.0), 1.0j / np.sqrt(2.0)]]]]),
        ),
        eigenvalues=(np.array([[0.0]]), np.array([[1.0]])),
    )

    result = PWUnfolder(data, np.eye(3, dtype=int)).compute(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]
    )

    assert np.allclose(result.weights, 1.0)
    assert np.allclose(result.eigenvalues[:, 0], [0.0, 1.0])


def test_result_plot_wraps_weighted_band_plot():
    data, M, folds, _ = _fixture_data()
    result = PWUnfolder(data, M).compute(folds)

    axis = result.plot(xqpts=np.arange(len(folds)), ylabel="Energy")

    assert axis.get_ylabel() == "Energy"
    assert axis.collections


def test_fold_points_follow_row_convention_for_nonsymmetric_matrix():
    M = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    folds = PWUnfolder.fold_kpoints(M)
    kappa = np.array([0.5, 0.0, 0.0])
    g_sc = np.rint(M @ kappa).astype(int)
    data = _single_k_data(
        K=(0.0, 0.0, 0.0),
        gvecs=[g_sc],
        coeff=np.array([[[[1.0 + 0.0j]]]]),
        eig=np.array([[0.0]]),
    )

    result = PWUnfolder(data, M).compute(folds)
    target = np.argmin(np.linalg.norm(folds - kappa, axis=1))
    assert np.allclose(result.weights[:, 0], np.eye(len(folds))[target])


def test_records_own_input_buffers_and_reject_inconsistent_kpoints():
    kpoints = np.array([[0.0, 0.0, 0.0]])
    gvecs = np.array([[0, 0, 0]])
    coeff = np.array([[[[1.0 + 0.0j]]]])
    eig = np.array([[0.0]])
    data = _single_k_data(kpoints[0], gvecs, coeff, eig)
    kpoints[:] = 0.25
    coeff[:] = 0.0

    result = PWUnfolder(data, np.eye(3, dtype=int)).compute([[0.0, 0.0, 0.0]])
    assert np.allclose(result.weights, 1.0)
    assert not data.kpoints.flags.writeable
    assert not result.weights.flags.writeable

    with pytest.raises(ValueError, match="at least one stored k-point"):
        PWEigenData(np.empty((0, 3)), (), (), ())
    with pytest.raises(ValueError, match="expected"):
        PWEigenData(
            np.zeros((2, 3)),
            (gvecs, gvecs),
            (coeff, np.concatenate([coeff, coeff], axis=0)),
            (eig, np.array([[0.0], [0.0]])),
        )


def test_pristine_degenerate_group_weight_is_binary_after_gauge_average():
    M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    folds = PWUnfolder.fold_kpoints(M)
    k0, k1 = folds[0], folds[1]
    gvecs = np.array([np.rint(M @ k0).astype(int), np.rint(M @ k1).astype(int)])
    rng = np.random.default_rng(19)
    unitary, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    coeff = unitary[None, :, None, :]
    data = _single_k_data(
        K=(0.0, 0.0, 0.0),
        gvecs=gvecs,
        coeff=coeff,
        eig=np.array([[0.0, 0.0]]),
    )

    per_band = PWUnfolder(data, M).compute(folds)
    grouped = per_band.average_degenerate(1e-8)

    assert np.all((per_band.weights[:2] > 1e-12) & (per_band.weights[:2] < 1.0 - 1e-12))
    assert np.allclose(grouped.weights[:2, 0], 1.0, atol=1e-12)
    assert np.allclose(grouped.weights[2:, 0], 0.0, atol=1e-12)

def test_pristine_degenerate_group_weight_is_binary_after_resolve_degenerate():
    """Eigen-assignment restores 0/1 branch weights under mixing gauges."""
    M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    folds = PWUnfolder.fold_kpoints(M)
    gvecs = np.array([np.rint(M @ folds[0]).astype(int),
                      np.rint(M @ folds[1]).astype(int)])
    rng = np.random.default_rng(23)
    unitary, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    coeff = unitary[None, :, None, :]
    data = _single_k_data(
        K=(0.0, 0.0, 0.0),
        gvecs=gvecs,
        coeff=coeff,
        eig=np.array([[0.0, 0.0]]),
    )

    plain = PWUnfolder(data, M).compute(folds)
    resolved = PWUnfolder(data, M).compute(folds, resolve_degenerate=1e-8)

    # arbitrary gauge: the stored mixture splits the weights fractionally
    assert np.all((plain.weights[:2] > 1e-12) & (plain.weights[:2] < 1.0 - 1e-12))
    # eigen-assignment: one branch carries the full sector weight again
    binary_residual = np.minimum(
        np.abs(resolved.weights[:2]), np.abs(resolved.weights[:2] - 1.0)
    ).max()
    assert binary_residual < 1e-12
    assert resolved.weights[2:].max() < 1e-12


def test_resolve_degenerate_rejects_negative_tolerance():
    M = np.eye(3, dtype=int)
    data = _single_k_data(
        K=(0.0, 0.0, 0.0),
        gvecs=np.zeros((1, 3), dtype=int),
        coeff=np.ones((1, 1, 1, 1), dtype=complex),
        eig=np.array([[0.0]]),
    )
    with pytest.raises(ValueError, match="resolve_degenerate"):
        PWUnfolder(data, M).compute(np.zeros((1, 3)), resolve_degenerate=-1.0)
