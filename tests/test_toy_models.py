"""Toy LCAO model: closed-form generalized eigenvalues, Hermiticity, S > 0."""
import numpy as np
import pytest
from scipy.linalg import eigh

from unfolding.toy_models import closed_form_bands, toy_lcao_model


def test_real_space_matrices_satisfy_hermitian_pairing():
    # H(R) itself need not be Hermitian; the pair condition H(-R) = H(R)^dagger
    # is what makes H(k) = sum_R H(R) e^{2pi i k R} Hermitian.
    m = toy_lcao_model()
    for R in (1, -1):
        assert np.allclose(m["H"][-R], m["H"][R].conj().T)
        assert np.allclose(m["S"][-R], m["S"][R].conj().T)
    assert np.allclose(m["H"][0], m["H"][0].conj().T)
    assert np.allclose(m["S"][0], m["S"][0].conj().T)
    assert set(m["H"]) == {0, 1, -1}  # nearest-neighbour range


def test_k_space_hermitian_and_s_positive_definite():
    m = toy_lcao_model()
    rng = np.random.default_rng(42)
    for k in rng.uniform(0, 1, size=(20,)):
        H = m["H_of_k"](k)
        S = m["S_of_k"](k)
        assert np.allclose(H, H.conj().T, atol=1e-14)
        assert np.allclose(S, S.conj().T, atol=1e-14)
        assert np.linalg.eigvalsh(S).min() > 0.1  # well conditioned


def test_bands_match_closed_form():
    m = toy_lcao_model()
    rng = np.random.default_rng(7)
    for k in rng.uniform(0, 1, size=(25,)):
        numeric = eigh(m["H_of_k"](k), m["S_of_k"](k), eigvals_only=True)
        analytic = np.sort(closed_form_bands(m, k))
        assert np.allclose(numeric, analytic, atol=1e-12), (k, numeric, analytic)


def test_band_extrema_sanity():
    # finite nearest-neighbour chain: bands bounded by onsite +- |hopping|
    m = toy_lcao_model()
    ks = np.linspace(0, 1, 101)
    bands = np.array([np.sort(closed_form_bands(m, k)) for k in ks])
    assert bands.min() > -2.0 and bands.max() < 2.0
    assert bands[:, 0].min() < -1.0 and bands[:, 1].max() > 0.5  # dispersion exists
