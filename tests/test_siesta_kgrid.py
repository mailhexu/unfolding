"""Story-010 real SIESTA k-grid fixture validation.

The fixture was generated on nic6 with SIESTA
``dev@a23c0d21a-gcc`` from a 2x2x2 Monkhorst-Pack run. It is kept
separate from the Gamma-only fixture because its HSX contains the full
R-shell set needed by generic-k ideal unfolding.
"""
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data", "si_example_kgrid")


def _unfolder():
    pytest.importorskip("HamiltonIO")
    from siesta_helpers import make_si_unfolder

    return make_si_unfolder(os.path.join(DATA, "si_sc_kgrid.fdf"))


def _sc_reduced_kpoints(u, wfsx):
    physical_k, _, _ = wfsx.read_info()
    cell = np.asarray(u._model.atoms.cell)
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    return physical_k @ np.linalg.inv(reciprocal)


def test_real_kgrid_has_multishell_hsx():
    """The real SIESTA k-grid archive contains the expected 5^3 R shell set."""
    u = _unfolder()
    assert len(u._model.SR) == 125
    assert u._n_orb_sc == 32 and u._n_orb_prim == 8


def test_real_kgrid_ideal_generic_weights():
    """The standard ideal weight is binary for pristine Si at generic k."""
    u = _unfolder()
    result = u.compute(np.array([[0.13, 0.27, 0.41]]), method="ideal")
    weights = result.weights[0]
    assert weights.min() > -1e-8
    assert weights.max() < 1.0 + 1e-8
    assert np.abs(weights - np.round(weights)).max() < 1e-8
    assert abs(weights.sum() - 8.0) < 1e-8


def test_real_kgrid_gamma_ring_and_ideal_agree():
    """The ring and ideal definitions coincide at Gamma on real data."""
    u = _unfolder()
    k = np.zeros((1, 3))
    ring = u.compute(k, method="ring").weights
    ideal = u.compute(k, method="ideal").weights
    assert np.abs(ring - ideal).max() < 1e-8


def test_real_kgrid_bands_match_primitive_siesta():
    """The eight ideal-weight-one SC bands match the independently parsed
    primitive k-grid bands within the converged-SCF tolerance."""
    sisl = pytest.importorskip("sisl")
    from scipy.linalg import eigh
    from siesta_helpers import read_si_model
    from unfolding.lcao_unfolder import HamiltonIOModel

    u = _unfolder()
    wfsx = sisl.get_sile(os.path.join(DATA, "si_sc_kgrid.selected.WFSX"))
    sc_kpoints = _sc_reduced_kpoints(u, wfsx)
    prim = HamiltonIOModel(read_si_model(os.path.join(DATA, "si_prim_kgrid.fdf")))
    inv_b_t = np.linalg.inv(u._scmat.T)

    for K in sc_kpoints:
        primitive_k = K @ inv_b_t
        result = u.compute(primitive_k[None, :], method="ideal")
        selected = result.weights[0] > 0.5
        assert selected.sum() == 8
        H, S = prim.hs_and_eigen(primitive_k)[:2]
        eps_prim = eigh(H, S, eigvals_only=True)
        # Separate primitive/SC SCF runs use equivalent but not bit-identical
        # density histories; the observed 3.7 meV maximum is the run tolerance.
        assert np.abs(
            np.sort(result.eigenvalues[0, selected]) - np.sort(eps_prim)
        ).max() < 5e-3


def test_real_kgrid_wfsx_eigenvalues_match_hsx_at_selected_kpoints():
    """Every selected real WFSX k-point matches HSX eigenvalues.

    The independent WFSX eigenvector-weight oracle is sealed at Gamma
    by ``test_wfsx_gamma_weights_oracle``. At non-Gamma k-points this
    test deliberately limits the cross-check to eigenvalues because the
    WFSX physical-k gauge and the primitive fractional-k ideal gauge
    require a separate convention adjudication.
    """
    sisl = pytest.importorskip("sisl")
    from scipy.linalg import eigh
    from siesta_helpers import read_fermi_energy

    u = _unfolder()
    fermi = read_fermi_energy(os.path.join(DATA, "si_sc_kgrid.EIG"))
    wfsx = sisl.get_sile(os.path.join(DATA, "si_sc_kgrid.selected.WFSX"))
    states = list(wfsx.yield_eigenstate())
    sc_kpoints = _sc_reduced_kpoints(u, wfsx)
    assert len(states) == len(sc_kpoints) == 4

    for K, state in zip(sc_kpoints, states):
        out = u._model.hs_and_eigen(K)
        eps = eigh(out[0], out[1], eigvals_only=True)
        eps_wfsx = np.asarray(state.c, dtype=float) - fermi
        assert np.abs(np.sort(eps) - np.sort(eps_wfsx)).max() < 1e-4
