"""Story-010 real displaced Si supercell fixture checks."""
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data", "si_example_defect_kgrid")


def _unfolder():
    pytest.importorskip("HamiltonIO")
    from siesta_helpers import make_si_unfolder

    return make_si_unfolder(
        os.path.join(DATA, "si_defect_kgrid.fdf"), tol_r=0.1
    )


def _fermi_energy(path):
    with open(path) as fh:
        return float(fh.readline())


def test_real_defect_fixture_is_multishell():
    u = _unfolder()
    assert len(u._model.SR) == 125


def test_real_defect_wfsx_eigenvalues_match_hsx():
    """The displaced-cell WFSX Gamma state energies match HSX."""
    sisl = pytest.importorskip("sisl")
    from scipy.linalg import eigh

    u = _unfolder()
    out = u._model.hs_and_eigen(np.zeros(3))
    eps = eigh(out[0], out[1], eigvals_only=True)
    state = sisl.get_sile(
        os.path.join(DATA, "si_defect_kgrid.selected.WFSX")
    ).read_eigenstate()
    eps_wfsx = np.asarray(state.c, dtype=float) - _fermi_energy(
        os.path.join(DATA, "si_defect_kgrid.EIG")
    )
    assert np.abs(np.sort(eps) - np.sort(eps_wfsx)).max() < 1e-4


def test_real_defect_ideal_weights_are_finite():
    """The ideal generic-k definition remains finite for the displaced SC.

    The exact ring projection is asserted on commensurate grids; its
    generic-k sector Gram is not Hermitian for a position-dependent defect.
    """
    u = _unfolder()
    result = u.compute(np.array([[0.13, 0.27, 0.41]]), method="ideal")
    weights = result.weights[0]
    assert np.isfinite(weights).all()
    assert weights.min() > -1e-7
