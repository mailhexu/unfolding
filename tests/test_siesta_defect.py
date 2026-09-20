"""Story-010 real displaced Si supercell fixture checks."""
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data", "si_example_defect_kgrid")


def _unfolder():
    pytest.importorskip("HamiltonIO")
    from HamiltonIO.siesta.sisl_wrapper import SislParser

    class RListParser(SislParser):
        def read_Rlist(self, geom=None):
            return self.ham.lattice.sc_off

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = RListParser(os.path.join(ROOT, "tests", "data", "si_example", "si_prim.fdf")).get_model()
    sc = RListParser(os.path.join(DATA, "si_defect_kgrid.fdf")).get_model()
    B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    rm = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, B, tol_r=0.1,
        orb_counts_sc=[4] * 8, orb_counts_prim=[4, 4]
    )
    return LCAOUnfolder(HamiltonIOModel(sc), rm)


def test_real_defect_fixture_is_multishell():
    u = _unfolder()
    assert len(u._model.SR) == 125


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
