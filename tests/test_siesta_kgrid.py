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
    from HamiltonIO.siesta.sisl_wrapper import SislParser

    class RListParser(SislParser):
        def read_Rlist(self, geom=None):
            return self.ham.lattice.sc_off

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = RListParser(os.path.join(ROOT, "tests", "data", "si_example", "si_prim.fdf")).get_model()
    sc = RListParser(os.path.join(DATA, "si_sc_kgrid.fdf")).get_model()
    B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    rm = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, B, orb_counts_sc=[4] * 8, orb_counts_prim=[4, 4]
    )
    return LCAOUnfolder(HamiltonIOModel(sc), rm)


def test_real_kgrid_has_multishell_hsx():
    """The real SIESTA k-grid archive contains the expected 5^3 R shell set."""
    u = _unfolder()
    assert len(u._model.SR) == 125
    assert u._n_orb_sc == 32 and u._n_orb_prim == 8


def test_real_kgrid_ideal_generic_weights():
    """The standard ideal weight is binary for pristine Si at generic k."""
    u = _unfolder()
    k = np.array([[0.13, 0.27, 0.41]])
    result = u.compute(k, method="ideal")
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
