"""Story 010: multi-shell synthetic k-grid fixture through the production
SIESTA parse path.

The fixture (tests/data/si_synth_kgrid) is a Si-diamond tight-binding
model written in the SIESTA HSX binary format with a full 3x3x3 shell
set on the 8-atom conventional supercell. It seals the multi-shell
parse -> relabel -> weight chain end to end, including the generic-k
ideal (Popescu-Zunger/Lee) weight, which a single-shell archive cannot
validate. Regenerate with tests/data/make_synth_kgrid_fixture.py.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "tests", "data", "si_synth_kgrid")


def _unfolder():
    hio = pytest.importorskip("HamiltonIO")
    from HamiltonIO.siesta.sisl_wrapper import SislParser

    class TorusSislParser(SislParser):
        """Rlist from the Hamiltonian file's actual supercell translations."""

        def read_Rlist(self, geom=None):
            return self.ham.lattice.sc_off

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = TorusSislParser(os.path.join(DATA, "si_prim.fdf")).get_model()
    sc = TorusSislParser(os.path.join(DATA, "si_sc.fdf")).get_model()
    B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    rm = RelabelMap.from_atoms(sc.atoms, prim.atoms, B,
                               orb_counts_sc=[1] * 8, orb_counts_prim=[1, 1])
    return LCAOUnfolder(HamiltonIOModel(sc), rm)


@pytest.fixture(scope="module")
def unf():
    return _unfolder()


def test_multi_shell_data_parsed(unf):
    """The SC archive carries the full 3x3x3 shell set (27 translations)."""
    assert len(unf._model.SR) == 27
    assert unf._n_orb_sc == 8 and unf._n_orb_prim == 2


def test_gamma_ring_groups(unf):
    """Pristine Gamma: the 2 folded primitive bands carry weight 1."""
    res = unf.compute(np.zeros((1, 3)))
    w = res.weights[0]
    groups = np.round(w, 6)
    assert (groups == 1).sum() == 2
    assert (groups == 0).sum() == 6


def _group_sums(eps, w, tol=1e-8):
    """Sum weights within exactly-degenerate eigenvalue groups (per-band
    weights are gauge-dependent inside a degenerate manifold; the group
    sum is the invariant)."""
    order = np.argsort(eps)
    out = []
    i = 0
    while i < len(eps):
        j = i
        while j < len(eps) and abs(eps[order[j]] - eps[order[i]]) < tol:
            j += 1
        out.append(float(np.sum(w[order[i:j]])))
        i = j
    return sorted(out)


def test_ideal_gamma_groups(unf):
    """Ideal weights at Gamma: the 2 folded primitive bands carry 1."""
    res = unf.compute(np.zeros((1, 3)), method="ideal")
    w = res.weights[0]
    assert np.abs(w - np.round(w)).max() < 1e-8, w
    assert (np.round(w) == 1).sum() == 2


def test_ideal_binary_at_generic_k(unf):
    """Multi-shell ideal weights: 0/1 at non-degenerate generic momenta."""
    res = unf.compute(np.array([[0.13, 0.27, 0.41]]), method="ideal")
    w = res.weights[0]
    assert np.abs(w - np.round(w)).max() < 1e-8, w
    assert abs(w.sum() - 2) < 1e-8


def test_ring_and_ideal_agree_at_gamma(unf):
    """Ring == ideal at Gamma for the multi-shell archive."""
    w_ring = unf.compute(np.zeros((1, 3)), method="ring").weights
    w_ideal = unf.compute(np.zeros((1, 3)), method="ideal").weights
    assert np.abs(w_ring - w_ideal).max() < 1e-8
