"""Slab/ribbon magnon unfolding regression tests.

Builds a synthetic two-sided FM chain with the TB2J edit API
(``make_supercell`` + ``open_boundary``) and seals that the unfolding
machinery handles supercells with an open direction: ``M = diag(N, 1, 1)``
flows through ``MagnonUnfolder`` unchanged, weights select the correct
folded branch, and the open direction carries no dispersion.

Skipped when the TB2J-supercell worktree (which provides ``open_boundary``)
is unavailable. magnon3's cell-origin gauge is the sealed convention here;
see the TB2J record fix-magnon3-jq-bloch-phases-for-supercells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

TB2J_SUPERCELL = Path("/home/hexu/projects/TB2J_dev/TB2J-supercell")
if not (TB2J_SUPERCELL / "TB2J" / "io_exchange" / "edit.py").exists():
    pytest.skip("TB2J-supercell worktree (open_boundary) not available", allow_module_level=True)
# force the supercell worktree (provides open_boundary) even if another
# TB2J (installed or from a sibling test) was already imported
for _name in [n for n in sys.modules if n == "TB2J" or n.startswith("TB2J.")]:
    del sys.modules[_name]
sys.path.insert(0, str(TB2J_SUPERCELL))

import TB2J  # noqa: E402

assert "TB2J-supercell" in Path(TB2J.__file__).resolve().parts, TB2J.__file__

from ase import Atoms  # noqa: E402
from TB2J.io_exchange.edit import load, make_supercell, open_boundary, save  # noqa: E402
from TB2J.io_exchange.io_exchange import SpinIO, gen_distance_dict  # noqa: E402
from TB2J.magnon.magnon3 import Magnon  # noqa: E402

from unfolding.tb2j_unfold import magnon_eigendata_from_tb2j  # noqa: E402
from unfolding.magnon_unfolder import MagnonUnfolder  # noqa: E402

M = np.diag([6, 1, 1]).astype(int)


def _chain_spinio():
    """Two-sided NN+2NN FM chain (real TB2J pickles store both directions)."""
    atoms = Atoms("Fe", cell=np.eye(3), scaled_positions=[[0, 0, 0]], pbc=True)
    terms = {
        ((1, 0, 0), 0, 0): 2.0,
        ((2, 0, 0), 0, 0): 0.5,
        ((-1, 0, 0), 0, 0): 2.0,
        ((-2, 0, 0), 0, 0): 0.5,
    }
    return SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 3.0]]),
        charges=np.array([6.0]),
        index_spin=[0],
        distance_dict=gen_distance_dict([0], atoms, sorted({R for (R, _, _) in terms})),
        exchange_Jdict=terms,
        dmi_ddict=None,
        Jani_dict=None,
        sia_tensor=None,
    )


def _magnon(result_dir):
    m = Magnon.from_TB2J_results(path=str(result_dir), Jiso=True)
    m.set_reference(
        Q=(0, 0, 0),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([1.0, 0.0, 0.0]),
        magmoms=np.asarray(m.magmom, dtype=float),
    )
    return m


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ribbon")
    prim_dir = tmp / "prim"
    save(_chain_spinio(), prim_dir)
    sc_dir = tmp / "sc6"
    save(make_supercell(load(str(prim_dir)), [6, 1, 1]), sc_dir)
    ribbon_dir = tmp / "ribbon"
    ribbon = make_supercell(load(str(prim_dir)), [6, 1, 1])
    open_boundary(ribbon, ["a"])
    save(ribbon, ribbon_dir)
    return _magnon(prim_dir), _magnon(sc_dir), _magnon(ribbon_dir)


QPTS = np.array([[0.0, 0, 0], [0.15, 0, 0], [0.25, 0, 0], [0.3, 0.1, 0]])


def _fold_sum_rule(magnon, ka=0.6):
    """Per-mode sum rule at one stored K: the weights of each mode,
    summed over all six primitive folds q_j = ((ka + j)/6, 0, 0) sharing
    that stored K, equal 1 (unresolved weights; sealed invariant
    sum_q W_q(nu) = |psi_nu|^2 = 1)."""
    qpts = np.array([[(ka + j) / 6.0, 0.0, 0.0] for j in range(6)])
    kpts = np.mod(qpts @ M.T, 1.0)
    assert np.allclose(kpts[:, 0], ka)  # same stored K for every fold
    data = magnon_eigendata_from_tb2j(magnon, kpts)
    res = MagnonUnfolder(data, M).compute(qpts)
    assert np.allclose(res.weights.sum(axis=0), 1.0, atol=1e-8)
    return res


def test_replicated_supercell_unfolds_to_primitive_band(models):
    prim, sc, _ = models
    _fold_sum_rule(sc)
    _, prim_bands, _ = prim.get_magnon_bands(kpoints=QPTS)
    data = magnon_eigendata_from_tb2j(sc, np.mod(QPTS @ M.T, 1.0))
    res = MagnonUnfolder(data, M).compute(QPTS, resolve_degenerate=1e-9)
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-8)  # mode totals
    # the primitive band at q must appear as a mode with (near-)unit weight
    for iq, q in enumerate(QPTS):
        match = np.abs(res.energies[iq] - prim_bands[iq, 0]) < 1e-8
        assert match.any(), (iq, res.energies[iq], prim_bands[iq, 0])
        assert res.weights[iq][match].max() > 0.99


def test_open_ribbon_unfolds_without_dispersion_along_open_axis(models):
    _, _, ribbon = models
    # the ribbon is open along a: identical bands at k_a = 0 and 1/2
    _, b0, _ = ribbon.get_magnon_bands(kpoints=np.array([[0.0, 0, 0]]))
    _, bh, _ = ribbon.get_magnon_bands(kpoints=np.array([[0.5, 0, 0]]))
    assert np.allclose(b0, bh, atol=1e-10)
    # per-mode fold sum rule still holds on the open model
    _fold_sum_rule(ribbon, ka=0.6)
    _fold_sum_rule(ribbon, ka=0.2)
