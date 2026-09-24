"""TB2J-adapter contract tests (story 023); skipped without TB2J."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

pytest.importorskip("TB2J", reason="TB2J extra required")

sys.path.insert(0, "/home/hexu/projects/TB2J_dev/TB2J")  # dev checkout if not installed

from TB2J.magnon.magnon3 import Magnon  # noqa: E402

from unfolding.tb2j_unfold import magnon_eigendata_from_tb2j, unfold_tb2j  # noqa: E402

DATA = Path(__file__).parent / "data" / "tb2j_srmmo3" / "TB2J_results"
# 10-atom sqrt2^3 G-AFM cell (fcc setting) = 2x the 5-atom pseudo-cubic
# primitive cell: rows of M are the TB2J cell axes in pseudo-cubic units.
M_AFM = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _magnon():
    m = Magnon.from_TB2J_results(path=str(DATA))
    m.set_reference(
        Q=(0, 0, 0),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([1.0, 0.0, 0.0]),
    )
    return m


def test_eigendata_contract():
    m = _magnon()
    qpts = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0.25, 0.25]])
    data = magnon_eigendata_from_tb2j(m, np.mod(qpts @ M_AFM.T, 1.0))
    assert data.wavefunctions.shape == (3, 2, 4)  # 2 Mn -> 2 modes x 2N
    # canonical amplitudes: Euclidean-normalized rows
    assert np.allclose(np.linalg.norm(data.wavefunctions, axis=2), 1.0, atol=1e-10)
    # Goldstone: lowest mode at Gamma within the anisotropy gap scale
    assert data.energies[0].min() < 5e-3


def test_unfold_weights_binary_and_sum_rule():
    m = _magnon()
    qpts = np.array([[0.0, 0, 0], [0.2, 0, 0], [0.4, 0.1, 0]])
    data = magnon_eigendata_from_tb2j(m, np.mod(qpts @ M_AFM.T, 1.0))
    from unfolding import MagnonUnfolder

    unf = MagnonUnfolder(data, M_AFM)
    plain = unf.compute(qpts)
    # the G-AFM dispersion is Q-periodic, so the two folds sharing a
    # stored K are always degenerate: plain weights show the mixed
    # gauge (0.5/0.5) with the exact per-mode sum rule
    assert np.allclose(plain.weights.sum(axis=1), 1.0, atol=1e-8)
    # the group-total presentation is binary: one mode per fold
    res = unf.compute(qpts, resolve_degenerate=1e-5)
    binary = np.minimum(np.abs(res.weights), np.abs(res.weights - 1.0)).max()
    assert binary < 1e-8
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-8)


def test_one_call_adapter_plots():
    ax = unfold_tb2j(
        str(DATA),
        M_AFM,
        np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]]),
        knames=[r"$\Gamma$", "X", "X"],
        xqpts=[0.0, 0.5, 1.0],
        Xqpts=[0.0, 0.5, 1.0],
        spin_conf=[[0, 0, 2.81], [0, 0, -2.81]],
    )
    assert ax is not None
