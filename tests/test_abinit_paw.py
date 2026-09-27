"""Real JTH-PAW WFK unfolding: metric norm, pristine and Si:P defect."""
from pathlib import Path

import numpy as np
import pytest

from HamiltonIO.abinit import read_paw_wfk
from unfolding.abinit_paw import unfold_abinit_paw
from unfolding.abinit_unfold import unfold_abinit

DATA = Path(__file__).resolve().parent / "data/abinit_paw"
XML = {symbol: DATA / f"{symbol}.xml" for symbol in ("Si", "P")}
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
PRIM = DATA / "si_primitive_foldso_WFK.nc"
SI = DATA / "si8_gammao_WFK.nc"
DOPED = DATA / "si7p_gammao_WFK.nc"
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in [*XML.values(), PRIM, SI, DOPED]),
    reason="ABINIT PAW WFK and XML fixtures absent",
)


def test_reader_rejects_nonpaw_wfk():
    norm_conserving = DATA.parent / "abinit_si/si8_gammao_WFK.nc"
    with pytest.raises(ValueError, match="not a PAW calculation"):
        read_paw_wfk(norm_conserving, XML)


def test_pristine_reference_folds_have_paw_unit_norm_and_binary_weights():
    data = read_paw_wfk(SI, XML)
    assert data.wavefunctions.usepaw == 1
    assert data.species == ("Si",) * 8
    with pytest.raises(ValueError, match="requires unfold_abinit_paw"):
        unfold_abinit(data=data.wavefunctions, unfold_sc_mat=M, kpts=[[0, 0, 0]])
    result = unfold_abinit_paw(data, PRIM, XML, M)
    assert np.max(np.abs(result.norm_residuals)) < 2e-6
    np.testing.assert_allclose(result.weights.sum(axis=0), 1, atol=2e-6)
    resolved = unfold_abinit_paw(data, PRIM, XML, M, resolve_degenerate=1e-5)
    assert np.max(np.minimum(np.abs(resolved.weights), np.abs(1-resolved.weights))) < 2e-6
    # Gauge rotation reorders states within each degenerate group independently;
    # raw per-band sums above, not resolved column labels, obey Parseval.

def test_dopant_changes_paw_weights_relative_to_pseudo_cosets():
    result = unfold_abinit_paw(DOPED, PRIM, XML, M)
    assert np.max(np.abs(result.norm_residuals)) < 2e-6
    assert np.max(np.abs(result.weights - result.pseudo_weights)) > 0.005
    donor = result.weights[:, 16]
    assert 0.04 < donor.min() < 0.1
    assert 0.2 < donor.max() < 0.4
