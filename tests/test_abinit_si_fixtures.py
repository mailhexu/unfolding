"""Story-019 real ABINIT Si and Si7P planewave fixture seals.

The committed WFK fixtures are small. The nic6 generator can additionally
produce the 300-point documentation path, but that large artifact is kept
outside Git and the test follows whichever path WFK is available.
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "abinit_si"
MATRIX = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int)
FOLD_KPOINTS = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
PRIMITIVE = DATA / "si_primitive_matchedo_WFK.nc"
PRISTINE = DATA / "si8_gammao_WFK.nc"
DOPED = DATA / "si7p_gammao_WFK.nc"
DOPED_PATH = DATA / "si7p_gamma_x_patho_DS2_WFK.nc"
FIXTURES = (PRIMITIVE, PRISTINE, DOPED, DOPED_PATH)
pytestmark = pytest.mark.skipif(not all(path.is_file() for path in FIXTURES), reason="ABINIT Si/Si7P WFK fixtures are not present")


def _reader_and_unfolder():
    pytest.importorskip("netCDF4")
    from unfolding.abinit_unfold import HARTREE_TO_EV, read_wfk
    from unfolding.pw_unfolder import PWUnfolder
    return HARTREE_TO_EV, read_wfk, PWUnfolder


def _max_norm_residual(data):
    return max(np.abs((np.abs(coefficients) ** 2).sum(axis=(2, 3)) - 1.0).max() for coefficients in data.coefficients)


@pytest.mark.parametrize("path", FIXTURES)
def test_real_wfk_coefficients_obey_parseval(path):
    _, read_wfk, _ = _reader_and_unfolder()
    assert _max_norm_residual(read_wfk(path)) < 2e-13


def test_pristine_si_core_bands_have_binary_fold_weights():
    _, read_wfk, PWUnfolder = _reader_and_unfolder()
    result = PWUnfolder(read_wfk(PRISTINE), MATRIX).compute(FOLD_KPOINTS)
    assert np.abs(result.weights.sum(axis=0) - 1.0).max() < 2e-13
    core = result.weights[:, :21]
    assert np.minimum(np.abs(core), np.abs(core - 1.0)).max() < 1e-4


def test_pristine_si_folded_spectrum_matches_matched_primitive_reference():
    _, read_wfk, PWUnfolder = _reader_and_unfolder()
    primitive = read_wfk(PRIMITIVE)
    result = PWUnfolder(read_wfk(PRISTINE), MATRIX).compute(FOLD_KPOINTS)
    residuals = []
    for ifold, fold in enumerate(FOLD_KPOINTS):
        distances = np.linalg.norm(((primitive.kpoints - fold + 0.5) % 1.0) - 0.5, axis=1)
        ikpt = int(np.argmin(distances))
        selected = result.weights[ifold] > 0.99
        sc_energies = np.sort(result.eigenvalues[ifold, selected])
        reference = np.sort(primitive.eigenvalues[ikpt][0])[: len(sc_energies)]
        assert 3 <= len(sc_energies) <= 6
        shift = np.median(sc_energies - reference)
        residuals.append(np.abs(sc_energies - reference - shift).max())
    assert max(residuals) < 1.5e-3


def test_si7p_donor_window_is_fractional_while_host_triplet_is_nearly_binary():
    hartree_to_ev, read_wfk, PWUnfolder = _reader_and_unfolder()
    data = read_wfk(DOPED)
    result = PWUnfolder(data, MATRIX).compute(FOLD_KPOINTS)
    energies = (data.eigenvalues[0][0] - data.fermi_energy) * hartree_to_ev
    max_weights = result.weights.max(axis=0)
    assert np.abs(result.weights.sum(axis=0) - 1.0).max() < 2e-13
    # one partially-filled donor band sits essentially at the Fermi
    # level (33 electrons: 16.5 bands); the host valence triplet stays
    # nearly binary a third of an eV below it
    donor = np.abs(energies) < 0.1
    host = (energies > -0.4) & (energies < -0.25)
    assert donor.sum() == 1
    assert host.sum() == 3
    assert max_weights[donor].max() < 0.4
    assert max_weights[host].min() > 0.95


def test_si7p_path_maps_primitive_gamma_x_to_stored_supercell_points():
    _, read_wfk, PWUnfolder = _reader_and_unfolder()
    data = read_wfk(DOPED_PATH)
    sc_kpoints = np.mod(data.kpoints, 1.0)
    kpoints = np.mod(sc_kpoints @ np.linalg.inv(MATRIX.T), 1.0)
    result = PWUnfolder(data, MATRIX).compute(kpoints)
    # WFK k-points may carry components outside [0, 1) (e.g. W stored as
    # (0.5, 1, 0)); they are equivalent modulo the SC reciprocal lattice.
    assert np.allclose(np.mod(result.sc_kpoints, 1.0), sc_kpoints)
    assert result.weights.min() >= -1e-12
    assert result.weights.max() <= 1.0 + 1e-12
