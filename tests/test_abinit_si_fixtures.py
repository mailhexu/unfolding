"""Story-019 real ABINIT Si and Si7P planewave fixture seals.

Generated on nic6 with ``abinit/dev@d5380e0cb-gcc`` (ABINIT 10.9.0),
serial login-node runs, ``iomode 3``, ``istwfk 1``, ``chkprim 0``,
``ecut 12 Ha``, ``nband 24``, and ``tolvrs 1e-10``. The primitive
reference explicitly samples Gamma plus the three fcc X folds of the
8-atom conventional-cubic supercell. The pseudo-dojo NC-SR PBE psp8
records were normalized by dropping the ``Begin PSPCODE8`` marker required
by this ABINIT reader:

* Si: md5 ``5585c83a67fb0d1a382bb2440d0c41cc``
* P: md5 ``9d5d14a249222bf69d06c5ae4322bdf1``

For pristine Si, the first 21 SC bands are below the deliberately
headroom-limited edge. Their binary-sector residual is 5.63e-5. The
matched-sampling folded-spectrum constant shifts are -2.57 mHa at Gamma
and -2.04 mHa at X, with a maximum residual of 1.14 mHa. The Si7P donor
window is the three states within 0.3 eV of the Fermi level; their maximum
unfolded sector weight is 0.368, while the -6.8 eV host triplet exceeds
0.991.
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "abinit_si"
MATRIX = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int)
FOLD_KPOINTS = np.array(
    [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
)
PRIMITIVE = DATA / "si_primitive_matchedo_WFK.nc"
PRISTINE = DATA / "si8_gammao_WFK.nc"
DOPED = DATA / "si7p_gammao_WFK.nc"
DOPED_PATH = DATA / "si7p_gamma_x_patho_DS2_WFK.nc"
FIXTURES = (PRIMITIVE, PRISTINE, DOPED, DOPED_PATH)
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in FIXTURES),
    reason="ABINIT Si/Si7P WFK fixtures are not present",
)


def _reader_and_unfolder():
    pytest.importorskip("netCDF4")
    from unfolding.abinit_unfold import HARTREE_TO_EV, read_wfk
    from unfolding.pw_unfolder import PWUnfolder

    return HARTREE_TO_EV, read_wfk, PWUnfolder


def _max_norm_residual(data):
    return max(
        np.abs((np.abs(coefficients) ** 2).sum(axis=(2, 3)) - 1.0).max()
        for coefficients in data.coefficients
    )


@pytest.mark.parametrize("path", (PRIMITIVE, PRISTINE, DOPED, DOPED_PATH))
def test_real_wfk_coefficients_obey_parseval(path):
    _, read_wfk, _ = _reader_and_unfolder()

    data = read_wfk(path)

    assert _max_norm_residual(data) < 2e-13


def test_pristine_si_core_bands_have_binary_fold_weights():
    _, read_wfk, PWUnfolder = _reader_and_unfolder()
    data = read_wfk(PRISTINE)
    result = PWUnfolder(data, MATRIX).compute(FOLD_KPOINTS)

    assert np.abs(result.weights.sum(axis=0) - 1.0).max() < 2e-13
    core_weights = result.weights[:, :21]
    binary_residual = np.minimum(np.abs(core_weights), np.abs(core_weights - 1.0)).max()
    assert binary_residual < 1e-4


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
    donor = np.abs(energies) < 0.3
    host = (energies > -6.9) & (energies < -6.75)
    assert donor.sum() == 3
    assert host.sum() == 3
    assert max_weights[donor].max() < 0.4
    assert max_weights[host].min() > 0.99


def test_si7p_path_maps_primitive_gamma_x_to_stored_supercell_points():
    _, read_wfk, PWUnfolder = _reader_and_unfolder()
    kpoints = np.array([[0.0, t / 2.0, t / 2.0] for t in (0.0, 0.25, 0.5, 0.75, 1.0)])
    result = PWUnfolder(read_wfk(DOPED_PATH), MATRIX).compute(kpoints)

    expected = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert np.allclose(result.sc_kpoints, expected)
    assert result.weights.min() >= -1e-12
    assert result.weights.max() <= 1.0 + 1e-12
