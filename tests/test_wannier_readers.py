"""Regression tests for the built-in Wannier90 readers.

Covers real-output formats the synthetic fixtures don't exercise: the
"written on ..." comment line Wannier90 >= 2.0 prepends to *_hr.dat, and
WF centres read from the .wout Final State when the *_centres.xyz is
stale (different num_wann -- e.g. the shipped SrTiO3 dataset).
"""
from pathlib import Path
import numpy as np
import pytest
from unfolding.wannier_unfold import (Wannier90Model, WannierUnfolder,
                                      read_wannier90_hr,
                                      read_wannier90_wout_centres)

HR_BODY = """\
    2    1
  -1   0   0   1   1     0.5000000000     0.0000000000
  -1   0   0   1   2     0.1000000000     0.0000000000
  -1   0   0   2   1     0.1000000000     0.0000000000
  -1   0   0   2   2     0.5000000000     0.0000000000
   0   0   0   1   1     1.0000000000     0.0000000000
   0   0   0   1   2     0.2000000000     0.0000000000
   0   0   0   2   1     0.2000000000     0.0000000000
   0   0   0   2   2     1.0000000000     0.0000000000
"""
HR_V2 = " written on  2May2018 at 13:37:20 \n          2\n          2\n" + HR_BODY
HR_V1 = "2\n2\n" + HR_BODY

WIN = """\
begin unit_cell_cart
 4.0 0.0 0.0
 0.0 4.0 0.0
 0.0 0.0 4.0
end unit_cell_cart

begin atoms_cart
Ti 0.0 0.0 0.0
Ti 2.0 2.0 2.0
end atoms_cart
"""

# minimisation-log centres first (must be ignored), Final State last
WOUT = """\
 earlier cycles report the same lines
  WF centre and spread    1  (  9.900000,  9.900000,  9.900000 )     9.0
  WF centre and spread    2  (  9.900000,  9.900000,  9.900000 )     9.0
 Final State
  WF centre and spread    1  (  0.100000,  0.000000,  0.000000 )     0.5
  WF centre and spread    2  (  2.000000,  2.000000,  2.000000 )     0.5
"""


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "seed.win").write_text(WIN)
    (tmp_path / "seed.wout").write_text(WOUT)
    return tmp_path


def test_hr_with_written_on_header(model_dir):
    path = model_dir / "seed_hr.dat"
    path.write_text(HR_V2)
    Rs, H = read_wannier90_hr(str(path))
    assert Rs.shape == (2, 3)
    assert H.shape == (2, 2, 2)
    assert Rs[0] == pytest.approx([-1.0, 0.0, 0.0])
    assert H[0][0, 0] == pytest.approx(0.25)  # WS degeneracy 2
    assert H[0][0, 1] == pytest.approx(0.05)
    assert H[1][0, 0] == pytest.approx(1.0)   # WS degeneracy 1


def test_hr_legacy_format_without_header(model_dir):
    path = model_dir / "seed_hr.dat"
    path.write_text(HR_V1)
    Rs, H = read_wannier90_hr(str(path))
    assert Rs.shape == (2, 3)
    assert H.shape == (2, 2, 2)


def test_hr_rejects_files_without_num_wann(model_dir):
    path = model_dir / "seed_hr.dat"
    path.write_text(" written by something else \n    1.0    1.0\n")
    with pytest.raises(ValueError, match="num_wann"):
        read_wannier90_hr(str(path))


def test_wout_centres_final_state(model_dir):
    cart = read_wannier90_wout_centres(str(model_dir / "seed.wout"))
    assert cart.shape == (2, 3)
    assert cart[0] == pytest.approx([0.1, 0.0, 0.0])
    assert cart[1] == pytest.approx([2.0, 2.0, 2.0])


def test_stale_centres_xyz_falls_back_to_wout(model_dir):
    # 3 centres for 2 orbitals: the xyz is from a different run and must
    # not be used; the wout Final State supplies the positions instead.
    (model_dir / "seed_centres.xyz").write_text(
        "3\nstale\nX 0.0 0.0 0.0\nX 1.0 1.0 1.0\nX 2.0 2.0 2.0\n")
    (model_dir / "seed_hr.dat").write_text(HR_V2)
    model = Wannier90Model(str(model_dir), "seed")
    expected = np.array([[0.1, 0.0, 0.0], [2.0, 2.0, 2.0]])
    expected = expected @ np.linalg.inv(np.eye(3) * 4.0)
    assert model._orb == pytest.approx(expected, abs=1e-12)


def test_wannier_centres_keep_hr_cell_image(model_dir):
    (model_dir / "seed_hr.dat").write_text(HR_V2)
    # A center just below zero must remain just below zero: wrapping it to
    # one shifts the Wannier Bloch phase relative to the hr.dat R vectors.
    cart = np.array([[-1e-8, 0, 0], [2.0, 2.0, 2.0]])
    xyz = "2\ncentres\n" + "".join(
        f"X {x:.12f} {y:.12f} {z:.12f}\n" for x, y, z in cart)
    (model_dir / "seed_centres.xyz").write_text(xyz)
    model = Wannier90Model(str(model_dir), "seed")
    expected = cart @ np.linalg.inv(np.eye(3) * 4.0)
    np.testing.assert_allclose(model._orb, expected, atol=1e-14)

def test_eigenvectors_are_band_major_and_solve_hamiltonian(model_dir):
    (model_dir / "seed_hr.dat").write_text(HR_V2)
    model = Wannier90Model(str(model_dir), "seed")
    energies, vectors = model.solve_all([[0.0, 0.0, 0.0]], eig_vectors=True)
    h_gamma = np.sum(model.H, axis=0)
    for band in range(model.norb):
        np.testing.assert_allclose(h_gamma @ vectors[band, 0],
                                   energies[band, 0] * vectors[band, 0],
                                   atol=1e-12)


def test_real_sto_pristine_gamma_binary_and_vacancy_fractional():
    base = Path(__file__).resolve().parents[1] / "examples" / "wannier_STO"
    scmat = [[1, -1, 0], [1, 1, 0], [0, 0, 2]]
    labels = (["pz", "px", "py"] * 12
              + ["dz2", "dxy", "dyz", "dx2", "dxz"] * 4)
    def gamma_weights(directory):
        model = Wannier90Model(base / directory, "wannier90", scmat=scmat)
        return WannierUnfolder(model, labels, scmat).unfold([[0, 0, 0]])[0]
    pristine = gamma_weights("data_nodefect")
    np.testing.assert_allclose(pristine, np.round(pristine), atol=1e-7)
    assert np.any(pristine > 0.9) and np.any(pristine < 0.1)
    vacancy = gamma_weights("data")
    assert np.any((vacancy > 0.1) & (vacancy < 0.9))
def test_real_sto_pristine_path_weights_resolve_folded_gauges():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.dft.kpoints import bandpath

    base = Path(__file__).resolve().parents[1] / "examples" / "wannier_STO"
    scmat = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]])
    labels = (["pz", "px", "py"] * 12
              + ["dz2", "dxy", "dyz", "dx2", "dxz"] * 4)
    vertices = [[0, 0, 0], [.5, 0, 0], [.5, .5, 0], [0, 0, 0], [.5, .5, .5]]

    def path_weights(directory):
        model = Wannier90Model(base / directory, "wannier90", scmat=scmat)
        unfolder = WannierUnfolder(model, labels, scmat)
        path = bandpath([np.dot(k, scmat.T) for k in vertices],
                        unfolder.cell, npoints=200)
        raw = unfolder.unfold(path.kpts)
        ax = unfolder.plot_unfolded_band(
            kvectors=vertices, knames=["Γ", "X", "M", "Γ", "R"],
            npoints=200,
            resolve_degenerate=0.1 if directory == "data_nodefect" else None)
        plotted = unfolder.last_result.weights
        plt.close(ax.figure)
        return raw, plotted

    pristine_raw, pristine = path_weights("data_nodefect")
    assert np.max(np.minimum(pristine_raw, 1.0 - pristine_raw)) > 0.49
    np.testing.assert_allclose(pristine, np.round(pristine), atol=1e-4)
    vacancy_raw, vacancy = path_weights("data")
    assert np.any((vacancy_raw > 0.1) & (vacancy_raw < 0.9))
    np.testing.assert_array_equal(vacancy, vacancy_raw)
def test_real_sto_vacancy_matches_historical_independent_reference():
    base = Path(__file__).resolve().parents[1] / "examples" / "wannier_STO" / "data"
    scmat = [[1, -1, 0], [1, 1, 0], [0, 0, 2]]
    labels = (["pz", "px", "py"] * 12
              + ["dz2", "dxy", "dyz", "dx2", "dxz"] * 4)
    kpoints = np.array([[0.0, 0.0, 0.0], [0.125, 0.25, 0.25],
                        [0.25, 0.125, 0.375]])
    # Frozen from the historical PythTB w90 reader (min_hopping_norm=1e-4)
    # and the same Unfolder translation maps; sorted bands at each k.
    reference_energy = np.array([6.030001405745244, 5.773361770572301,
                                 6.739842132363256])
    reference_weight = np.array([0.42479819700941823, 0.4699316044154523,
                                 0.4912436207537373])

    model = Wannier90Model(base, "wannier90", scmat=scmat)
    unfolder = WannierUnfolder(model, labels, scmat)
    weights = unfolder.unfold(kpoints)
    energies = unfolder.evals
    for ik, band in enumerate((39, 37, 39)):
        assert energies[band, ik] == pytest.approx(reference_energy[ik], abs=7e-3)
        assert weights[ik, band] == pytest.approx(reference_weight[ik], abs=7e-3)
