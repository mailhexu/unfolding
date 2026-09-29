"""Regression tests for the built-in Wannier90 readers.

Covers real-output formats the synthetic fixtures don't exercise: the
"written on ..." comment line Wannier90 >= 2.0 prepends to *_hr.dat, and
WF centres read from the .wout Final State when the *_centres.xyz is
stale (different num_wann -- e.g. the shipped SrTiO3 dataset).
"""
import numpy as np
import pytest

from unfolding.wannier_unfold import (Wannier90Model, read_wannier90_hr,
                                      read_wannier90_wout_centres)

HR_BODY = """\
    1.00000000000000E+000    1.00000000000000E+000
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
HR_V1 = "2\n" + HR_BODY

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
    assert Rs[0] == pytest.approx([-1.0, 0.0, 0.0])   # sorted R order
    assert H[0][0, 0] == pytest.approx(0.5)           # R = (-1,0,0) block
    assert H[0][0, 1] == pytest.approx(0.1)
    assert H[1][0, 0] == pytest.approx(1.0)           # R = (0,0,0) block


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
    expected = np.mod(np.array([[0.1, 0.0, 0.0], [2.0, 2.0, 2.0]])
                      @ np.linalg.inv(np.eye(3) * 4.0), 1.0)
    assert model._orb == pytest.approx(expected, abs=1e-12)
