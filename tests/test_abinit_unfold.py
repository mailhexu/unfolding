"""Synthetic ETSF-WFK reader and one-call adapter contracts (story 018)."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
netCDF4 = pytest.importorskip("netCDF4")

from unfolding.abinit_unfold import HARTREE_TO_EV, WFKData, read_wfk, unfold_abinit


def _write_wfk(
    path: Path,
    *,
    istwfk=1,
    codvsn="10.9.0",
    nspin=1,
    nspinor=1,
    ncomplex=2,
    eigenvalues=(-0.20, 0.10),
    fermi_energy=0.05,
):
    """Write the subset of ETSF Nanoquanta WFK schema the reader consumes."""
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("number_of_kpoints", 1)
        nc.createDimension("number_of_reduced_dimensions", 3)
        nc.createDimension("max_number_of_coefficients", 4)
        nc.createDimension("number_of_spins", nspin)
        nc.createDimension("max_number_of_states", 2)
        nc.createDimension("number_of_spinor_components", nspinor)
        nc.createDimension("real_or_complex_coefficients", ncomplex)
        nc.createDimension("number_of_vectors", 3)
        nc.createDimension("number_of_cartesian_directions", 3)
        nc.createDimension("codvsnlen", 8)

        nc.createVariable("reduced_coordinates_of_kpoints", "f8", ("number_of_kpoints", "number_of_reduced_dimensions"))[:] = [[0.0, 0.0, 0.0]]
        nc.createVariable("number_of_coefficients", "i4", ("number_of_kpoints",))[:] = [2]
        nc.createVariable("number_of_states", "i4", ("number_of_spins", "number_of_kpoints"))[:] = 2
        nc.createVariable("istwfk", "i4", ("number_of_kpoints",))[:] = [istwfk]
        nc.createVariable("primitive_vectors", "f8", ("number_of_vectors", "number_of_cartesian_directions"))[:] = [
            [1.0, 0.0, 0.0], [0.25, 2.0, 0.0], [0.0, 0.0, 3.0]
        ]
        if fermi_energy is not None:
            nc.createVariable("fermi_energy", "f8")[:] = fermi_energy
        cod = nc.createVariable("codvsn", "S1", ("codvsnlen",))
        encoded = np.zeros(8, dtype="S1")
        encoded[: len(codvsn)] = np.array(list(codvsn), dtype="S1")
        cod[:] = encoded

        gvec = nc.createVariable(
            "reduced_coordinates_of_plane_waves", "i4",
            ("number_of_kpoints", "max_number_of_coefficients", "number_of_reduced_dimensions"),
        )
        gvec[:] = [[[0, 0, 0], [1, 0, 0], [99, 99, 99], [99, 99, 99]]]
        eig = nc.createVariable(
            "eigenvalues", "f8",
            ("number_of_spins", "number_of_kpoints", "max_number_of_states"),
        )
        eig[:] = np.array([[eigenvalues]] * nspin)
        occ = nc.createVariable(
            "occupations", "f8",
            ("number_of_spins", "number_of_kpoints", "max_number_of_states"),
        )
        occ[:] = 0.0
        coeff = nc.createVariable(
            "coefficients_of_wavefunctions", "f8",
            (
                "number_of_spins", "number_of_kpoints", "max_number_of_states",
                "number_of_spinor_components", "max_number_of_coefficients",
                "real_or_complex_coefficients",
            ),
        )
        raw = np.zeros((nspin, 1, 2, nspinor, 4, ncomplex))
        raw[:, 0, 0, 0, 0, 0] = 1.0
        raw[:, 0, 1, 0, 1, 0] = 1.0
        if nspinor > 1 and ncomplex == 2:
            raw[:, 0, 0, 1, 1, :] = [0.25, 0.5]
        raw[:, 0, :, :, 2:, :] = 17.0  # must be trimmed by npw=2
        coeff[:] = raw


class _NoFullReadVariable:
    def __init__(self, variable):
        self._variable = variable

    @property
    def shape(self):
        return self._variable.shape

    def __getitem__(self, key):
        if isinstance(key, slice) and key == slice(None):
            raise AssertionError("reader materialized a padded WFK variable")
        return self._variable[key]


class _StreamingDataset:
    _protected = {
        "coefficients_of_wavefunctions",
        "eigenvalues",
        "reduced_coordinates_of_plane_waves",
    }

    def __init__(self, *args):
        self._dataset = netCDF4.Dataset(*args)

    @property
    def variables(self):
        return {
            name: _NoFullReadVariable(variable) if name in self._protected else variable
            for name, variable in self._dataset.variables.items()
        }

    def close(self):
        self._dataset.close()


def test_read_wfk_trims_padding_converts_complex_and_keeps_metadata(tmp_path):
    path = tmp_path / "synthetic_WFK.nc"
    _write_wfk(path)

    data = read_wfk(path)

    assert isinstance(data, WFKData)
    assert data.codvsn == "10.9.0"
    assert data.nspinor == 1
    assert np.array_equal(data.istwfk, [1])
    assert np.allclose(data.rprimd, [[1, 0, 0], [0.25, 2, 0], [0, 0, 3]])
    assert data.gvecs[0].shape == (2, 3)
    assert data.coefficients[0].shape == (1, 2, 1, 2)
    assert np.iscomplexobj(data.coefficients[0])
    assert np.allclose(data.coefficients[0][0, 0, 0], [1.0, 0.0])
    assert data.fermi_energy == pytest.approx(0.05)


def test_read_wfk_streams_padded_arrays_per_kpoint(tmp_path, monkeypatch):
    path = tmp_path / "streamed_WFK.nc"
    _write_wfk(path)
    module = importlib.import_module("unfolding.abinit_unfold")
    monkeypatch.setattr(module, "_require_netcdf4", lambda: _StreamingDataset)

    data = read_wfk(path)

    assert data.gvecs[0].shape == (2, 3)
    assert data.coefficients[0].shape == (1, 2, 1, 2)
    assert data.eigenvalues[0].shape == (1, 2)


def test_read_wfk_rejects_real_coefficient_storage(tmp_path):
    path = tmp_path / "real_WFK.nc"
    _write_wfk(path, ncomplex=1)

    with pytest.raises(ValueError, match="real coefficient storage"):
        read_wfk(path)


def test_read_wfk_distinguishes_missing_file_from_bad_format(tmp_path):
    with pytest.raises(FileNotFoundError, match="WFK not found"):
        read_wfk(tmp_path / "missing_WFK.nc")


def test_read_wfk_preserves_synthetic_spinor_components(tmp_path):
    path = tmp_path / "spinor_WFK.nc"
    _write_wfk(path, nspinor=2)

    data = read_wfk(path)

    assert data.nspinor == 2
    assert data.coefficients[0].shape == (1, 2, 2, 2)
    assert data.coefficients[0][0, 0, 1, 1] == pytest.approx(0.25 + 0.5j)


def test_read_wfk_rejects_half_sphere_storage(tmp_path):
    path = tmp_path / "half_WFK.nc"
    _write_wfk(path, istwfk=2)

    with pytest.raises(ValueError, match=r"istwfk 1"):
        read_wfk(path)


def test_read_wfk_rejects_non_netcdf_with_iomode_fix(tmp_path):
    path = tmp_path / "fortran_WFK"
    path.write_bytes(b"not a netcdf WFK")

    with pytest.raises(ValueError, match=r"iomode 3"):
        read_wfk(path)


def test_read_wfk_rejects_unknown_major_version(tmp_path):
    path = tmp_path / "new_WFK.nc"
    _write_wfk(path, codvsn="11.0.0")

    with pytest.raises(ValueError, match=r"supported ABINIT WFK major"):
        read_wfk(path)


def test_unfold_abinit_smoke_uses_fermi_header_and_returns_axes(tmp_path):
    path = tmp_path / "synthetic_WFK.nc"
    _write_wfk(path)

    axis = unfold_abinit(
        path,
        np.eye(3, dtype=int),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        xqpts=[0.0, 1.0],
        fermi_shift=True,
        ylabel="Energy (eV)",
    )

    assert axis.get_ylabel() == "Energy (eV)"
    assert axis.collections
    # plot_band_weight adds a faint gray line per band; its y-data expose
    # the consumer-visible Hartree->eV + Fermi-header conversion.
    got = sorted(
        float(line.get_ydata()[0])
        for line in axis.lines
        if line.get_color() == "gray" and len(line.get_ydata())
    )
    expected = sorted((np.array([-0.20, 0.10]) - 0.05) * HARTREE_TO_EV)
    assert np.allclose(got[:2], expected)


def test_unfold_abinit_requires_fermi_header_for_default_shift(tmp_path):
    path = tmp_path / "no_fermi_WFK.nc"
    _write_wfk(path, fermi_energy=None)
    kpts = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="no fermi_energy header"):
        unfold_abinit(path, np.eye(3, dtype=int), kpts)

    axis = unfold_abinit(path, np.eye(3, dtype=int), kpts, fermi_shift=False)
    assert axis.collections


def test_unfold_abinit_average_degenerate_tolerance_is_ev(tmp_path):
    path = tmp_path / "near_degenerate_WFK.nc"
    _write_wfk(path, eigenvalues=(-0.20, -0.19995))
    kpts = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    narrow = unfold_abinit(path, np.eye(3, dtype=int), kpts, average_degenerate=1e-4)
    wide = unfold_abinit(path, np.eye(3, dtype=int), kpts, average_degenerate=2e-3)

    assert len(narrow.collections) == 2
    assert len(wide.collections) == 1


def test_unfold_abinit_accepts_preparsed_wfkdata(tmp_path):
    path = tmp_path / "synthetic_WFK.nc"
    _write_wfk(path, nspin=2)
    data = read_wfk(path)

    axis = unfold_abinit(
        data=data,
        unfold_sc_mat=np.eye(3, dtype=int),
        kpts=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        spin=1,
    )

    assert axis.collections


@pytest.mark.skipif(
    not os.environ.get("UNFOLDING_WFK_SMOKE"),
    reason="set UNFOLDING_WFK_SMOKE to an ABINIT iomode=3, istwfk=1 WFK path",
)
def test_real_wfk_smoke():
    for path in os.environ["UNFOLDING_WFK_SMOKE"].split(os.pathsep):
        data = read_wfk(path)

        assert data.codvsn.split(".")[0] in {"9", "10"}
        assert np.all(data.istwfk == 1)
        for coeff in data.coefficients:
            assert np.allclose((np.abs(coeff) ** 2).sum(axis=(2, 3)), 1.0, atol=1e-10)
