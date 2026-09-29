"""Story-038: unfolded-band JSON dataset round trip (schema v1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from unfolding.dataset import SCHEMA, Dataset, load_dataset, save_dataset


class _Result:
    def __init__(self, *, spin=1, fold=True, nan=False):
        rng = np.random.default_rng(7)
        nk, nb = 5, 4
        self.kpoints = rng.normal(size=(nk, 3))
        eig = rng.normal(size=(nk, nb) if spin == 1 else (spin, nk, nb))
        self.eigenvalues = eig
        w = rng.random(eig.shape)
        if nan:
            w[0, 0] = np.nan
        self.weights = w
        if fold:
            self.fold_kpoints = rng.normal(size=(nk, 3))
            self.sc_kpoints = rng.normal(size=(nk, 3))


def test_roundtrip_collinear(tmp_path: Path):
    res = _Result()
    out = tmp_path / "ds.json"
    save_dataset(res, out, route="pw", fermi_energy=-0.25,
                 energy_reference="absolute")
    ds = load_dataset(out)
    assert ds.route == "pw"
    assert ds.fermi_energy == pytest.approx(-0.25)
    assert ds.spin_channels == 1
    np.testing.assert_allclose(ds.kpoints, res.kpoints)
    np.testing.assert_allclose(ds.eigenvalues, res.eigenvalues)
    np.testing.assert_allclose(ds.weights, res.weights)
    np.testing.assert_allclose(ds.fold_kpoints, res.fold_kpoints)
    np.testing.assert_allclose(ds.sc_kpoints, res.sc_kpoints)
    assert ds.energy_unit == "eV"


def test_roundtrip_spin_and_nan(tmp_path: Path):
    res = _Result(spin=2, nan=True)
    out = tmp_path / "ds.json"
    save_dataset(res, out, route="lcao")
    ds = load_dataset(out)
    assert ds.spin_channels == 2
    assert ds.eigenvalues.shape == (2, 5, 4)
    assert np.isnan(ds.weights[0, 0]).all()
    np.testing.assert_allclose(np.nan_to_num(ds.weights), np.nan_to_num(res.weights))


def test_missing_optional_keys_are_none(tmp_path: Path):
    res = _Result(fold=False)
    out = tmp_path / "ds.json"
    save_dataset(res, out, route="magnon")
    ds = load_dataset(out)
    assert ds.fold_kpoints is None and ds.sc_kpoints is None
    assert "fold_kpoints" not in json.loads(out.read_text())


def test_schema_field_and_unknown_schema(tmp_path: Path):
    out = tmp_path / "ds.json"
    save_dataset(_Result(), out, route="pw")
    doc = json.loads(out.read_text())
    assert doc["schema"] == SCHEMA
    doc["schema"] = "unfolding.dataset/999"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="schema"):
        load_dataset(bad)


def test_resave_byte_stable(tmp_path: Path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    kwargs = dict(route="pw", provenance={"unfold_sc_mat": [[1]]})
    save_dataset(_Result(), a, **kwargs)
    save_dataset(_Result(), b, **kwargs)
    assert a.read_bytes() == b.read_bytes()


def test_json_is_plain_lists_and_null_nan(tmp_path: Path):
    out = tmp_path / "ds.json"
    save_dataset(_Result(nan=True), out, route="pw")
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    doc = json.loads(text)
    assert isinstance(doc["kpoints"][0], list)
    assert doc["weights"][0][0] is None


def test_dataset_frozen_arrays():
    ds = Dataset.from_result(_Result(), route="pw")
    with pytest.raises(ValueError):
        ds.kpoints[0, 0] = 5.0


def test_energy_unit_roundtrip(tmp_path: Path):
    out = tmp_path / "mev.json"
    save_dataset(_Result(), out, route="magnon", energy_unit="meV")
    assert load_dataset(out).energy_unit == "meV"


def test_resave_parsed_dataset_preserves_metadata(tmp_path: Path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    save_dataset(_Result(), a, route="magnon", energy_unit="meV",
                 fermi_energy=-1.5, energy_reference="absolute",
                 provenance={"supercell_matrix": [[2]]})
    save_dataset(load_dataset(a), b)
    assert a.read_bytes() == b.read_bytes()


def test_infinite_values_rejected(tmp_path: Path):
    res = _Result()
    eig = res.eigenvalues.copy()
    eig[0, 0] = np.inf
    res.eigenvalues = eig
    with pytest.raises(ValueError, match="infinite"):
        save_dataset(res, tmp_path / "bad.json", route="pw")


def test_nan_not_allowed_in_energies_or_coordinates(tmp_path: Path):
    res = _Result()
    res.eigenvalues[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN values"):
        save_dataset(res, tmp_path / "bad.json", route="pw")


def test_coordinate_shapes_validated():
    with pytest.raises(ValueError, match="kpoints"):
        Dataset(route="pw", kpoints=np.zeros((2, 2)),
                eigenvalues=np.zeros((2, 3)), weights=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="fold_kpoints"):
        Dataset(route="pw", kpoints=np.zeros((2, 3)),
                eigenvalues=np.zeros((2, 3)), weights=np.zeros((2, 3)),
                fold_kpoints=np.zeros((3, 3)))


def test_save_requires_route_for_engine_results(tmp_path: Path):
    with pytest.raises(ValueError, match="route"):
        save_dataset(_Result(), tmp_path / "x.json")
