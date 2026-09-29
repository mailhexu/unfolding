"""Story-039: ``[output] data`` JSON datasets on the routes (TOML ≡ flags ≡ API)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))  # reuse the story-033 parity harness

from test_cli_config import (  # noqa: E402
    M_AFM,
    M_SI,
    PRIM_POSCAR,
    SIESTA_FDF,
    TB2J_RESULTS,
    WFK,
    _run_cli,
    _write_toml,
)


def test_all_routes_expose_data_flag():
    from unfolding.cli import build_parser
    from unfolding.config import ROUTES

    subs = build_parser()._subparsers._group_actions[0].choices
    assert set(subs) == set(ROUTES)
    for name, sub in subs.items():
        flags = {a.option_strings[0] for a in sub._actions}
        assert "--data" in flags, f"route {name} lacks --data"




def test_config_toml_accepts_output_data(tmp_path: Path):
    from unfolding.config import load_config

    prim = tmp_path / "prim.vasp"
    prim.write_text(PRIM_POSCAR)
    cfg_file = _write_toml(
        tmp_path / "siesta.toml",
        {
            "route": "siesta",
            "input": {"fdf": str(SIESTA_FDF)},
            "structure": {
                "primitive": str(prim),
                "supercell_matrix": M_SI,
            },
            "path": {"special_points": "GX", "npts": 4},
            "output": {"output": "fig.png", "data": "bands.json"},
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.data == "bands.json"



def test_siesta_route_writes_dataset(tmp_path: Path):
    pytest.importorskip("HamiltonIO")
    from ase.dft.kpoints import bandpath, get_special_points
    from ase.io import read as ase_read

    from unfolding import unfold_siesta

    prim = tmp_path / "prim.vasp"
    prim.write_text(PRIM_POSCAR)
    out_json = tmp_path / "siesta.json"
    out_png = tmp_path / "siesta.png"

    toml = _write_toml(
        tmp_path / "siesta.toml",
        {
            "route": "siesta",
            "input": {"fdf": str(SIESTA_FDF)},
            "structure": {"primitive": str(prim), "supercell_matrix": M_SI},
            "path": {"special_points": "GX", "npts": 8},
            "options": {"method": "ideal", "efermi": 0.25},
            "output": {"output": str(out_png), "data": str(out_json)},
        },
    )
    _run_cli(["--config", str(toml)])
    assert out_json.is_file() and out_png.is_file()

    from unfolding import load_dataset

    ds = load_dataset(out_json)
    assert ds.route == "siesta"
    assert ds.energy_unit == "eV"


    # parity with the direct adapter call
    cell = ase_read(prim).cell.array
    points = get_special_points(cell, eps=0.01)
    path = bandpath([points[c] for c in "GX"], cell, 8)
    _ax, res = unfold_siesta(
        fdf=str(SIESTA_FDF),
        prim_atoms=str(prim),
        unfold_sc_mat=np.asarray(M_SI),
        kpts=path.kpts,
        method="ideal",
        return_result=True,
    )
    np.testing.assert_allclose(ds.kpoints, res.kpoints, atol=1e-12)
    np.testing.assert_allclose(ds.eigenvalues, res.eigenvalues, atol=1e-10)
    np.testing.assert_allclose(ds.weights, res.weights, atol=1e-10)


def test_abinit_wfk_route_writes_dataset(tmp_path: Path):
    pytest.importorskip("netCDF4")
    pytest.importorskip("HamiltonIO")
    from HamiltonIO.abinit import read_wfk

    matrix = np.asarray(M_SI)
    data = read_wfk(WFK)
    kpts = np.mod(np.asarray(data.kpoints[:4]) @ np.linalg.inv(matrix.T), 1.0)
    del data

    out_json = tmp_path / "wfk.json"
    out_png = tmp_path / "wfk.png"
    toml = _write_toml(
        tmp_path / "wfk.toml",
        {
            "route": "abinit-wfk",
            "input": {"wfk": str(WFK)},
            "structure": {"supercell_matrix": M_SI},
            "path": {"kpoints": kpts.tolist()},
            "output": {"output": str(out_png), "data": str(out_json)},
        },
    )
    _run_cli(["--config", str(toml)])
    assert out_json.is_file()

    from unfolding import load_dataset

    ds = load_dataset(out_json)
    assert ds.route == "abinit-wfk"
    assert ds.fold_kpoints is not None and ds.sc_kpoints is not None
    assert ds.fold_kpoints.shape == ds.kpoints.shape
    np.testing.assert_allclose(ds.kpoints, kpts, atol=1e-12)
    # pristine-supercell-like sanity: weights live in [0, 1 + eps]
    assert np.nanmax(ds.weights) <= 1.0 + 1e-9


def test_magnon_route_writes_dataset_mev(tmp_path: Path):
    pytest.importorskip("TB2J")
    out_json = tmp_path / "magnon.json"
    out_png = tmp_path / "magnon.png"

    toml = _write_toml(
        tmp_path / "magnon.toml",
        {
            "route": "magnon",
            "input": {"results": str(TB2J_RESULTS)},
            "structure": {"supercell_matrix": M_AFM},
            "path": {"special_points": "GXM", "npts": 6},
            "output": {"output": str(out_png), "data": str(out_json)},
        },
    )
    _run_cli(["--config", str(toml)])
    assert out_json.is_file()

    from unfolding import load_dataset

    ds = load_dataset(out_json)
    assert ds.route == "magnon"
    assert ds.energy_unit == "meV"
    assert ds.eigenvalues.shape[0] == len(ds.kpoints)
    # magnon branches are non-negative to plotting accuracy
    assert ds.eigenvalues.min() > -1e-6


def test_no_data_field_no_dataset(tmp_path: Path):
    pytest.importorskip("HamiltonIO")
    prim = tmp_path / "prim.vasp"
    prim.write_text(PRIM_POSCAR)
    out_png = tmp_path / "siesta.png"
    toml = _write_toml(
        tmp_path / "siesta.toml",
        {
            "route": "siesta",
            "input": {"fdf": str(SIESTA_FDF)},
            "structure": {"primitive": str(prim), "supercell_matrix": M_SI},
            "path": {"special_points": "GX", "npts": 4},
            "options": {"method": "ideal"},
            "output": {"output": str(out_png)},
        },
    )
    _run_cli(["--config", str(toml)])
    assert out_png.is_file()
    assert list(tmp_path.glob("*.json")) == []
