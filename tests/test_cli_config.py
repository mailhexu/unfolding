"""Story-033: config core + CLI framework parity (TOML ≡ flags ≡ Python API)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SIESTA_FDF = REPO / "tests" / "data" / "si_example" / "si_sc_pso.fdf"
PHONOPY_DIR = REPO / "examples" / "phonopy"
WFK = REPO / "tests" / "data" / "abinit_si" / "si7p_gamma_x_patho_DS2_WFK.nc"
DDB = REPO / "examples" / "Cu_fcc" / "out_DDB"
TB2J_RESULTS = REPO / "tests" / "data" / "tb2j_srmmo3" / "TB2J_results"

M_SI = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
M_CU_DDB = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]  # = inv([[0,1,1],[1,0,1],[1,1,0]]/2)
M_AFM = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

# abinit builds sometimes live outside PATH (dev checkouts); abipy's shell
# adapter resolves anaddb through PATH, so the DDB test prepends it.
ANADDB_SEARCH = ("/home/hexu/tmp/abinit_lucia/98_main",)


def _anaddb_dir():
    """PATH dir providing anaddb, or None when it is unavailable."""
    if shutil.which("anaddb"):
        return None  # already on PATH
    for d in ANADDB_SEARCH:
        if os.path.isfile(os.path.join(d, "anaddb")):
            return d
    return None


def _anaddb_missing():
    return _anaddb_dir() is None


def _write_toml(path, payload):
    lines = []

    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return repr(v)

    for key, value in payload.items():
        if key == "route":
            lines.append(f"route = {fmt(value)}")
            continue
        lines.append(f"[{key}]")
        for k, v in value.items():
            lines.append(f"{k} = {fmt(v)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _fingerprint(ax):
    """(x, y, weight) arrays extracted from a weight-coded band figure."""
    xs, ys, ws = [], [], []
    for coll in ax.collections:
        segs = np.asarray(coll.get_segments())
        if segs.size == 0:
            continue
        colors = np.asarray(coll.get_colors())
        xs.append(segs[..., 0].mean(axis=1))
        ys.append(segs[..., 1].mean(axis=1))
        ws.append(colors[:, 3])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    order = np.lexsort((w, y, x))
    return x[order], y[order], w[order]


def _assert_parity(axes):
    assert len(axes) == 3
    prints = [_fingerprint(ax) for ax in axes]
    for x, y, w in prints:
        assert len(x) > 0 and w.max() > 0.1  # non-trivial weight-coded figure
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for pa, pb in zip(prints[a], prints[b]):
            np.testing.assert_allclose(pa, pb, rtol=1e-10, atol=1e-12)


def _run_cli(argv):
    from unfolding.cli import main

    plt = pytest.importorskip("matplotlib.pyplot")
    rc = main(argv)
    fig = plt.gcf()
    ax = fig.axes[-1] if fig.axes else None
    plt.close("all")
    assert rc == 0
    return ax


# ---------------------------------------------------------------------------
# siesta
# ---------------------------------------------------------------------------

PRIM_POSCAR = """Si2 primitive fcc cell
1.0
0.0 2.715 2.715
2.715 0.0 2.715
2.715 2.715 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""


def test_siesta_parity(tmp_path):
    pytest.importorskip("HamiltonIO")
    from ase.dft.kpoints import bandpath, get_special_points
    from ase.io import read as ase_read

    from unfolding import unfold_siesta

    prim = tmp_path / "prim.vasp"
    prim.write_text(PRIM_POSCAR)
    out_toml = tmp_path / "siesta_toml.png"
    out_flags = tmp_path / "siesta_flags.png"

    toml = _write_toml(
        tmp_path / "siesta.toml",
        {
            "route": "siesta",
            "input": {"fdf": str(SIESTA_FDF)},
            "structure": {"primitive": str(prim), "supercell_matrix": M_SI},
            "path": {"special_points": "GXWGLX", "npts": 12},
            "options": {"method": "ideal"},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "siesta",
            "--fdf", str(SIESTA_FDF),
            "--primitive", str(prim),
            "--unfold-mat", *(str(v) for row in M_SI for v in row),
            "--special-points", "GXWGLX",
            "--npts", "12",
            "--method", "ideal",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # direct adapter call with the same independently built path
    cell = ase_read(prim).cell.array
    points = get_special_points(cell, eps=0.01)
    path = bandpath([points[c] for c in "GXWGLX"], cell, 12)
    x, X, _labels = path.get_linear_kpoint_axis()
    ax_api = unfold_siesta(
        fdf=str(SIESTA_FDF),
        prim_atoms=str(prim),
        unfold_sc_mat=np.asarray(M_SI),
        kpts=path.kpts,
        knames=list("GXWGLX"),
        xqpts=x,
        Xqpts=X,
        method="ideal",
    )
    _assert_parity([ax_toml, ax_flags, ax_api])


# ---------------------------------------------------------------------------
# phonopy
# ---------------------------------------------------------------------------

def test_phonopy_parity(tmp_path):
    pytest.importorskip("phonopy")
    from ase.dft.kpoints import bandpath, get_special_points
    from ase.io import read as ase_read

    from unfolding.phonopy_unfolder import phonopy_unfold

    fc = str(PHONOPY_DIR / "FORCE_CONSTANTS")
    sposcar = str(PHONOPY_DIR / "SPOSCAR")
    out_toml = tmp_path / "phonopy_toml.png"
    out_flags = tmp_path / "phonopy_flags.png"

    toml = _write_toml(
        tmp_path / "phonopy.toml",
        {
            "route": "phonopy",
            "input": {"force_constants": fc, "sposcar": sposcar},
            "structure": {"supercell_matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]},
            "path": {"special_points": "GXWGL", "npts": 10},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "phonopy",
            "--force-constants", fc,
            "--sposcar", sposcar,
            "--unfold-mat", "3", "0", "0", "0", "3", "0", "0", "0", "3",
            "--special-points", "GXWGL",
            "--npts", "10",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # direct adapter call; path cell = inv(M) @ SPOSCAR cell
    prim_cell = np.linalg.inv(np.diag([3.0, 3.0, 3.0])) @ ase_read(sposcar).cell.array
    points = get_special_points(prim_cell, eps=0.01)
    path = bandpath([points[c] for c in "GXWGL"], prim_cell, 10)
    x, X, _labels = path.get_linear_kpoint_axis()
    ax_api = phonopy_unfold(
        sc_mat=np.diag([1, 1, 1]),
        unfold_sc_mat=np.diag([3, 3, 3]),
        force_constants=fc,
        sposcar=sposcar,
        qpts=path.kpts,
        qnames=list("GXWGL"),
        xqpts=x,
        Xqpts=X,
    )
    _assert_parity([ax_toml, ax_flags, ax_api])


# ---------------------------------------------------------------------------
# abinit-wfk
# ---------------------------------------------------------------------------

def test_abinit_wfk_parity(tmp_path):
    pytest.importorskip("netCDF4")
    pytest.importorskip("HamiltonIO")
    from HamiltonIO.abinit import read_wfk

    from unfolding import unfold_abinit

    out_toml = tmp_path / "wfk_toml.png"
    out_flags = tmp_path / "wfk_flags.png"
    matrix = np.asarray(M_SI)

    # four stored supercell momenta mapped to the primitive frame
    data = read_wfk(WFK)
    kpts = np.mod(np.asarray(data.kpoints[:4]) @ np.linalg.inv(matrix.T), 1.0)
    del data

    toml = _write_toml(
        tmp_path / "wfk.toml",
        {
            "route": "abinit-wfk",
            "input": {"wfk": str(WFK)},
            "structure": {"supercell_matrix": M_SI},
            "path": {"kpoints": kpts.tolist()},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "abinit-wfk",
            "--wfk", str(WFK),
            "--unfold-mat", *(str(v) for row in M_SI for v in row),
            "--kpoints", *(repr(float(v)) for k in kpts for v in k),
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    ax_api = unfold_abinit(WFK, matrix, kpts)
    _assert_parity([ax_toml, ax_flags, ax_api])


# ---------------------------------------------------------------------------
# abinit-ddb
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_anaddb_missing(), reason="anaddb not available")
def test_abinit_ddb_parity(tmp_path, monkeypatch):
    d = _anaddb_dir()
    if d is not None:
        monkeypatch.setenv("PATH", d + os.pathsep + os.environ["PATH"])
    pytest.importorskip("abipy")

    from unfolding.DDB_unfolder import DDB_unfolder

    out_toml = tmp_path / "ddb_toml.png"
    out_flags = tmp_path / "ddb_flags.png"
    # conventional-cubic-cell DDB: vertices in the DDB fractional frame
    bounds = [[0, 0, 0], [0, 1, 0], [0.5, 1, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
    names = ["G", "X", "W", "G", "L"]

    toml = _write_toml(
        tmp_path / "ddb.toml",
        {
            "route": "abinit-ddb",
            "input": {"ddb": str(DDB)},
            "structure": {"sc_mat": M_CU_DDB},
            "path": {"kpoints": bounds, "names": names},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "abinit-ddb",
            "--ddb", str(DDB),
            "--sc-mat", *(str(v) for row in M_CU_DDB for v in row),
            "--kpoints", *(str(v) for k in bounds for v in k),
            "--names", *names,
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    sc_mat = np.linalg.inv(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0)
    ax_api = DDB_unfolder(
        str(DDB),
        kpath_bounds=bounds,
        sc_mat=sc_mat,
        knames=names,
    )
    _assert_parity([ax_toml, ax_flags, ax_api])


# ---------------------------------------------------------------------------
# magnon
# ---------------------------------------------------------------------------

def test_magnon_parity(tmp_path):
    pytest.importorskip("TB2J")
    from unfolding import unfold_tb2j
    from unfolding.cli_magnon import _primitive_path
    from TB2J.magnon.magnon3 import Magnon

    out_toml = tmp_path / "magnon_toml.png"
    out_flags = tmp_path / "magnon_flags.png"

    toml = _write_toml(
        tmp_path / "magnon.toml",
        {
            "route": "magnon",
            "input": {"results": str(TB2J_RESULTS)},
            "structure": {"supercell_matrix": M_AFM},
            "path": {"special_points": "GXMGR", "npts": 12},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "magnon",
            "--results", str(TB2J_RESULTS),
            "--unfold-mat", *(str(v) for row in M_AFM for v in row),
            "--special-points", "GXMGR",
            "--npts", "12",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # direct adapter call with the path built by the existing magnon CLI helper
    m = Magnon.from_TB2J_results(path=str(TB2J_RESULTS))
    m.set_reference(
        Q=(0, 0, 0),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([1.0, 0.0, 0.0]),
    )
    kpts, x, X, _labels = _primitive_path(m.cell, np.asarray(M_AFM), "GXMGR", 12)
    ax_api = unfold_tb2j(
        m,
        np.asarray(M_AFM),
        kpts,
        knames=list("GXMGR"),
        xqpts=x,
        Xqpts=X,
    )
    _assert_parity([ax_toml, ax_flags, ax_api])


# ---------------------------------------------------------------------------
# registry / exports / entry point
# ---------------------------------------------------------------------------

def test_registry_and_exports():
    import unfolding
    from unfolding.config import ROUTES

    assert set(ROUTES) == {
        "siesta", "siesta-wfsx", "phonopy", "abinit-wfk", "abinit-ddb",
        "magnon", "abinit-paw", "openmx", "gpaw", "abacus", "vasp-paw",
        "wannier",
    }
    assert callable(unfolding.run)
    assert callable(unfolding.load_config)
    assert issubclass(unfolding.ConfigError, Exception)


def test_pyproject_console_script():
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert data["project"]["scripts"]["unfolding"] == "unfolding.cli:main"
    # the existing magnon entry point keeps working
    assert data["project"]["scripts"]["unfolding-magnon"] == "unfolding.cli_magnon:main"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _siesta_payload(tmp_path, **overrides):
    prim = tmp_path / "prim.vasp"
    prim.write_text(PRIM_POSCAR)
    payload = {
        "route": "siesta",
        "input": {"fdf": str(SIESTA_FDF)},
        "structure": {"primitive": str(prim), "supercell_matrix": M_SI},
        "path": {"special_points": "GXWGLX", "npts": 12},
    }
    for section, kv in overrides.items():
        if section == "route":
            if kv is None:
                payload.pop("route", None)
            else:
                payload["route"] = kv
        elif kv is None:
            payload.pop(section, None)
        else:
            payload.setdefault(section, {}).update(kv)
    return payload


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"route": None}, "missing required key 'route'"),
        ({"route": "nope"}, "unknown route"),
        ({"input": {"fdf": 3}}, "input.fdf"),
        ({"input": None}, "input.fdf"),  # missing required key
        ({"structure": {"supercell_matrix": [[1, 2], [3, 4]]}},
         "structure.supercell_matrix"),
        ({"path": {"special_points": "GXWGLX", "kpoints": [[0, 0, 0], [0.5, 0, 0.5]]}},
         "not both"),
        ({"path": None}, "path.special_points or path.kpoints"),
        ({"structure": {"primitive": "/nonexistent/prim.vasp"}}, "file not found"),
        ({"options": {"nope": 1}}, "unknown key"),
        ({"path": {"npts": "many"}}, "path.npts"),
    ],
)
def test_load_config_validation_errors(tmp_path, overrides, match):
    from unfolding.config import ConfigError, load_config

    payload = _siesta_payload(tmp_path, **overrides)
    toml = _write_toml(tmp_path / "bad.toml", payload)
    with pytest.raises(ConfigError, match=match):
        load_config(toml)


def test_load_config_invalid_toml(tmp_path):
    from unfolding.config import ConfigError, load_config

    bad = tmp_path / "bad.toml"
    bad.write_text("route = [unclosed")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(bad)


def test_load_config_missing_file():
    from unfolding.config import ConfigError, load_config

    with pytest.raises(ConfigError, match="config file not found"):
        load_config("/nonexistent/unfold.toml")


def test_names_without_xticks_rejected(tmp_path):
    from unfolding.config import ConfigError, load_config

    payload = _siesta_payload(
        tmp_path,
        path={"special_points": None, "kpoints": [[0, 0, 0], [0.5, 0, 0.5], [1, 0, 0]],
              "names": ["G", "X", "G"]},
    )
    del payload["path"]["special_points"]
    toml = _write_toml(tmp_path / "names.toml", payload)
    with pytest.raises(ConfigError, match="xticks"):
        load_config(toml)


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------

def test_cli_config_and_route_are_mutually_exclusive(tmp_path):
    toml = _write_toml(tmp_path / "ok.toml", _siesta_payload(tmp_path))
    from unfolding.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--config", str(toml), "siesta", "--fdf", str(SIESTA_FDF)])
    assert exc.value.code == 2


def test_cli_requires_route_or_config():
    from unfolding.cli import main

    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_cli_returns_2_for_missing_config_file():
    from unfolding.cli import main

    assert main(["--config", "/nonexistent/unfold.toml"]) == 2


def test_cli_flag_error_missing_required(tmp_path, capsys):
    from unfolding.cli import main

    prim = tmp_path / "prim.vasp"
    prim.write_text(PRIM_POSCAR)
    rc = main(["siesta", "--primitive", str(prim)])
    assert rc == 2
    assert "fdf" in capsys.readouterr().err


def test_cli_toml_route_flag_boolean(tmp_path):
    """--no-fermi-shift round-trips through the flag builder."""
    import dataclasses

    from unfolding.config import ROUTES

    fields = {f.name for f in dataclasses.fields(ROUTES["abinit-wfk"].config)}
    assert "fermi_shift" in fields
