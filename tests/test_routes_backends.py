"""Story-034: backend route parity (TOML ≡ flags ≡ Python API).

Each route is exercised three ways on committed fixtures -- ``--config``
TOML, explicit flags, and the Python API (the raw adapter where one
exists, the runner config object otherwise) -- and the resulting
weight-coded figures must be identical. Private/licensed inputs (VASP
POTCAR/WAVECAR) and unavailable optional backends (minimulti's Wannier90
reader) are guarded skips, never faked.
"""
from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
M_SI = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]

ABINIT_PAW = REPO / "tests" / "data" / "abinit_paw"
SI_EXAMPLE = REPO / "tests" / "data" / "si_example"
GPAW = REPO / "tests" / "data" / "gpaw_example"
ABACUS = REPO / "tests" / "data" / "abacus_example"

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


def _expected_print(x, energies, weights):
    """Fingerprint a plot_band_weight alpha-style figure would produce."""
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    xs, ys, ws = [], [], []
    for ib in range(weights.shape[1]):
        pts = np.array([x, energies[:, ib]], dtype=float).T
        segs = np.concatenate([pts[:-1][:, None, :], pts[1:][:, None, :]], axis=1)
        sw = np.minimum(weights[:-1, ib], weights[1:, ib])
        xs.append(segs[..., 0].mean(axis=1))
        ys.append(segs[..., 1].mean(axis=1))
        ws.append(np.abs(sw / 2.011))  # alpha = w / (width + 0.011), width=2
    xf, yf, wf = (np.concatenate(a) for a in (xs, ys, ws))
    order = np.lexsort((wf, yf, xf))
    return xf[order], yf[order], wf[order]


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


def _flat(matrix):
    return [str(v) for row in matrix for v in row]


# ---------------------------------------------------------------------------
# abinit-paw
# ---------------------------------------------------------------------------

def test_abinit_paw_parity(tmp_path):
    pytest.importorskip("netCDF4")
    pytest.importorskip("HamiltonIO")
    from HamiltonIO.abinit import HARTREE_TO_EV, read_paw_wfk

    from unfolding.abinit_paw import unfold_abinit_paw
    from unfolding.config import AbinitPawConfig
    from unfolding.routes import run

    supercell = ABINIT_PAW / "si7p_gammao_WFK.nc"
    primitive = ABINIT_PAW / "si_primitive_foldso_WFK.nc"
    out_toml = tmp_path / "paw_toml.png"
    out_flags = tmp_path / "paw_flags.png"

    toml = _write_toml(
        tmp_path / "paw.toml",
        {
            "route": "abinit-paw",
            "input": {
                "supercell": str(supercell),
                "primitive": str(primitive),
                "paw": str(ABINIT_PAW),
            },
            "structure": {"supercell_matrix": M_SI},
            "options": {"resolve_degenerate": 1e-3},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "abinit-paw",
            "--supercell", str(supercell),
            "--primitive", str(primitive),
            "--paw", str(ABINIT_PAW),
            "--unfold-mat", *_flat(M_SI),
            "--resolve-degenerate", "0.001",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    cfg = AbinitPawConfig(
        supercell=str(supercell), primitive=str(primitive),
        paw=str(ABINIT_PAW), supercell_matrix=M_SI,
        resolve_degenerate=1e-3, output=str(tmp_path / "paw_api.png"))
    ax_api = run(cfg)
    _assert_parity([ax_toml, ax_flags, ax_api])

    # the figure must encode the raw adapter's numbers (eV vs E_F, weights)
    xml = {s: ABINIT_PAW / f"{s}.xml" for s in ("Si", "P")}
    sc_data = read_paw_wfk(supercell, xml)
    prim_data = read_paw_wfk(primitive, xml)
    result = unfold_abinit_paw(
        sc_data, prim_data, xml, np.asarray(M_SI),
        resolve_degenerate=1e-3 / HARTREE_TO_EV)
    energies = (result.eigenvalues - sc_data.wavefunctions.fermi_energy) \
        * HARTREE_TO_EV
    bcart = 2 * np.pi * np.linalg.inv(prim_data.wavefunctions.rprimd).T
    dk = np.diff(result.kpoints, axis=0) @ bcart
    x = np.concatenate([[0.0], np.cumsum(np.linalg.norm(dk, axis=1))])
    np.testing.assert_allclose(
        _fingerprint(ax_toml), _expected_print(x, energies, result.weights),
        rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# openmx
# ---------------------------------------------------------------------------

def test_openmx_parity(tmp_path):
    pytest.importorskip("HamiltonIO")
    from ase.dft.kpoints import bandpath, get_special_points
    from ase.io import read as ase_read

    from unfolding import unfold_openmx

    scfout = SI_EXAMPLE / "openmx_si_sc.scfout"
    if not scfout.is_file():
        pytest.skip("OpenMX Si8 fixture missing")
    prim = tmp_path / "prim.vasp"
    prim.write_text(PRIM_POSCAR)
    out_toml = tmp_path / "openmx_toml.png"
    out_flags = tmp_path / "openmx_flags.png"

    toml = _write_toml(
        tmp_path / "openmx.toml",
        {
            "route": "openmx",
            "input": {"scfout": str(scfout)},
            "structure": {"primitive": str(prim), "supercell_matrix": M_SI},
            "path": {"special_points": "GXWGLX", "npts": 8},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "openmx",
            "--scfout", str(scfout),
            "--primitive", str(prim),
            "--unfold-mat", *_flat(M_SI),
            "--special-points", "GXWGLX",
            "--npts", "8",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # direct adapter call with the same independently built path
    cell = ase_read(prim).cell.array
    points = get_special_points(cell, eps=0.01)
    path = bandpath([points[c] for c in "GXWGLX"], cell, 8)
    x, X, _labels = path.get_linear_kpoint_axis()
    ax_api = unfold_openmx(
        scfout=str(scfout),
        prim_atoms=ase_read(prim),
        unfold_sc_mat=np.asarray(M_SI),
        kpts=path.kpts,
        knames=list("GXWGLX"),
        xqpts=x,
        Xqpts=X,
        method="ideal",
    )
    _assert_parity([ax_toml, ax_flags, ax_api])


# ---------------------------------------------------------------------------
# gpaw
# ---------------------------------------------------------------------------

def test_gpaw_lcao_parity(tmp_path):
    pytest.importorskip("gpaw")
    pytest.importorskip("HamiltonIO")
    from HamiltonIO.gpaw import GpawLcaoModel
    from unfolding.lcao_unfolder import LCAOUnfolder
    from unfolding.mapping import RelabelMap
    from unfolding.plotphon import plot_band_weight

    sc_gpw = GPAW / "si_sc_lcao.gpw"
    prim_gpw = GPAW / "si_prim_lcao.gpw"
    if not (sc_gpw.is_file() and prim_gpw.is_file()):
        pytest.skip("GPAW LCAO fixtures missing")
    out_toml = tmp_path / "gpaw_toml.png"
    out_flags = tmp_path / "gpaw_flags.png"

    toml = _write_toml(
        tmp_path / "gpaw.toml",
        {
            "route": "gpaw",
            "input": {"supercell": str(sc_gpw)},
            "structure": {"primitive": str(prim_gpw),
                          "supercell_matrix": M_SI},
            "path": {"special_points": "GXWGLWX", "npts": 8},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "gpaw",
            "--supercell", str(sc_gpw),
            "--primitive", str(prim_gpw),
            "--unfold-mat", *_flat(M_SI),
            "--special-points", "GXWGLWX",
            "--npts", "8",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # raw composition oracle (the sealed test_gpaw_unfolder pipeline)
    prim = GpawLcaoModel.from_file(prim_gpw)
    sc = GpawLcaoModel.from_file(sc_gpw)
    rm = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, np.asarray(M_SI),
        orb_counts_sc=[4] * 8, orb_counts_prim=[4, 4],
    )
    path = _gxwglwx_path(prim.atoms.cell, 8)
    res = LCAOUnfolder(sc, rm).compute(path.kpts, method="ideal")
    x, X, _labels = path.get_linear_kpoint_axis()
    energies = res.eigenvalues - sc.efermi
    w = np.clip(res.weights, 0.0, 1.0)
    nb = w.shape[1]
    ax_api = plot_band_weight(
        [x] * nb, [energies[:, ib] for ib in range(nb)],
        [w[:, ib] for ib in range(nb)],
        xticks=[list("GXWGLWX"), X], ylabel="Energy (eV)", ypad=1.5)
    _assert_parity([ax_toml, ax_flags, ax_api])


def _gxwglwx_path(cell, npts):
    """k-path object over GXWGLWX on a primitive cell."""
    from ase.dft.kpoints import bandpath, get_special_points

    points = get_special_points(np.asarray(cell, dtype=float), eps=0.01)
    return bandpath([points[c] for c in "GXWGLWX"],
                    np.asarray(cell, dtype=float), npts)


def test_gpaw_pw_parity(tmp_path):
    pytest.importorskip("gpaw")
    pytest.importorskip("HamiltonIO")
    from ase.dft.kpoints import bandpath

    from HamiltonIO.gpaw import GpawPWParser
    from unfolding.pw_unfolder import PWUnfolder
    from unfolding.plotphon import plot_band_weight

    from unfolding.pw_unfolder import PWEigenData

    sc_gpw = GPAW / "si8_pw.gpw"
    if not sc_gpw.is_file():
        pytest.skip("GPAW PW fixture missing")
    data = GpawPWParser(sc_gpw).read()
    kprim = (np.asarray(data.kpoints) @ np.linalg.inv(np.asarray(M_SI)))[:6]
    del data
    out_toml = tmp_path / "gpawpw_toml.png"
    out_flags = tmp_path / "gpawpw_flags.png"

    toml = _write_toml(
        tmp_path / "gpawpw.toml",
        {
            "route": "gpaw",
            "input": {"supercell": str(sc_gpw)},
            "structure": {"supercell_matrix": M_SI},
            "path": {"kpoints": kprim.tolist()},
            "options": {"mode": "pw", "resolve_degenerate": 1e-4},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "gpaw",
            "--supercell", str(sc_gpw),
            "--unfold-mat", *_flat(M_SI),
            "--kpoints", *(repr(float(v)) for k in kprim for v in k),
            "--mode", "pw",
            "--resolve-degenerate", "1e-4",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # raw composition oracle
    data = GpawPWParser(sc_gpw).read()
    eigendata = PWEigenData(
        data.kpoints, data.gvecs,
        [c[None, :, None, :] for c in data.coefficients],
        data.eigenvalues[:, None, :],
    )
    res = PWUnfolder(eigendata, np.asarray(M_SI)).compute(
        kprim, resolve_degenerate=1e-4)
    x = np.arange(len(kprim), dtype=float)
    energies = res.eigenvalues - data.efermi
    w = np.clip(res.weights, 0.0, 1.0)
    nb = w.shape[1]
    ax_api = plot_band_weight(
        [x] * nb, [energies[:, ib] for ib in range(nb)],
        [w[:, ib] for ib in range(nb)],
        ylabel="Energy (eV)", ypad=1.5)
    _assert_parity([ax_toml, ax_flags, ax_api])


# ---------------------------------------------------------------------------
# abacus
# ---------------------------------------------------------------------------

def test_abacus_lcao_parity(tmp_path):
    pytest.importorskip("HamiltonIO")
    from ase.dft.kpoints import bandpath

    from HamiltonIO.abacus.abacus_wrapper import AbacusParser
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap
    from unfolding.plotphon import plot_band_weight

    sc_dir = ABACUS / "si_conv" / "OUT.si_conv"
    prim_dir = ABACUS / "si_prim" / "OUT.si_prim"
    if not (sc_dir.is_dir() and prim_dir.is_dir()):
        pytest.skip("ABACUS LCAO fixtures missing")
    out_toml = tmp_path / "abacus_toml.png"
    out_flags = tmp_path / "abacus_flags.png"

    toml = _write_toml(
        tmp_path / "abacus.toml",
        {
            "route": "abacus",
            "input": {"supercell": str(sc_dir)},
            "structure": {"primitive": str(prim_dir),
                          "supercell_matrix": M_SI},
            "path": {"special_points": "GXWGLX", "npts": 8},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "abacus",
            "--supercell", str(sc_dir),
            "--primitive", str(prim_dir),
            "--unfold-mat", *_flat(M_SI),
            "--special-points", "GXWGLX",
            "--npts", "8",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # raw composition oracle (counts fixed to the sealed DZP values)
    sc = AbacusParser(outpath=sc_dir).get_models()
    prim = AbacusParser(outpath=prim_dir).get_models()
    rm = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, np.asarray(M_SI),
        orb_counts_sc=[13] * 8, orb_counts_prim=[13, 13],
    )
    cell = np.linalg.inv(np.asarray(M_SI, dtype=float)) \
        @ np.asarray(sc.atoms.cell, dtype=float)
    path = _gxwglx_path(cell, 8)
    res = LCAOUnfolder(HamiltonIOModel(sc), rm).compute(
        path.kpts, method="ideal")
    x, X, _labels = path.get_linear_kpoint_axis()
    energies = res.eigenvalues - sc.efermi
    w = np.clip(res.weights, 0.0, 1.0)
    nb = w.shape[1]
    ax_api = plot_band_weight(
        [x] * nb, [energies[:, ib] for ib in range(nb)],
        [w[:, ib] for ib in range(nb)],
        xticks=[list("GXWGLX"), X], ylabel="Energy (eV)", ypad=1.5)
    _assert_parity([ax_toml, ax_flags, ax_api])


def _gxwglx_path(cell, npts):
    from ase.dft.kpoints import bandpath, get_special_points

    points = get_special_points(np.asarray(cell, dtype=float), eps=0.01)
    return bandpath([points[c] for c in "GXWGLX"],
                    np.asarray(cell, dtype=float), npts)


def test_abacus_pw_parity(tmp_path):
    pytest.importorskip("HamiltonIO")
    from HamiltonIO.abacus.pw_wfc import AbacusPWParser
    from unfolding.plotphon import plot_band_weight
    from unfolding.pw_unfolder import PWEigenData, PWUnfolder

    sc_dir = ABACUS / "si_pw_path" / "OUT.si_pw_path"
    if not sc_dir.is_dir():
        pytest.skip("ABACUS PW fixture missing")
    data = AbacusPWParser(outpath=sc_dir).read()
    kprim = (np.asarray(data.kpoints) @ np.linalg.inv(np.asarray(M_SI)))[:6]
    del data
    out_toml = tmp_path / "abacuspw_toml.png"
    out_flags = tmp_path / "abacuspw_flags.png"

    toml = _write_toml(
        tmp_path / "abacuspw.toml",
        {
            "route": "abacus",
            "input": {"supercell": str(sc_dir)},
            "structure": {"supercell_matrix": M_SI},
            "path": {"kpoints": kprim.tolist()},
            "options": {"mode": "pw", "resolve_degenerate": 1e-4},
            "output": {"output": str(out_toml)},
        },
    )

    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "abacus",
            "--supercell", str(sc_dir),
            "--unfold-mat", *_flat(M_SI),
            "--kpoints", *(repr(float(v)) for k in kprim for v in k),
            "--mode", "pw",
            "--resolve-degenerate", "1e-4",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # raw composition oracle
    data = AbacusPWParser(outpath=sc_dir).read()
    eigendata = PWEigenData(
        data.kpoints, data.gvecs,
        [c[None, :, None, :] for c in data.coefficients],
        data.eigenvalues[:, None, :],
    )
    res = PWUnfolder(eigendata, np.asarray(M_SI)).compute(
        kprim, resolve_degenerate=1e-4)
    x = np.arange(len(kprim), dtype=float)
    energies = res.eigenvalues - data.efermi
    w = np.clip(res.weights, 0.0, 1.0)
    nb = w.shape[1]
    ax_api = plot_band_weight(
        [x] * nb, [energies[:, ib] for ib in range(nb)],
        [w[:, ib] for ib in range(nb)],
        ylabel="Energy (eV)", ypad=1.5)
    _assert_parity([ax_toml, ax_flags, ax_api])


# ---------------------------------------------------------------------------
# siesta (match_species) and siesta-wfsx
# ---------------------------------------------------------------------------

def test_siesta_match_species_parity(tmp_path):
    """Si:P (substitutional dopant) needs match_species=false; the TOML
    and flags routes agree with the direct adapter call."""
    pytest.importorskip("HamiltonIO")
    from ase.dft.kpoints import bandpath, get_special_points
    from ase.io import read as ase_read

    from unfolding import unfold_siesta

    scfdf = SI_EXAMPLE / "si_sc_p.fdf"
    if not scfdf.is_file():
        pytest.skip("doped SIESTA fixture missing")
    prim = tmp_path / "prim.vasp"
    prim.write_text(PRIM_POSCAR)
    out_toml = tmp_path / "siesta_p_toml.png"
    out_flags = tmp_path / "siesta_p_flags.png"

    toml = _write_toml(
        tmp_path / "siesta_p.toml",
        {
            "route": "siesta",
            "input": {"fdf": str(scfdf)},
            "structure": {"primitive": str(prim), "supercell_matrix": M_SI},
            "path": {"kpoints": [[0, 0, 0], [0, 0.5, 0.5],
                                 [0.5, 0, 0.5], [0.5, 0.5, 0]]},
            "options": {"match_species": False},
            "output": {"output": str(out_toml)},
        },
    )
    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "siesta",
            "--fdf", str(scfdf),
            "--primitive", str(prim),
            "--unfold-mat", *_flat(M_SI),
            "--kpoints", "0.0", "0.0", "0.0", "0.0", "0.5", "0.5",
            "0.5", "0.0", "0.5", "0.5", "0.5", "0.0",
            "--no-match-species",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # commensurate fold points (the star of Gamma) with the route-default
    # ring weights: the torus projection is exact there
    kpts = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
                    dtype=float)
    x = np.arange(len(kpts), dtype=float)
    ax_api = unfold_siesta(
        fdf=str(scfdf),
        prim_atoms=str(prim),
        unfold_sc_mat=np.asarray(M_SI),
        kpts=kpts,
        xqpts=x,
        match_species=False,
    )
    _assert_parity([ax_toml, ax_flags, ax_api])


def test_siesta_wfsx_parity(tmp_path):
    """WFSX route on the committed path fixture: TOML ≡ flags ≡ the raw
    WFSXUnfolder composition (weights equal the HSX-diagonalization
    reference, sealed in test_wfsx_unfolder.py)."""
    pytest.importorskip("HamiltonIO")
    pytest.importorskip("sisl")
    from HamiltonIO.siesta.sisl_wrapper import SislParser
    from HamiltonIO.siesta.wfsx import SiestaWFSXParser

    from unfolding.lcao_unfolder import HamiltonIOModel
    from unfolding.mapping import RelabelMap
    from unfolding.plotphon import plot_band_weight
    from unfolding.wfsx_unfolder import WFSXUnfolder

    wfsx_path = SI_EXAMPLE / "si_sc_path.selected.WFSX"
    sc_fdf = SI_EXAMPLE / "si_sc.fdf"
    prim_fdf = SI_EXAMPLE / "si_prim.fdf"
    if not (wfsx_path.is_file() and sc_fdf.is_file() and prim_fdf.is_file()):
        pytest.skip("SIESTA WFSX fixtures missing")

    sc = SislParser(sc_fdf).get_model()
    prim = SislParser(prim_fdf).get_model()
    M = np.asarray(M_SI)
    # request a subset of the STORED path points mapped to the primitive
    # frame: the unfolder matches requested supercell momenta to stored
    # entries at 1e-6, so only stored points are answerable
    stored = SiestaWFSXParser(
        wfsx_path, cell=np.asarray(sc.atoms.cell, dtype=float)).read()
    kprim = (np.asarray(stored.kpoints, dtype=float)
             @ np.linalg.inv(M))[::30]
    del stored

    out_toml = tmp_path / "wfsx_toml.png"
    out_flags = tmp_path / "wfsx_flags.png"
    toml = _write_toml(
        tmp_path / "wfsx.toml",
        {
            "route": "siesta-wfsx",
            "input": {"wfsx": str(wfsx_path), "hs_fdf": str(sc_fdf)},
            "structure": {"primitive": str(prim_fdf),
                          "supercell_matrix": M_SI},
            "path": {"kpoints": kprim.tolist()},
            "options": {"method": "ideal"},
            "output": {"output": str(out_toml)},
        },
    )
    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "siesta-wfsx",
            "--wfsx", str(wfsx_path),
            "--hs-fdf", str(sc_fdf),
            "--primitive", str(prim_fdf),
            "--unfold-mat", *_flat(M_SI),
            "--kpoints", *(repr(float(v)) for k in kprim for v in k),
            "--method", "ideal",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    # raw composition oracle (the pipeline of docs/examples + EIG header E_F)
    rm = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, M,
        orb_counts_sc=[4] * 8, orb_counts_prim=[4, 4],
    )
    wfsx = SiestaWFSXParser(
        wfsx_path, cell=np.asarray(sc.atoms.cell, dtype=float)).read()
    unf = WFSXUnfolder(wfsx, HamiltonIOModel(sc), rm, M)
    res = unf.compute(kprim, method="ideal")
    with open(SI_EXAMPLE / "si_sc_path.EIG") as fh:
        efermi = float(fh.readline().split()[0])
    x = np.arange(len(kprim), dtype=float)
    energies = res.eigenvalues - efermi
    w = np.clip(res.weights, 0.0, 1.0)
    nb = w.shape[1]
    ax_api = plot_band_weight(
        [x] * nb, [energies[:, ib] for ib in range(nb)],
        [w[:, ib] for ib in range(nb)],
        ylabel="Energy (eV)", ypad=1.5)
    _assert_parity([ax_toml, ax_flags, ax_api])


# ---------------------------------------------------------------------------
# vasp-paw (licensed inputs; schema parity always, execution guarded)
# ---------------------------------------------------------------------------

def _vasp_payload(tmp_path, out):
    files = {}
    for name in ("WAVECAR_sc", "POSCAR_sc", "WAVECAR_prim", "POSCAR_prim",
                 "POTCAR"):
        path = tmp_path / name
        path.touch()
        files[name] = path
    return {
        "route": "vasp-paw",
        "input": {
            "supercell": str(files["WAVECAR_sc"]),
            "supercell_poscar": str(files["POSCAR_sc"]),
            "primitive": str(files["WAVECAR_prim"]),
            "primitive_poscar": str(files["POSCAR_prim"]),
            "potcar": str(files["POTCAR"]),
        },
        "structure": {"supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "path": {"names": ["G", "H"], "xticks": [0.0, 1.0]},
        "options": {"spin": 1, "resolve_degenerate": 0.002},
        "output": {"output": str(out)},
    }, files


def test_vasp_paw_schema_parity(tmp_path):
    from unfolding.cli import build_parser
    from unfolding.config import ConfigError, config_from_flags, load_config

    payload, _files = _vasp_payload(tmp_path, tmp_path / "out.png")
    toml = _write_toml(tmp_path / "vasp.toml", payload)
    cfg_toml = load_config(toml)

    argv = ["vasp-paw"]
    for key, value in payload["input"].items():
        argv += [f"--{key.replace('_', '-')}", value]
    argv += ["--unfold-mat", "1", "0", "0", "0", "1", "0", "0", "0", "1",
             "--names", "G", "H", "--xticks", "0.0", "1.0",
             "--spin", "1", "--resolve-degenerate", "0.002",
             "--output", payload["output"]["output"]]
    cfg_flags = config_from_flags("vasp-paw", build_parser().parse_args(argv))
    assert cfg_toml == cfg_flags

    # stored-k route: [path] is optional and labels-only
    payload.pop("path")
    toml = _write_toml(tmp_path / "vasp_nopath.toml", payload)
    load_config(toml)

    # ...but names/xticks stay paired
    payload["path"] = {"names": ["G", "H"]}
    toml = _write_toml(tmp_path / "vasp_bad.toml", payload)
    with pytest.raises(ConfigError, match="path.names and path.xticks"):
        load_config(toml)


_VASP_SEED = Path(os.environ.get("UNFOLDING_VASP_FE_SEED", ""))
_VASP_RUNS = Path(os.environ.get("UNFOLDING_VASP_FE_PATH", ""))
_VASP_MATRIX = [[0, 2, 2], [2, 0, 2], [2, 2, 0]]  # 2x2x2 conventional bcc
_VASP_GUARD = (
    os.environ.get("UNFOLDING_VASP_FE_SEED")
    and (_VASP_SEED / "POTCAR").is_file()
    and (_VASP_RUNS / "sc16_nscf" / "WAVECAR").is_file()
    and (_VASP_RUNS / "sc16_nscf" / "POSCAR").is_file()
    and (_VASP_RUNS / "prim_nscf" / "WAVECAR").is_file()
    and (_VASP_RUNS / "prim_nscf" / "POSCAR").is_file()
)


@pytest.mark.skipif(
    not _VASP_GUARD,
    reason="private licensed VASP fixture unavailable "
           "(set UNFOLDING_VASP_FE_SEED and UNFOLDING_VASP_FE_PATH)",
)
def test_vasp_paw_execution(tmp_path):
    """Real bcc-Fe PAW unfolding end to end: SC16 nscf supercell vs
    primitive nscf reference, private POTCAR. The three entry points are
    each compared against the raw adapter's numbers (weights/energies)
    rather than to each other: the PAW weights carry conditioning-level
    (1e-4 alpha) run-to-run noise under concurrent load, while the
    adapter result is the stable anchor."""
    from unfolding.config import VaspPawConfig
    from unfolding.routes import run
    from unfolding.vasp_paw import read_wavecar_ordered, unfold_vasp_paw

    sc = _VASP_RUNS / "sc16_nscf"
    prim = _VASP_RUNS / "prim_nscf"
    out_toml = tmp_path / "vasp_toml.png"
    out_flags = tmp_path / "vasp_flags.png"

    toml = _write_toml(
        tmp_path / "vasp.toml",
        {
            "route": "vasp-paw",
            "input": {
                "supercell": str(sc / "WAVECAR"),
                "supercell_poscar": str(sc / "POSCAR"),
                "primitive": str(prim / "WAVECAR"),
                "primitive_poscar": str(prim / "POSCAR"),
                "potcar": str(_VASP_SEED / "POTCAR"),
            },
            "structure": {"supercell_matrix": _VASP_MATRIX},
            "output": {"output": str(out_toml)},
        },
    )
    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "vasp-paw",
            "--supercell", str(sc / "WAVECAR"),
            "--supercell-poscar", str(sc / "POSCAR"),
            "--primitive", str(prim / "WAVECAR"),
            "--primitive-poscar", str(prim / "POSCAR"),
            "--potcar", str(_VASP_SEED / "POTCAR"),
            "--unfold-mat", *_flat(_VASP_MATRIX),
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()
    ax_api = run(VaspPawConfig(
        supercell=str(sc / "WAVECAR"),
        supercell_poscar=str(sc / "POSCAR"),
        primitive=str(prim / "WAVECAR"),
        primitive_poscar=str(prim / "POSCAR"),
        potcar=str(_VASP_SEED / "POTCAR"),
        supercell_matrix=_VASP_MATRIX,
        output=str(tmp_path / "vasp_api.png")))

    # adapter oracle: PAW restoration + bounded weights, then the exact
    # arrays behind the figure
    supercell = read_wavecar_ordered(sc / "WAVECAR", sc / "POSCAR")
    primitive = read_wavecar_ordered(prim / "WAVECAR", prim / "POSCAR")
    result = unfold_vasp_paw(
        supercell, primitive, _VASP_SEED / "POTCAR",
        np.asarray(_VASP_MATRIX), spin=0)
    assert np.max(np.abs(result.norm_residuals)) < 1e-3
    assert result.weights.min() > -1e-3
    assert result.weights.max() < 1.0 + 1e-3

    from unfolding.routes import _cumulative_cart_x

    energies = result.eigenvalues - supercell.fermi_energy
    x = _cumulative_cart_x(result.kpoints, primitive.lattice)
    expected = _expected_print(x, energies, result.weights)
    for tag, ax in (("toml", ax_toml), ("flags", ax_flags), ("api", ax_api)):
        got = _fingerprint(ax)
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(
            got[2], expected[2], rtol=0.05, atol=2e-3,
            err_msg=f"{tag} leg weights drift from the adapter")


# ---------------------------------------------------------------------------
# wannier (minimulti-gated; schema parity always, execution guarded)
# ---------------------------------------------------------------------------

def test_wannier_schema_parity(tmp_path):
    from unfolding.cli import build_parser
    from unfolding.config import ConfigError, config_from_flags, load_config

    hr_dir = tmp_path / "wannier"
    hr_dir.mkdir()
    (hr_dir / "wannier90_hr.dat").touch()
    labels = ["d_xy", "d_yz", "d_zx"] * 2
    out = str(tmp_path / "sto.png")
    payload = {
        "route": "wannier",
        "input": {"path": str(hr_dir), "prefix": "wannier90"},
        "structure": {
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "labels": labels,
            "cell": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "path": {
            "kpoints": [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]],
            "names": ["G", "X", "M"],
        },
        "options": {"npoints": 20},
        "output": {"output": out},
    }
    toml = _write_toml(tmp_path / "wannier.toml", payload)
    cfg_toml = load_config(toml)

    argv = [
        "wannier",
        "--path", str(hr_dir),
        "--prefix", "wannier90",
        "--unfold-mat", "2", "0", "0", "0", "2", "0", "0", "0", "2",
        "--labels", *labels,
        "--cell", "1", "0", "0", "0", "1", "0", "0", "0", "1",
        "--kpoints", "0.0", "0.0", "0.0", "0.5", "0.0", "0.0",
        "0.5", "0.5", "0.0",
        "--names", "G", "X", "M",
        "--npoints", "20",
        "--output", out,
    ]
    cfg_flags = config_from_flags(
        "wannier", build_parser().parse_args(argv))
    assert cfg_toml == cfg_flags

    # defaults round-trip: omitting --prefix keeps the dataclass default
    argv2 = [
        "wannier",
        "--path", str(hr_dir),
        "--unfold-mat", "2", "0", "0", "0", "2", "0", "0", "0", "2",
        "--labels", *labels,
        "--cell", "1", "0", "0", "0", "1", "0", "0", "0", "1",
        "--kpoints", "0.0", "0.0", "0.0", "0.5", "0.0", "0.0",
        "0.5", "0.5", "0.0",
        "--names", "G", "X", "M",
        "--npoints", "20",
        "--output", out,
    ]
    cfg_default = config_from_flags(
        "wannier", build_parser().parse_args(argv2))
    assert cfg_default == cfg_toml

    # required fields are enforced
    bad = dict(payload)
    bad["structure"] = {"supercell_matrix": payload["structure"]["supercell_matrix"]}
    toml = _write_toml(tmp_path / "wannier_bad.toml", bad)
    with pytest.raises(ConfigError, match="structure.labels: is required"):
        load_config(toml)


def test_wannier_execution(tmp_path):
    """Real wannier route on the shipped synthetic STO fixture (bundled
    t2g hr model, cell pinned): TOML ≡ flags ≡ Python API."""
    bundle = REPO / "docs" / "static" / "downloads" / "wannier-sto.tar.gz"
    if not bundle.is_file():
        pytest.skip("wannier-sto bundle not built yet")
    work = tmp_path / "wannier"
    work.mkdir()
    with tarfile.open(bundle) as tar:
        tar.extractall(tmp_path / "bundle")
    data = tmp_path / "bundle" / "wannier-sto" / "data" / "example_hr.dat"
    if not data.is_file():
        pytest.skip("wannier-sto bundle has no example_hr.dat")
    (work / "wannier90_hr.dat").write_bytes(data.read_bytes())

    labels = ["d_xy", "d_yz", "d_zx"] * 8
    eye = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    diag2 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    kvectors = [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
    out_toml = tmp_path / "sto_toml.png"
    out_flags = tmp_path / "sto_flags.png"
    payload = {
        "route": "wannier",
        "input": {"path": str(work), "prefix": "wannier90"},
        "structure": {"supercell_matrix": diag2, "labels": labels,
                      "cell": eye},
        "path": {"kpoints": kvectors, "names": ["G", "X", "M", "G", "R"]},
        "options": {"npoints": 20},
        "output": {"output": str(out_toml)},
    }
    toml = _write_toml(tmp_path / "wannier.toml", payload)
    ax_toml = _run_cli(["--config", str(toml)])
    ax_flags = _run_cli(
        [
            "wannier",
            "--path", str(work),
            "--unfold-mat", "2", "0", "0", "0", "2", "0", "0", "0", "2",
            "--labels", *labels,
            "--cell", "1", "0", "0", "0", "1", "0", "0", "0", "1",
            "--kpoints", *(repr(float(v)) for k in kvectors for v in k),
            "--names", "G", "X", "M", "G", "R",
            "--npoints", "20",
            "--output", str(out_flags),
        ]
    )
    assert out_toml.is_file() and out_flags.is_file()

    from unfolding.config import WannierConfig
    from unfolding.routes import run

    ax_api = run(WannierConfig(
        path=str(work), prefix="wannier90", supercell_matrix=diag2,
        labels=labels, cell=eye, kpoints=kvectors,
        names=["G", "X", "M", "G", "R"], npoints=20,
        output=str(tmp_path / "sto_api.png")))
    _assert_parity([ax_toml, ax_flags, ax_api])
