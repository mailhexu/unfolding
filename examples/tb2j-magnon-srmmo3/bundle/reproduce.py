#!/usr/bin/env python
"""Reproduce the G-AFM SrMnO3 downfolded-magnon figure (TB2J example).

Run this script from the unpacked bundle directory (it is the working
directory), with the ``unfolding`` package importable, e.g. after
``pip install unfolding[tb2j]``.

Two routes are exercised:

1. Python API (the documented adapter call): load the TB2J
   pickle, set the collinear two-sublattice reference, build the
   pseudo-cubic primitive q-path G-X-M-G-R, unfold, and render the
   weight-coded magnon bands to ``srmmo3_unfolded_api.png``.
2. One-command CLI (``unfolding-magnon``): the exact command documented
   on the example page, writing ``srmmo3.png`` and ``srmmo3.json``.

Both routes check the physics seals: a Goldstone mode at Gamma
(< 0.05 meV; essentially zero with the shipped pickle) and binary
unfolding weights for the pristine G-AFM (every weight is 0 or 1 to
1e-8).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parent
DATA = BUNDLE / "data" / "TB2J_results"
# A_afm = M @ A_pc: the G-AFM sqrt(2)x sqrt(2)x sqrt(2) cell axes are
# (0,1,1), (1,0,1), (1,1,0) in pseudo-cubic units (rows of M).
M_AFM = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
LETTERS = "GXMGR"


def api_route(png_path):
    """Docgen-style API route: returns (goldstone_mev, binary_dev)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.dft.kpoints import bandpath, get_special_points

    from TB2J.magnon.magnon3 import Magnon

    from unfolding import MagnonUnfolder
    from unfolding.plotphon import plot_band_weight
    from unfolding.tb2j_unfold import magnon_eigendata_from_tb2j

    magnon = Magnon.from_TB2J_results(path=str(DATA))
    # collinear two-sublattice frame: Q = 0, quantization axis z
    magnon.set_reference(
        Q=(0, 0, 0),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([1.0, 0.0, 0.0]),
    )

    prim = np.linalg.solve(M_AFM.astype(float), np.asarray(magnon.cell, float))
    points = get_special_points(prim, eps=0.01)
    path = bandpath([points[c] for c in LETTERS], prim, 200)
    kpts, (x, X, labels) = path.kpts, path.get_linear_kpoint_axis()

    eigendata = magnon_eigendata_from_tb2j(magnon, np.mod(kpts @ M_AFM.T, 1.0))
    unf = MagnonUnfolder(eigendata, M_AFM)
    res = unf.compute(kpts, resolve_degenerate=1e-5)

    energies = res.energies * 1000.0  # meV
    nmode = energies.shape[1]
    ax = plot_band_weight(
        [x for _ in range(nmode)],
        [energies[:, i] for i in range(nmode)],
        [res.weights[:, i] for i in range(nmode)],
        xticks=(labels, X),
        ylabel="Energy (meV)",
        ypad=2.0,
    )
    ax.set_title("G-AFM SrMnO$_3$ downfolded magnons (TB2J)")
    ax.figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)

    return _seals(energies, res.weights, x)


def cli_route(png_path, json_path):
    """One-command CLI route; returns (goldstone_mev, binary_dev)."""
    exe = shutil.which("unfolding-magnon")
    base = [exe] if exe else [sys.executable, "-m", "unfolding.cli_magnon"]
    cmd = base + [
        str(DATA),
        "--unfold-mat", "0", "1", "1", "1", "0", "1", "1", "1", "0",
        "--kpath", LETTERS,
        "--npts", "100",
        "--output", str(png_path),
        "--json", str(json_path),
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    payload = json.loads(Path(json_path).read_text())
    energies = np.asarray(payload["energies_mev"])
    weights = np.asarray(payload["weights"])
    x = np.asarray(payload["x"])
    return _seals(energies, weights, x)


def _seals(energies_mev, weights, x):
    gamma = int(np.argmin(np.abs(x)))
    goldstone = float(np.min(energies_mev[gamma]))
    assert goldstone < 0.05, f"Goldstone seal failed: {goldstone:.4f} meV at Gamma"
    binary = float(
        np.min(np.stack([np.abs(weights), np.abs(weights - 1.0)]))
    )
    assert binary < 1e-8, f"binary-weight seal failed: max deviation {binary:.3e}"
    return goldstone, binary


def main():
    print(f"TB2J results: {DATA}")
    print("\n--- route 1: one-command CLI (unfolding-magnon) ---")
    goldstone, binary = cli_route(BUNDLE / "srmmo3.png", BUNDLE / "srmmo3.json")
    print(f"seals OK: Goldstone at Gamma = {goldstone:.4f} meV (< 0.05), "
          f"max |w - nearest(0, 1)| = {binary:.2e} (< 1e-8)")

    print("\n--- route 2: Python API figure ---")
    goldstone, binary = api_route(BUNDLE / "srmmo3_unfolded_api.png")
    print(f"seals OK: Goldstone at Gamma = {goldstone:.4f} meV (< 0.05), "
          f"max |w - nearest(0, 1)| = {binary:.2e} (< 1e-8)")

    print("\nDone. Outputs:")
    for name in ("srmmo3.png", "srmmo3.json", "srmmo3_unfolded_api.png"):
        p = BUNDLE / name
        print(f"  {name}  ({p.stat().st_size / 1024:.0f} KB)")
    print("Expected physics: weight-coded magnon bands with binary weights "
          "(pristine G-AFM) and a Goldstone mode at Gamma (< 0.05 meV).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
