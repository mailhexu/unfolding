#!/usr/bin/env python
"""Reproduce the VASP PAW bcc-Fe unfolding figure (vasp_fe_path.png).

Docs example: docs/content/examples/vasp-paw.md in the unfolding repo.

Two modes:

1. Snapshot (no VASP needed, out of the box):

       python reproduce.py --snapshot

   Redraws the committed spin-up/spin-down figure from the shipped
   data/vasp_fe_path_result.npz -- the unfolding output (weights,
   eigenvalues, primitive reference bands, Fermi levels) of the
   fixture run, stored as plain arrays.

2. Full pipeline (needs your licensed POTCAR and two VASP runs):

       python make_inputs.py && cd runs/... (VASP) && python reproduce.py

   See README.txt. Unfolds the supercell and primitive WAVECARs with
   the PAW-metric adapter for both spins, overlays the primitive-cell
   bands, prints the consistency checks, and writes
   vasp_fe_path.png.

Requires the ``unfolding`` package importable (pip install -e <unfolding
repo checkout>, or PYTHONPATH=<unfolding repo checkout>), numpy and
matplotlib. The full pipeline additionally needs pymatgen (WAVECAR
parsing) and, of course, VASP with a PAW_PBE Fe POTCAR -- the POTCAR is
licensed and is never shipped or stored.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

DEGEN_TOL_EV = 1e-3  # gauge-invariant degenerate-group weights
SIGMA_EV = 0.045  # Gaussian map width of the committed spectral map
WINDOW_EV = (-8.0, 8.0)  # metallic Fe: d bands + sp states around E_F


def read_efermi(outcar):
    """E-fermi (eV) from an OUTCAR."""
    match = re.search(r"E-fermi\s*:\s*([-\d.]+)", Path(outcar).read_text())
    if match is None:
        raise ValueError(f"no E-fermi line in {outcar}")
    return float(match.group(1))


def snapshot(figure=None):
    """Redraw the committed figure from the shipped result arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = np.load(DATA / "vasp_fe_path_result.npz", allow_pickle=True)
    x = res["x"]
    X = res["X"]
    names = [r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H"]
    ef_sc = float(res["ef_sc"])
    ef_prim = float(res["ef_prim"])
    egrid = np.linspace(WINDOW_EV[0] - 0.8, WINDOW_EV[1] + 0.8, 1100)
    pre = 1.0 / (SIGMA_EV * np.sqrt(2 * np.pi))
    figure = Path(figure) if figure else HERE / "vasp_fe_path.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, spin, title in zip(
            axes, (0, 1),
            (r"$2{\times}2{\times}2$ bcc Fe unfolded, spin up",
             r"$2{\times}2{\times}2$ bcc Fe unfolded, spin down")):
        energies = res["eigenvalues"][spin] - ef_sc
        weights = np.clip(res["weights"][spin], 0.0, 1.0)
        A = pre * np.exp(-0.5 * ((egrid[:, None, None] - energies) / SIGMA_EV) ** 2)
        A = (A * weights[None]).sum(axis=2)
        ax.pcolormesh(x, egrid, A, cmap="Blues", vmin=0.0, vmax=2.0,
                      shading="auto", rasterized=True)
        ax.plot(x, res["prim_eigenvalues"][spin] - ef_prim, color="crimson",
                lw=1.0, alpha=0.9, zorder=5)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor=matplotlib.colormaps["Blues"](0.75),
                      label="unfolded spectral weight"),
                Line2D([0], [0], color="crimson", lw=1.0,
                       label="primitive-cell bands"),
            ],
            loc="upper right", fontsize=8, framealpha=0.85,
        )
        for xt in X[1:-1]:
            ax.axvline(xt, color="gray", lw=0.5)
        ax.axhline(0.0, ls="--", color="k", lw=0.7)
        ax.set_xticks(list(X))
        ax.set_xticklabels(names)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(*WINDOW_EV)
        ax.set_title(title)
    axes[0].set_ylabel(r"Energy relative to $E_F$ (eV)")
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print("wrote", figure)
    return figure


def full_pipeline(potcar):
    """Unfold the user-produced WAVECARs and draw the figure."""
    runs = HERE / "runs"
    sc = runs / "sc16_nscf"
    prim = runs / "prim_nscf"
    for need in (potcar, sc / "WAVECAR", prim / "WAVECAR",
                 runs / "sc16_scf/OUTCAR", runs / "prim_scf/OUTCAR"):
        if not need.is_file():
            raise SystemExit(
                f"missing {need} -- see README.txt: run make_inputs.py, "
                "copy your POTCAR into each run directory, run VASP in "
                "all four inputs/* directories (keep the directory "
                "names under runs/), then re-run this script")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import sys

    sys.path.insert(0, str(HERE))
    from make_inputs import MATRIX, PATH, primitive_path
    from read_wavecar_ordered import read_wavecar_ordered
    from unfolding.vasp_paw import unfold_vasp_paw

    ef_sc = read_efermi(runs / "sc16_scf/OUTCAR")
    ef_prim = read_efermi(runs / "prim_scf/OUTCAR")
    sc_data = read_wavecar_ordered(sc / "WAVECAR", sc / "POSCAR")
    prim_data = read_wavecar_ordered(prim / "WAVECAR", prim / "POSCAR")

    kprim, x, X, labels = primitive_path()
    if not np.allclose(np.mod(prim_data.kpoints, 1.0), np.mod(kprim, 1.0),
                       atol=1e-7):
        raise ValueError("primitive WAVECAR k-points differ from the "
                         f"{PATH} path of make_inputs.py")

    figure = HERE / "vasp_fe_path.png"
    names = [r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H"]
    egrid = np.linspace(WINDOW_EV[0] - 0.8, WINDOW_EV[1] + 0.8, 1100)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    prim_eigs = np.array(prim_data.eigenvalues)  # (nk, nspin, nb)
    for ax, spin, title in zip(
            axes, (0, 1),
            (r"$2{\times}2{\times}2$ bcc Fe unfolded, spin up",
             r"$2{\times}2{\times}2$ bcc Fe unfolded, spin down")):
        res = unfold_vasp_paw(sc_data, prim_data, potcar, MATRIX, spin=spin,
                              resolve_degenerate=DEGEN_TOL_EV)
        energies = res.eigenvalues - ef_sc
        weights = np.clip(res.weights, 0.0, 1.0)
        pre = 1.0 / (SIGMA_EV * np.sqrt(2 * np.pi))
        A = pre * np.exp(-0.5 * ((egrid[:, None, None] - energies) / SIGMA_EV) ** 2)
        A = (A * weights[None]).sum(axis=2)
        ax.pcolormesh(x, egrid, A, cmap="Blues", vmin=0.0, vmax=2.0,
                      shading="auto", rasterized=True)
        ax.plot(x, prim_eigs[:, spin, :] - ef_prim, color="crimson",
                lw=1.0, alpha=0.9, zorder=5)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor=matplotlib.colormaps["Blues"](0.75),
                      label="unfolded spectral weight"),
                Line2D([0], [0], color="crimson", lw=1.0,
                       label="primitive-cell bands"),
            ],
            loc="upper right", fontsize=8, framealpha=0.85,
        )
        for xt in X[1:-1]:
            ax.axvline(xt, color="gray", lw=0.5)
        ax.axhline(0.0, ls="--", color="k", lw=0.7)
        ax.set_xticks(list(X))
        ax.set_xticklabels(names)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(*WINDOW_EV)
        ax.set_title(title)
        resid = np.max(np.abs(res.norm_residuals))
        print(f"spin {spin}: PAW norm residual max {resid:.2e}")
    axes[0].set_ylabel(r"Energy relative to $E_F$ (eV)")
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print(f"E_F supercell {ef_sc:.4f} eV, primitive {ef_prim:.4f} eV")
    print("wrote", figure)
    return figure


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", action="store_true",
                    help="redraw the committed figure from the shipped "
                         "result arrays (no VASP needed)")
    ap.add_argument("--potcar", default=os.environ.get("UNFOLDING_VASP_FE_POTCAR"),
                    help="path to your licensed PAW_PBE Fe POTCAR "
                         "(full-pipeline mode; or set UNFOLDING_VASP_FE_POTCAR)")
    args = ap.parse_args()
    if args.snapshot:
        snapshot()
    else:
        if not args.potcar:
            ap.error("--potcar (or UNFOLDING_VASP_FE_POTCAR) is required "
                     "without --snapshot")
        full_pipeline(Path(args.potcar))
