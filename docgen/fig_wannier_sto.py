#!/usr/bin/env python
"""Regenerate the Wannier90 SrTiO3 unfolded-band figures (docs/static/images/).

Runs the committed real Wannier90 datasets (examples/wannier_STO:
pristine and Ti-vacancy sqrt(2)xsqrt(2)x2 supercells of cubic SrTiO3,
a = 3.9 A, 56 Wannier functions = 12 O-2p + 4 Ti-3d shells) through the
package's built-in Wannier90 reader and WannierUnfolder, unfolding each
supercell back onto the 5-atom cubic primitive cell. Headless (Agg).

Usage:
    fig_wannier_sto.py [OUT] [VACANCY_OUT]
With no arguments both default figures are (re)written.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, "examples", "wannier_STO"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from unfolding.wannier_unfold import Wannier90Model, WannierUnfolder

# sqrt(2)xsqrt(2)x2 tetragonal supercell of the cubic 5-atom cell.
SCMAT = [[1, -1, 0], [1, 1, 0], [0, 0, 2]]
LABELS = ["pz", "px", "py"] * 12 + ["dz2", "dxy", "dyz", "dx2", "dxz"] * 4
KVECTORS = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
KNAMES = [r"$\Gamma$", "X", "M", r"$\Gamma$", "R"]


def unfold(model_dir, out_path, title):
    model = Wannier90Model(model_dir, "wannier90", scmat=SCMAT)
    u = WannierUnfolder(model, labels=LABELS, sc_matrix=SCMAT)
    ax = u.plot_unfolded_band(kvectors=KVECTORS, knames=KNAMES, npoints=200)
    ax.figure.set_size_inches(7.2, 5.2)
    # weight-coded segments scale with the default width=2; scale up so the
    # weight contrast survives dpi=200
    for lc in ax.collections:
        lc.set_linewidths(np.asarray(lc.get_linewidths()) * 1.8)
    ax.set_title(title)
    ax.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=3,
                   label="unfolded spectral weight"),
            Line2D([0], [0], color="gray", lw=1,
                   label="supercell bands"),
        ],
        loc="upper right", fontsize=8, framealpha=0.85,
    )
    ax.figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


def main(out_path, vacancy_path=None):
    unfold(
        "data_nodefect", out_path,
        "Pristine SrTiO$_3$: $\\sqrt{2}\\times\\sqrt{2}\\times 2$ supercell "
        "unfolded to the 5-atom cubic cell")
    if vacancy_path:
        unfold(
            "data", vacancy_path,
            "Ti-vacancy SrTiO$_3$ supercell unfolded to the 5-atom cubic cell")
    return out_path


if __name__ == "__main__":
    images = os.path.join(ROOT, "docs", "static", "images")
    if len(sys.argv) >= 3:
        out, vac = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        out, vac = sys.argv[1], None   # single-output mode (tests)
    else:
        out = os.path.join(images, "wannier_sto_unfolded.png")
        vac = os.path.join(images, "wannier_sto_ti_vacancy.png")
    print(main(out, vac))
