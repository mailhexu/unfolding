#!/usr/bin/env python
"""ABACUS planewave example: unfold the 8-atom conventional cell (PW).

Reads the committed out_wfc_pw/out_band fixture (WAVEFUNC*.txt at 31
k-points along Gamma-X-W-Gamma-L-W-X, ecutwfc 10 Ry, 16 bands), builds
PWEigenData and unfolds with PWUnfolder. Energies are plotted in eV
relative to the Fermi level parsed from the run log, in the house
[-13, 8] eV window. Writes
docs/static/images/abacus_si_pw_unfolded.png.

Usage: python examples/abacus_si_pw/unfold.py [docs/static/images]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# supercell = M @ primitive cell (rows); conv cubic = B @ fcc-primitive
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
# ABACUS Line path in supercell fractional coordinates: G-X-W-G-L-W-X,
# 5 points per segment (the stored fixture has 31 k-points)
KNAMES = [r"$\Gamma$", "X", "W", r"$\Gamma$", "L", "W", "X"]
SEGMENTS = 5


def path_axis(kprim):
    """Path length coordinates and corner ticks for the stored k order."""
    corners = np.arange(0, len(kprim), SEGMENTS)
    if corners[-1] != len(kprim) - 1:
        corners = np.append(corners, len(kprim) - 1)
    x = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(kprim, axis=0), axis=1))]
    )
    Xq = x[corners]
    return x, Xq, KNAMES


def main(images_dir):
    import matplotlib.pyplot as plt

    from HamiltonIO.abacus.pw_wfc import AbacusPWParser
    from unfolding.pw_unfolder import PWEigenData, PWUnfolder

    fixture = os.path.join(
        ROOT, "tests", "data", "abacus_example", "si_pw_path", "OUT.si_pw_path"
    )
    data = AbacusPWParser(fixture).read()
    core = PWEigenData(
        kpoints=data.kpoints,
        gvecs=data.gvecs,
        coefficients=[c[None, :, None, :] for c in data.coefficients],
        eigenvalues=data.eigenvalues[:, None, :],
    )
    unfolder = PWUnfolder(core, M)
    kprim = data.kpoints @ np.linalg.inv(M)  # supercell -> primitive fractional
    res = unfolder.compute(kprim, resolve_degenerate=1e-4)

    # house style: energies in eV, zero at the run's Fermi level
    ek = res.eigenvalues - data.efermi

    x, Xq, knames = path_axis(kprim)
    nb = res.weights.shape[1]
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(ek[:, ib]) for ib in range(nb)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)]

    from unfolding.plotphon import plot_band_weight

    ax = plot_band_weight(
        kslist, ekslist, wkslist,
        xticks=[knames, Xq],
        ylabel="Energy (eV)",
        ypad=1.5,
        efermi=0.0,
        yrange=(-13.0, 8.0),
    )
    ax.set_title("ABACUS Si$_8$ (PW) unfolded onto the primitive cell")
    out_path = os.path.join(images_dir, "abacus_si_pw_unfolded.png")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "docs", "static", "images")
    os.makedirs(out_dir, exist_ok=True)
    print(main(out_dir))
