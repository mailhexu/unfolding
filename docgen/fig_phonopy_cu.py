#!/usr/bin/env python
"""Regenerate the phonopy Cu unfolded-band figure (docs/static/images/).

Runs the committed example (examples/phonopy: FORCE_CONSTANTS + SPOSCAR,
3x3x3 fcc Cu) through the public ``phonopy_unfold`` API and writes the
weight-coded band figure. Headless (Agg).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "phonopy"))
os.chdir(os.path.join(ROOT, "examples", "phonopy"))

import matplotlib.pyplot as plt
from ase.build import bulk
from ase.dft.kpoints import bandpath, get_special_points

from unfolding.phonopy_unfolder import phonopy_unfold


def main(out_path):
    atoms = bulk("Cu", "fcc", a=3.61)
    points = get_special_points("fcc", atoms.cell, eps=0.01)
    kpts, x, X = bandpath([points[k] for k in "GXWGL"], atoms.cell, 300)
    names = [r"$\Gamma$", "X", "W", r"$\Gamma$", "L"]
    ax = phonopy_unfold(
        sc_mat=np.diag([1, 1, 1]),
        unfold_sc_mat=np.diag([3, 3, 3]),
        force_constants="FORCE_CONSTANTS",
        sposcar="SPOSCAR",
        qpts=kpts,
        qnames=names,
        xqpts=x,
        Xqpts=X,
    )
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "docs", "static", "images", "phonopy_unfolded_band_structure.png")
    print(main(out))
