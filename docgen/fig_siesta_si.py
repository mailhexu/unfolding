#!/usr/bin/env python
"""Regenerate the SIESTA Si unfolded-band figure (docs/static/images/).

Unfolds the committed 8-atom conventional-cell Si supercell (story 010
fixtures, tests/data/si_example) onto the primitive-cell path
Gamma-X-W-Gamma-L-X and writes the weight-coded band figure. Headless
(Agg); no SIESTA run needed (committed HSX fixture).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "tests", "data", "si_example")


def main(out_path):
    import sisl
    from HamiltonIO.siesta.sisl_wrapper import SislParser

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap
    from unfolding.plotphon import plot_band_weight

    class TorusSislParser(SislParser):
        """Rlist from the Hamiltonian file's actual supercell translations."""

        def read_Rlist(self, geom=None):
            return self.ham.lattice.sc_off

    pp = TorusSislParser(os.path.join(DATA, "si_prim.fdf"))
    prim = pp.get_model()
    ps = TorusSislParser(os.path.join(DATA, "si_sc.fdf"))
    sc = ps.get_model()

    # conventional cell = B @ primitive cell (rows)
    B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    rm = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, B, orb_counts_sc=[4] * 8, orb_counts_prim=[4, 4]
    )
    unf = LCAOUnfolder(HamiltonIOModel(sc), rm)

    # Gamma-X-W-Gamma-L-X path of the fcc conventional cell, in
    # primitive reciprocal fractional coordinates
    conv = 0.5  # conventional-cubic units -> primitive units
    pts = {
        "G": np.zeros(3),
        "X": np.array([0.5, 0.0, 0.5]) * conv,
        "W": np.array([0.5, 0.25, 0.75]) * conv,
        "L": np.array([0.5, 0.5, 0.5]) * conv,
    }
    import matplotlib.pyplot as plt

    seg = [("G", "X", 40), ("X", "W", 20), ("W", "G", 40), ("G", "L", 30), ("L", "X", 40)]
    kpts, x, Xq, knames = [], [], [], []
    x0 = 0.0
    for a, b, npts in seg:
        pa, pb = pts[a], pts[b]
        t = np.linspace(0.0, 1.0, npts)
        kpts.extend(pa + (pb - pa) * t[:, None])
        xs = x0 + t * np.linalg.norm(pb - pa)
        x.extend(xs)
        Xq.append(xs[0])
        knames.append(a)
        x0 = xs[-1]
    Xq.append(x0)
    knames.append("X")
    kpts = np.array(kpts)
    x = np.array(x)

    res = unf.compute(kpts)
    nb = res.weights.shape[1]
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(res.eigenvalues[:, ib]) for ib in range(nb)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)]

    from unfolding.plotphon import plot_band_weight

    ax = plot_band_weight(
        kslist,
        ekslist,
        wkslist,
        xticks=[knames, Xq],
        ylabel="Energy (eV)",
        ypad=1.5,
    )
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "docs", "static", "images", "si_unfolded.png")
    print(main(out))
