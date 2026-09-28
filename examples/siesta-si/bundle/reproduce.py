#!/usr/bin/env python
"""Reproduce the SIESTA Si unfolded-band figure (si_unfolded.png).

Documentation: https://mailhexu.github.io/unfolding/examples/siesta-si/
Unfolds the committed 8-atom conventional-cell Si supercell (HSX from a
2x2x2 k-grid SCF) onto the primitive fcc high-symmetry path
Gamma-X-W-Gamma-L-W-X with the generic-k "ideal" weight, and overlays the
independently computed primitive-cell band structure in red. Every
weight-1 unfolded branch lies on a primitive band.

Run from the unpacked bundle directory (this file's directory):

    python reproduce.py [output.png]

Requires the ``unfolding`` package importable (pip install unfolding) plus numpy,
scipy, matplotlib, sisl and HamiltonIO (the SIESTA parser). No SIESTA
run is needed: the fixtures in data/ are committed.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = B @ prim
PATH = "GXWGLWX"
NPTS = 300

# fcc high-symmetry points in primitive reciprocal fractional coordinates
# (Setyawan-Curtarolo values).
SPECIAL = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75),
    "K": (0.375, 0.375, 0.75),
    "L": (0.5, 0.5, 0.5),
    "U": (0.625, 0.25, 0.625),
}


SIESTA_REGENERATE = """
The .HSX Hamiltonian files are missing from data/ (they ship with the
bundle -- did the unpacking drop them?).  Regenerate them from the fdf
inputs with SIESTA >= 4 (the pseudopotential named by the fdf placed
next to it; see pseudos/), e.g. for each fdf F in data/:

    siesta < F.fdf > F.out               # SCF with SaveHS (already in the fdf)
    cp F.fdf F.HSX data/

then re-run this script.
"""

def read_si_model(fdf):
    """Parse a SIESTA fdf (+ sibling .HSX) through the production path.

    Plain SislParser works as shipped: HamiltonIO's read_Rlist takes the
    interaction shells from the Hamiltonian lattice itself.
    """
    from HamiltonIO.siesta.sisl_wrapper import SislParser

    return SislParser(fdf).get_model()


def path_grid(npts=NPTS):
    """k-grid over PATH in primitive fractional coordinates.

    Returns ``(kpts, seg_starts)``; segment point counts are
    proportional to Cartesian length.
    """
    cell = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * 5.430
    bcart = 2 * np.pi * np.linalg.inv(cell).T
    segs = list(zip(PATH, PATH[1:]))
    lengths = [
        np.linalg.norm((np.array(SPECIAL[b]) - np.array(SPECIAL[a])) @ bcart)
        for a, b in segs
    ]
    total = sum(lengths)
    kpts, seg_starts = [], []
    for i, (a, b) in enumerate(segs):
        pa, pb = np.array(SPECIAL[a]), np.array(SPECIAL[b])
        n = max(2, round(npts * lengths[i] / total) + 1)
        last = i == len(segs) - 1
        t = np.linspace(0.0, 1.0, n, endpoint=last)
        seg_starts.append(len(kpts))
        kpts.extend(pa + (pb - pa) * t[:, None])
    return np.array(kpts), seg_starts


def band_path(prim_atoms):
    """Primitive high-symmetry path: (kpts, x, Xq, knames)."""
    cell = np.asarray(prim_atoms.cell)
    bcart = 2 * np.pi * np.linalg.inv(cell).T
    kpts, seg_starts = path_grid()
    d = np.linalg.norm(np.diff(kpts, axis=0) @ bcart, axis=1)
    x = np.concatenate([[0.0], np.cumsum(d)])
    Xq = [x[i] for i in seg_starts] + [x[-1]]
    knames = [{"G": "\u0393"}.get(s, s) for s in PATH]
    return kpts, x, np.array(Xq), knames


def prim_bands(prim, kfrac):
    """Primitive-cell eigenvalues along the path, shape (nk, n_orb_prim)."""
    from scipy.linalg import eigh

    from unfolding.lcao_unfolder import HamiltonIOModel

    pm = HamiltonIOModel(prim)
    return np.array([eigh(*pm.hs_and_eigen(k), eigvals_only=True) for k in kfrac])


def main(out_path):
    import matplotlib.pyplot as plt

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap
    from unfolding.plotphon import plot_band_weight

    _miss = [h for h in ['si_prim.HSX', 'si_sc.HSX'] if not os.path.isfile(os.path.join(DATA, h))]
    if _miss:
        print("Missing HSX fixtures: " + ", ".join(_miss))
        print(SIESTA_REGENERATE)
        raise SystemExit(1)
    prim = read_si_model(os.path.join(DATA, "si_prim.fdf"))
    sc = read_si_model(os.path.join(DATA, "si_sc.fdf"))

    rm = RelabelMap.from_atoms(
        sc.atoms,
        prim.atoms,
        B,
        orb_counts_sc=[4] * 8,
        orb_counts_prim=[4, 4],
    )
    unf = LCAOUnfolder(HamiltonIOModel(sc), rm)

    kpts, x, Xq, knames = band_path(prim.atoms)
    res = unf.compute(kpts, method="ideal")
    eprim = prim_bands(prim, kpts)

    nb = res.weights.shape[1]
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(res.eigenvalues[:, ib]) for ib in range(nb)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)]

    ax = plot_band_weight(
        kslist,
        ekslist,
        wkslist,
        xticks=[knames, Xq],
        ylabel="Energy (eV)",
        ypad=1.5,
    )
    lines = ax.plot(x, eprim, color="crimson", lw=1.0, alpha=0.9, zorder=5)
    lines[0].set_label("primitive-cell bands")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "si_unfolded.png"
    print("wrote", main(out))
