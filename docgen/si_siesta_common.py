"""Shared setup for the SIESTA Si docgen figures.

Unfolds committed 8-atom conventional-cell Si fixtures
(tests/data/si_example) onto the primitive-cell high-symmetry path
Gamma-X-W-Gamma-L-W-X (fcc points, primitive reciprocal fractional
coordinates via ase). Headless (Agg); no SIESTA run needed (committed
HSX fixtures).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "tests", "data", "si_example")

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = B @ prim
PATH = "GXWGLWX"
NPTS = 300

# fcc high-symmetry points in primitive reciprocal fractional coordinates
# (Setyawan-Curtarolo values; cartesian check: X=(0,1,0)*2pi/a etc.)
SPECIAL = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75),
    "K": (0.375, 0.375, 0.75),
    "L": (0.5, 0.5, 0.5),
    "U": (0.625, 0.25, 0.625),
}


def si_models(sc_fdf="si_sc.fdf"):
    """(prim model, sc model) parsed through the production SIESTA path."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from siesta_helpers import read_si_model

    prim = read_si_model(os.path.join(DATA, "si_prim.fdf"))
    sc = read_si_model(os.path.join(DATA, sc_fdf))
    return prim, sc


def make_unfolder(sc, prim, match_species=True):
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    rm = RelabelMap.from_atoms(
        sc.atoms,
        prim.atoms,
        B,
        orb_counts_sc=[4] * 8,
        orb_counts_prim=[4, 4],
        match_species=match_species,
    )
    return LCAOUnfolder(HamiltonIOModel(sc), rm)


def band_path(prim_atoms):
    """Primitive high-symmetry path: (kpts, x, Xq, knames).

    The fcc special points are exact rational coordinates in the
    primitive reciprocal basis (phase convention e^{-2 pi i k.f});
    derived from the conventional cubic coordinates g via
    c = (g + sum(g)) / 4. The supercell momenta used internally by the
    unfolder are K = scmat^T k, as everywhere in the package.
    """
    cell = np.asarray(prim_atoms.cell)
    bcart = 2 * np.pi * np.linalg.inv(cell).T
    segs = []
    for a, b in zip(PATH, PATH[1:]):
        pa = np.array(SPECIAL[a])
        pb = np.array(SPECIAL[b])
        segs.append((a, b, np.linalg.norm((pb - pa) @ bcart)))
    total = sum(d for *_, d in segs)

    kpts, x, Xq, knames = [], [], [], [PATH[0]]
    x0 = 0.0
    for i, (a, b, d) in enumerate(segs):
        pa = np.array(SPECIAL[a])
        pb = np.array(SPECIAL[b])
        n = max(2, round(NPTS * d / total) + 1)
        last = i == len(segs) - 1
        t = np.linspace(0.0, 1.0, n, endpoint=last)
        kpts.extend(pa + (pb - pa) * t[:, None])
        xs = x0 + t * d
        x.extend(xs)
        Xq.append(xs[0])
        knames.append(b)
        x0 = xs[-1]
    Xq.append(x0)
    knames = [{"G": "Γ"}.get(k, k) for k in knames]
    return np.array(kpts), np.array(x), np.array(Xq), knames


def prim_bands(prim, kfrac):
    """Primitive-cell eigenvalues along the path, shape (nk, n_orb_prim)."""
    from scipy.linalg import eigh

    from unfolding.lcao_unfolder import HamiltonIOModel

    pm = HamiltonIOModel(prim)
    return np.array([eigh(*pm.hs_and_eigen(k), eigvals_only=True) for k in kfrac])
