"""Unfold committed OpenMX Si/Si:P fixtures onto the primitive band path.

Uses the committed fixtures under ``../../tests/data/si_example``
(openmx_si_prim / openmx_si_sc / openmx_si_sc_p .scfout) so the example
runs headless without rerunning OpenMX. Produces
``openmx_si_unfolded.png`` (pristine Si8) and, when run with ``--doped``,
``openmx_si_p_doped.png`` (Si7P donor). Energies are referenced to the
Fermi level of the unfolded run (OpenMX ``ChemP`` from its ``.scfout``):
0 in the figures is E_F.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np

B_DIAMOND = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
PATH = "GXWGLWX"
NPTS = 300

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "tests", "data", "si_example")

# fcc special points in primitive reciprocal fractional coordinates
SPECIAL = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75),
    "L": (0.5, 0.5, 0.5),
}


def band_path(prim_atoms, npts=NPTS):
    """(kpts, x, Xq, knames) over PATH with cartesian-length spacing."""
    bcart = 2 * np.pi * np.linalg.inv(prim_atoms.cell[:].T)
    segs = list(zip(PATH, PATH[1:]))
    lengths = [
        np.linalg.norm((np.array(SPECIAL[b]) - np.array(SPECIAL[a])) @ bcart)
        for a, b in segs
    ]
    total = sum(lengths)
    kpts, x, Xq, knames = [], [], [0.0], [PATH[0]]
    for i, (a, b) in enumerate(segs):
        pa, pb = np.array(SPECIAL[a]), np.array(SPECIAL[b])
        n = max(2, round(npts * lengths[i] / total) + 1)
        t = np.linspace(0.0, 1.0, n, endpoint=i == len(segs) - 1)
        offset = x[-1] if x else 0.0
        seg_x = offset + np.linspace(0.0, lengths[i], n)
        if not x:
            x.extend(list(seg_x))
            kpts.extend(pa + (pb - pa) * t[:, None])
        else:
            x.extend(list(seg_x[1:]))
            kpts.extend(pa + (pb - pa) * t[1:, None])
        Xq.append(seg_x[-1])
        knames.append(b)
    knames = [r"$\Gamma$" if k == "G" else k for k in knames]
    return np.array(kpts), np.array(x), np.array(Xq), knames


def main(scfout, out_png, match_species=True):
    from ase import Atoms
    from unfolding import unfold_openmx

    a = 5.43
    prim = Atoms(
        "Si2",
        scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
        cell=np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]),
        pbc=True,
    )
    kpts, x, Xq, knames = band_path(prim)

    ax = unfold_openmx(
        scfout=scfout,
        prim_atoms=prim,
        unfold_sc_mat=B_DIAMOND,
        kpts=kpts,
        knames=knames,
        xqpts=x,
        Xqpts=Xq,
        match_species=match_species,
        method="ideal",
        output=out_png,
        yrange=(-13.0, 8.0),
    )
    print(f"wrote {out_png}")
    return ax


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "outdir",
        nargs="?",
        default=os.path.join(ROOT, "docs", "static", "images"),
        help="output directory for the figure (default: docs/static/images)",
    )
    ap.add_argument("--doped", action="store_true", help="unfold the Si7P run")
    args = ap.parse_args()
    if args.doped:
        main(
            os.path.join(DATA, "openmx_si_sc_p.scfout"),
            os.path.join(args.outdir, "openmx_si_p_doped.png"),
            match_species=False,
        )
    else:
        main(
            os.path.join(DATA, "openmx_si_sc.scfout"),
            os.path.join(args.outdir, "openmx_si_unfolded.png"),
        )
