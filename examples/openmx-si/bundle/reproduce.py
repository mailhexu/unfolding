#!/usr/bin/env python
"""Reproduce the OpenMX Si unfolded-band figure (openmx_si_unfolded.png).

Docs example: docs/content/examples/openmx-si.md in the unfolding repo.
Unfolds the committed 8-atom conventional-cubic Si supercell (``.scfout``
from an OpenMX run with ``HS.fileout on``) onto the 2-atom fcc primitive
cell along the high-symmetry path Gamma-X-W-Gamma-L-W-X, and overlays the
independently diagonalized OpenMX primitive-cell bands (crimson). Every
weight-1 unfolded branch lies on a primitive band: that is the practical
validation of the OpenMX real-space-Hamiltonian conventions (image
translations atv_ijk, Hartree->eV and Bohr->Angstrom conversion, ChemP
referencing). ``--doped`` regenerates openmx_si_p_doped.png the same way
(dopant mapped onto the host site; pristine primitive bands overlaid on
the doped run's Fermi zero).

Run from the unpacked bundle directory (this file's directory):

    python reproduce.py                # pristine Si8 -> openmx_si_unfolded.png
    python reproduce.py --doped        # Si7P -> openmx_si_p_doped.png

Requires the ``unfolding`` package importable (pip install -e <unfolding
repo checkout>, or PYTHONPATH=<unfolding repo checkout>) plus numpy,
matplotlib, ase, scipy and HamiltonIO (the OpenMX scfout parser). No
OpenMX run is needed for the shipped figures: data/ carries the
committed scfouts.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = B @ prim
PATH = "GXWGLWX"
NPTS = 300
A = 5.43  # Angstrom, conventional cubic lattice constant

# fcc high-symmetry points in primitive reciprocal fractional coordinates
# (Setyawan-Curtarolo values)
SPECIAL = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75),
    "L": (0.5, 0.5, 0.5),
}

DOPED_REGENERATE = """
The Si7P scfout is NOT shipped (bundle size cap). Produce it with OpenMX
from the included input: copy inputs/openmx_si_sc_p.dat next to your
OpenMX pseudopotentials (Si_PBE19.vps and P_PBE19.vps from the OpenMX
2019 data set, DATA.PATH in the .dat), run

    openmx openmx_si_sc_p.dat

and copy the resulting openmx_si_sc_p.scfout into data/. Then re-run
    python reproduce.py --doped
"""


def _model(name):
    from HamiltonIO.openmx import OpenmxParser

    return OpenmxParser(os.path.join(DATA, name)).get_model()


def band_path(prim_cell, npts=NPTS):
    """(kpts, x, Xq, knames) over PATH with Cartesian-length spacing."""
    bcart = 2 * np.pi * np.linalg.inv(prim_cell.T)
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


def prim_bands(prim, kfrac):
    """Primitive-cell eigenvalues along the path, shape (nk, n_orb_prim)."""
    from scipy.linalg import eigh

    from unfolding.lcao_unfolder import HamiltonIOModel

    pm = HamiltonIOModel(prim)
    return np.array([eigh(*pm.hs_and_eigen(k), eigvals_only=True) for k in kfrac])


def main(sc_name, out_png, match_species=True, atol=1e-6):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap
    from unfolding.plotphon import plot_band_weight

    prim = _model("openmx_si_prim.scfout")
    sc = _model(sc_name)
    # 13 orbitals per atom: Si7.0-s2p2d1 (and P7.0-s2p2d1 for the dopant)
    rm = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, B,
        orb_counts_sc=[13] * 8, orb_counts_prim=[13, 13],
        match_species=match_species,
    )
    # each panel is referenced to its own run's OpenMX ChemP; the
    # primitive overlay is shifted by the same value
    ef = sc.efermi
    unf = LCAOUnfolder(HamiltonIOModel(sc), rm)

    kpts, x, Xq, knames = band_path(prim.atoms.cell.array)
    res = unf.compute(kpts, method="ideal", atol_orth=atol, atol_imag=atol)
    res.eigenvalues = res.eigenvalues - ef  # 0 in the figure is E_F
    eprim = prim_bands(prim, kpts) - ef

    nb = res.weights.shape[1]
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(res.eigenvalues[:, ib]) for ib in range(nb)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)]

    ax = plot_band_weight(
        kslist, ekslist, wkslist,
        efermi=ef,
        yrange=(-13.0, 8.0),  # same window as the SIESTA figures
        xticks=[knames, Xq],
        ylabel="Energy (eV)",
        ypad=1.5,
    )
    lines = ax.plot(x, eprim, color="crimson", lw=1.0, alpha=0.9, zorder=5)
    lines[0].set_label("OpenMX primitive-cell bands")
    # plot_band_weight draws the weight-coded bands as blue lines with
    # alpha = w/(width + 0.011); the proxy matches the weight-1 appearance.
    unfolded = Line2D([0], [0], color="blue", alpha=0.5, lw=2,
                      label="unfolded spectral weight")
    ax.legend(handles=[unfolded, lines[0]], loc="upper right", fontsize=8,
              framealpha=0.85)
    ax.figure.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print("wrote", out_png)
    return out_png


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doped", action="store_true", help="unfold the Si7P run")
    ap.add_argument("out", nargs="?", default=None, help="output PNG path")
    args = ap.parse_args()

    if args.doped:
        if not os.path.isfile(os.path.join(DATA, "openmx_si_sc_p.scfout")):
            print("missing", os.path.join(DATA, "openmx_si_sc_p.scfout"))
            print(DOPED_REGENERATE)
            raise SystemExit(1)
        out = args.out or "openmx_si_p_doped.png"
        main("openmx_si_sc_p.scfout", out, match_species=False)
    else:
        out = args.out or "openmx_si_unfolded.png"
        main("openmx_si_sc.scfout", out)
