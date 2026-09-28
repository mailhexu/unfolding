#!/usr/bin/env python
"""Reproduce the 'ABINIT DDB phonons' example page.

Unfold phonons from an ABINIT DDB (derivatives database) with
unfolding.DDB_unfolder, which runs ABINIT's anaddb through abipy to get
the supercell eigenvectors along your path and computes the unfolding
weights.  The page documents two systems; this script implements both:

  --system cu      fcc Cu in the CONVENTIONAL cubic cell (natom=4):
                   watch the k-path frame -- ase's get_special_points
                   returns primitive-frame points, abipy interprets
                   qptbounds in the DDB cell's own frame; convert with
                   k_conv = k_prim @ sc_mat (see README.txt).
  --system catio3  CaTiO3 Pnma (20 atoms) unfolded onto the pseudo-cubic
                   cell along Gamma-X-M-Gamma-R, dipdip=0.

Run from the unpacked bundle directory:

    python reproduce.py                    # cu:  data/out_DDB  -> cu_unfolded.png
    python reproduce.py --system catio3    #     data/out.DDB   -> catio3_unfolded.png

Two bundled DDBs (data/out_DDB, data/out.DDB) are the committed example
files behind the published Cu and CaTiO3 figures.  Without a working
anaddb the script still parses the DDB (pure python) and writes a summary
file; the unfolding itself needs anaddb on PATH (or configured in abipy's
manager.yml).
"""
import argparse
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown

import numpy as np

DEFAULT_DDB = {"cu": os.path.join("data", "out_DDB"),
               "catio3": os.path.join("data", "out.DDB")}
DEFAULT_OUT = {"cu": "cu_unfolded.png", "catio3": "catio3_unfolded.png"}

HOWTO_DDB = """\
How to produce a supercell DFPT DDB with ABINIT
-----------------------------------------------
A phonon DDB is written by an ABINIT response-function run:

1. ground-state SCF run for your supercell (iscf 7, occopt/tsmear for
   metals or fixed occ for insulators, k-point grid, ecut ...),
2. DFPT run with optdriver 1:
     rfphon 1          # phonon perturbation
     rfatpol 1 natom   # which atoms to perturb
     rfdir   1 1 1
     nqpt 1
     qpt    0 0 0      # repeat the run for each q-point of your mesh
     prtddb 1          # WRITE THE DDB (out_DDB)
   (a GS run can also emit a DDB with prtddb; merge DDBs with ifc or
    mrgddb when needed.)

anaddb (shipped with ABINIT) then Fourier-interpolates the DDB onto your
path.  Make it visible to abipy: either put anaddb on PATH, or configure
a policy in ~/.abinit/abipy/manager.yml.  See inputs/*.in in this bundle
for runnable deck pairs (cu and CaTiO3) with the parameters recorded in
the shipped DDB headers, and pseudos/ for the CaTiO3 pseudopotentials.
"""


def summarize(ddb_path):
    """Parse the DDB with abipy (pure python, no anaddb) and return text."""
    from abipy.abilab import abiopen

    lines = ["DDB summary: %s" % ddb_path]
    with abiopen(ddb_path) as ddb:
        st = ddb.structure
        h = ddb.header
        abc = st.lattice.abc
        angles = st.lattice.angles
        lines.append("  formula        : %s (natom=%d)"
                     % (st.composition.formula, len(st)))
        lines.append("  lattice a,b,c  : %.6f %.6f %.6f Angstrom" % abc)
        lines.append("  angles ab,bc,ac: %.2f %.2f %.2f deg" % angles)
        lines.append("  q-points in DDB: %d" % len(ddb.qpoints))
        lines.append("  usepaw=%s ecut=%s pawecutdg=%s ixc=%s"
                     % (getattr(h, "usepaw", "?"), getattr(h, "ecut", "?"),
                        getattr(h, "pawecutdg", "?"), getattr(h, "ixc", "?")))
        lines.append("  nkpt=%s ngfft=%s nband=%s"
                     % (getattr(h, "nkpt", "?"), getattr(h, "ngfft", "?"),
                        getattr(h, "nband", "?")))
    text = "\n".join(lines)
    print(text)
    return text


def unfold_cu(ddb_path, out_png):
    """The documented Cu snippet: conventional-cubic-cell DDB.

    The k-path frame chain (see the docs page
    page): ase returns fcc PRIMITIVE-frame special points; abipy reads
    qptbounds in the DDB cell's own (conventional cubic) frame; convert
    with the supercell matrix, k_conv = k_prim @ sc_mat.
    """
    from ase.build import bulk
    from ase.dft.kpoints import get_special_points

    from unfolding.DDB_unfolder import DDB_unfolder

    # conventional cubic cell expressed in primitive fcc lattice units
    # (rows convention S = A_prim @ sc_mat); this symmetric matrix also
    # converts primitive-frame fractional k-points to the DDB frame
    sc_mat = np.linalg.inv(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0)

    atoms = bulk("Cu", "fcc")                      # labels only
    points = get_special_points(atoms.cell, eps=0.01)   # PRIMITIVE frame!
    knames = [r"$\Gamma$", "X", "W", r"$\Gamma$", "L"]
    kpath_prim = [points[k] for k in "GXWGL"]
    kpath_bounds = [np.dot(k, sc_mat) for k in kpath_prim]
    # -> conventional-frame Gamma-X-W-Gamma-L:
    #    [[0,0,0], [0,1,0], [1/2,1,0], [0,0,0], [1/2,1/2,1/2]]

    ax = DDB_unfolder(ddb_path, sc_mat=sc_mat,
                      kpath_bounds=kpath_bounds, knames=knames)
    ax.figure.savefig(out_png, dpi=300, bbox_inches="tight")
    print("wrote %s" % out_png)


def unfold_catio3(ddb_path, out_png):
    """The documented CaTiO3 snippet: Pnma cell -> pseudo-cubic path."""
    from unfolding.DDB_unfolder import DDB_unfolder

    # supercell matrix rows = Pnma axes in pseudo-cubic units
    # A_Pnma = M . A_pc; kpath_bounds in the DDB (Pnma) frame
    ax = DDB_unfolder(ddb_path,
                      sc_mat=[[1, -1, 0], [1, 1, 0], [0, 0, 2]],
                      kpath_bounds=[[0, 0, 0], [0, .5, 0], [.5, .5, 0],
                                    [0, 0, 0], [.5, .5, .5]],
                      knames=[r"$\Gamma$", "X", "M", r"$\Gamma$", "R"],
                      dipdip=0)
    ax.figure.savefig(out_png, dpi=300, bbox_inches="tight")
    print("wrote %s" % out_png)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Unfold phonons from an ABINIT DDB along a k-path "
                    "(abipy + anaddb).")
    parser.add_argument("--system", choices=("cu", "catio3"), default="cu",
                        help="which documented example to run (default %(default)s)")
    parser.add_argument("--ddb", default=None,
                        help="DDB path (default: data/out_DDB for cu, "
                             "data/out.DDB for catio3)")
    parser.add_argument("--output", default=None,
                        help="output figure (default cu_unfolded.png / "
                             "catio3_unfolded.png)")
    parser.add_argument("--summary-only", action="store_true",
                        help="parse the DDB and write the summary only "
                             "(no anaddb needed)")
    args = parser.parse_args(argv)

    ddb = args.ddb or DEFAULT_DDB[args.system]
    out_png = args.output or DEFAULT_OUT[args.system]

    if not os.path.exists(ddb):
        print("ERROR: DDB not found: %s" % ddb)
        print(HOWTO_DDB)
        return 1

    text = summarize(ddb)
    summary_file = "%s_ddb_summary.txt" % args.system
    with open(summary_file, "w") as fh:
        fh.write(text + "\n")
    print("wrote %s" % summary_file)

    if args.summary_only:
        return 0

    if shutil.which("anaddb") is None:
        print("""
anaddb was NOT found on PATH: the unfolding step (anaddb Fourier
interpolation of the DDB, run through abipy) cannot start.  Everything up
to this point succeeded: the DDB was parsed and summarized above.

To run the unfolding:
  * install ABINIT (anaddb ships with it) and put anaddb on PATH, or
    configure it in ~/.abinit/abipy/manager.yml,
  * or point --ddb at your own supercell DFPT DDB (see below).

The published figures on the example page were produced from the bundled
DDBs (data/out_DDB, data/out.DDB) with exactly the calls in this script.
""")
        print(HOWTO_DDB)
        return 0

    if args.system == "cu":
        unfold_cu(ddb, out_png)
    else:
        unfold_catio3(ddb, out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
