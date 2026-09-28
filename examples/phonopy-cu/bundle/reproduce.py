#!/usr/bin/env python
"""Reproduce the 'Phonopy Cu phonons' example page, fully offline.

The bundle needs no DFT run and ships no fixtures: it generates the phonopy
inputs (SPOSCAR + FORCE_CONSTANTS for a 3x3x3 fcc Cu supercell) from a
simple analytic force-constant spring model, then runs exactly the
documented ``phonopy_unfold`` call and writes the weight-coded band figure.

Run from the unpacked bundle directory:

    python reproduce.py

Prerequisites: numpy, matplotlib, ase, spglib, phonopy and the
``unfolding`` package (pip install "unfolding[phonopy]").  Output: FORCE_CONSTANTS, SPOSCAR and
unfolded_band_structure.png in the working directory.
"""
import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown

import numpy as np

# Lattice constant of the example (Angstrom); the published figure used the
# same value.  Frequencies come out in cm^-1 regardless of the spring
# constants' absolute scale through the Cu mass; the spring stiffnesses
# below only set the (approximate) dispersion shape.
A_CU = 3.61
# (shell radius, radial stiffness, transverse stiffness) for the first two
# fcc neighbour shells (12 x a/sqrt(2), 6 x a).  Both positive -> the model
# is mechanically stable, so all phonon frequencies are real.
def build_phonopy_supercell(a):
    """3x3x3 phonopy supercell of primitive fcc Cu (27 atoms)."""
    from ase.build import bulk
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    prim = bulk("Cu", "fcc", a=a)
    unitcell = PhonopyAtoms(
        symbols=prim.get_chemical_symbols(),
        scaled_positions=prim.get_scaled_positions(),
        cell=np.array(prim.cell),
    )
    phonon = Phonopy(unitcell, supercell_matrix=np.diag([3, 3, 3]))
    return phonon


def spring_force_constants(cell, frac, shells):
    """Monatomic fcc force constants from a radial/transverse spring model.

    For every pair (i, j) in the minimum-image convention the pair force
    constant is Phi(r) = k_r * n n^T + k_t * (I - n n^T) with n = r/|r|,
    giving fc[i, j] = -Phi(r_j - r_i) and the acoustic sum rule
    fc[i, i] = -sum_{j != i} fc[i, j].  The result is real symmetric and
    translationally invariant, exactly what a pristine-crystal unfolding
    demo needs.
    """
    natom = len(frac)
    dfrac = frac[None, :, :] - frac[:, None, :]      # (i, j, :)
    dfrac -= np.round(dfrac)                          # minimum image
    dcart = dfrac @ np.array(cell)                    # r_j - r_i
    dist = np.linalg.norm(dcart, axis=-1)

    fc = np.zeros((natom, natom, 3, 3))
    eye = np.eye(3)
    for r_shell, k_r, k_t in shells:
        mask = dist <= r_shell + 1e-4
        np.fill_diagonal(mask, False)
        ii, jj = np.where(mask)
        n_vec = dcart[ii, jj] / dist[ii, jj, None]
        proj = k_r * np.einsum("ni,nj->nij", n_vec, n_vec)
        pair = proj + k_t * (eye - proj)
        fc[ii, jj] = -pair
    idx = np.arange(natom)
    fc[idx, idx] = -fc.sum(axis=1)
    return fc


def write_sposcar(path, supercell):
    from ase import Atoms
    from ase.io import write as ase_write

    atoms = Atoms(
        symbols=supercell.symbols,
        scaled_positions=supercell.scaled_positions,
        cell=np.array(supercell.cell),
        pbc=True,
    )
    ase_write(path, atoms, format="vasp", direct=True)


def generate_inputs(a, fc_path="FORCE_CONSTANTS", sposcar="SPOSCAR"):
    """Write SPOSCAR + FORCE_CONSTANTS from the spring model; sanity-check."""
    from phonopy.file_IO import write_FORCE_CONSTANTS

    a_half = a / np.sqrt(2.0)
    shells = [
        (a_half, 1.00, 0.15),   # 12 nearest neighbours at a/sqrt(2)
        (a,      0.35, 0.05),   # 6 second neighbours at a
    ]
    phonon = build_phonopy_supercell(a)
    sc = phonon.supercell
    fc = spring_force_constants(np.array(sc.cell), sc.scaled_positions, shells)
    phonon.force_constants = fc

    # Sanity check: all Gamma-point frequencies real, 3 acoustic modes ~ 0.
    phonon.run_qpoints([[0.0, 0.0, 0.0]])
    freqs = phonon.get_qpoints_dict()["frequencies"][0]  # THz
    print("Gamma frequencies (THz): min %.4f max %.4f" % (freqs.min(), freqs.max()))
    if freqs.min() < -1e-3:
        print("WARNING: negative Gamma frequencies -> unstable spring model")

    write_sposcar(sposcar, sc)
    write_FORCE_CONSTANTS(phonon.force_constants, filename=fc_path)
    print("wrote %s (%d atoms) and %s" % (sposcar, len(sc.symbols), fc_path))
    return phonon


def unfold(a, out_png, npoints):
    """The documented example: unfold the 3x3x3 supercell onto primitive fcc."""
    from ase.build import bulk
    from ase.dft.kpoints import bandpath, get_special_points

    from unfolding.phonopy_unfolder import phonopy_unfold

    atoms = bulk("Cu", "fcc", a=a)
    points = get_special_points("fcc", atoms.cell, eps=0.01)
    kpts, x, X = bandpath([points[k] for k in "GXWGL"], atoms.cell, npoints)
    ax = phonopy_unfold(
        sc_mat=np.diag([1, 1, 1]),           # cell stored in SPOSCAR: as-is
        unfold_sc_mat=np.diag([3, 3, 3]),    # supercell the FC belong to
        force_constants="FORCE_CONSTANTS",
        sposcar="SPOSCAR",
        qpts=kpts,
        qnames=[r"$\Gamma$", "X", "W", r"$\Gamma$", "L"],
        xqpts=x,
        Xqpts=X,
    )
    ax.figure.savefig(out_png, dpi=300, bbox_inches="tight")
    print("wrote %s" % out_png)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cu phonon unfolding from a self-generated spring-model "
        "FORCE_CONSTANTS (no DFT needed).")
    parser.add_argument("--a", type=float, default=A_CU,
                        help="fcc Cu lattice constant in Angstrom (default %(default)s)")
    parser.add_argument("--npoints", type=int, default=300,
                        help="number of q-points along the path (default %(default)s)")
    parser.add_argument("--output", default="unfolded_band_structure.png",
                        help="output figure (default %(default)s)")
    parser.add_argument("--fc", default="FORCE_CONSTANTS",
                        help="FORCE_CONSTANTS file to write/read (default %(default)s)")
    parser.add_argument("--sposcar", default="SPOSCAR",
                        help="SPOSCAR file to write/read (default %(default)s)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="reuse existing FORCE_CONSTANTS/SPOSCAR")
    args = parser.parse_args(argv)

    if not args.skip_generate:
        t0 = time.time()
        generate_inputs(args.a, args.fc, args.sposcar)
        print("input generation took %.1f s" % (time.time() - t0))
    elif not (os.path.exists(args.fc) and os.path.exists(args.sposcar)):
        parser.error("--skip-generate given but %s/%s missing" % (args.fc, args.sposcar))

    t0 = time.time()
    unfold(args.a, args.output, args.npoints)
    print("unfolding took %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
