#!/usr/bin/env python
"""Generate VASP inputs for the bcc-Fe PAW dense-path unfolding example.

Writes ``inputs/{sc16_scf,sc16_nscf,prim_scf,prim_nscf}``:

- primitive bcc cell (1 atom, a = 2.866 Ang) and its 2x2x2 conventional
  supercell (16 atoms) via M = 2*[[0,1,1],[1,0,1],[1,1,0]] (det 16),
  rows convention A_sc = M @ A_prim;
- a 250-point primitive path Gamma-H-N-Gamma-P-H (ase bandpath), mapped to
  supercell fractional coordinates as K_sc = k_prim @ M.T for the NSCF run;
- SCF charge densities (ISPIN=2, ENCUT 300 eV, Gaussian smearing 0.05 eV,
  the settings of the validated projector seed run) and NSCF ICHARG=11
  band runs with NBANDS = 96 (>= 82 occupied majority bands plus margin).

The POTCAR is licensed: copy it into each run directory yourself from a
private PAW_PBE Fe (06Sep2000) dataset; it is never committed.
"""
import itertools
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.dft.kpoints import bandpath

ACELL = 2.866
PRIM = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * ACELL / 2
MATRIX = 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int)
PATH = "GHNGPH"
NPOINTS = 250
NBANDS = 96

SCF_INCAR = """SYSTEM = bcc Fe {tag} (SCF)
ISTART = 0
ICHARG = 2
ISPIN = 2
MAGMOM = {natoms}*2.2
ENCUT = 300
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6
NELM = 120
NBANDS = {nbands}
LREAL = .FALSE.
PREC = Normal
LWAVE = .FALSE.
LCHARG = .TRUE.
"""

NSCF_INCAR = """SYSTEM = bcc Fe {tag} (NSCF path, ICHARG=11)
ISTART = 0
ICHARG = 11
{extra}ISPIN = 2
MAGMOM = {natoms}*2.2
ENCUT = 300
NBANDS = {nbands}
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6
ISYM = -1
LREAL = .FALSE.
PREC = Normal
LWAVE = .TRUE.
LCHARG = .FALSE.
"""

AUTO_KPOINTS = """Automatic Gamma-centered mesh
0
Gamma
{n} {n} {n}
0 0 0
"""


def primitive_path():
    atoms = Atoms("Fe", cell=PRIM, pbc=True)
    bp = bandpath(PATH, atoms.cell, npoints=NPOINTS)
    x, X, labels = bp.get_linear_kpoint_axis()
    return np.asarray(bp.kpts), x, list(X), labels


def write_poscar(path, lattice, symbols, positions):
    lines = [path.name, "1.0"]
    lines += [" ".join(f"{v:.10f}" for v in row) for row in lattice]
    lines.append(" ".join(symbols))
    lines.append(" ".join(str(len(positions)) for _ in symbols))
    lines.append("Direct")
    lines += [" ".join(f"{v:.10f}" for v in p) for p in positions]
    path.write_text("\n".join(lines) + "\n")


def main(base=None):
    base = Path(base) if base else Path(__file__).resolve().parent / "inputs"
    kprim, x, X, labels = primitive_path()
    ksc = kprim @ MATRIX.T

    sc_scf = base / "sc16_scf"
    sc_nscf = base / "sc16_nscf"
    prim_scf = base / "prim_scf"
    prim_nscf = base / "prim_nscf"
    for d in (sc_scf, sc_nscf, prim_scf, prim_nscf):
        d.mkdir(parents=True, exist_ok=True)

    sc_positions = np.array([
        (i + t0, j + t1, k + t2)
        for t0, t1, t2 in ((0.0, 0.0, 0.0), (.25, .25, .25))
        for i, j, k in itertools.product((0.0, .5), (0.0, .5), (0.0, .5))
    ])
    write_poscar(sc_scf / "POSCAR", MATRIX @ PRIM, ["Fe"], sc_positions)
    write_poscar(sc_nscf / "POSCAR", MATRIX @ PRIM, ["Fe"], sc_positions)
    write_poscar(prim_scf / "POSCAR", PRIM, ["Fe"], [[0, 0, 0]])
    write_poscar(prim_nscf / "POSCAR", PRIM, ["Fe"], [[0, 0, 0]])

    (sc_scf / "INCAR").write_text(SCF_INCAR.format(tag="2x2x2 conv supercell", natoms=16, nbands=NBANDS))
    (prim_scf / "INCAR").write_text(SCF_INCAR.format(tag="primitive cell", natoms=1, nbands=NBANDS))
    # KPAR = 8: k-point parallelism; KPAR 16 crashes this build, and any
    # KPAR makes it write single-precision 45200 WAVECARs with unwrapped
    # k-coordinates (run examples/vasp_fe/wrap_wavecar.py on the result)
    (sc_nscf / "INCAR").write_text(NSCF_INCAR.format(
        tag="2x2x2 conv supercell", natoms=16, nbands=NBANDS,
        extra="KPAR = 8\n"))
    # the primitive bank needs all NBANDS bands in a small basis
    # (~140 plane waves): blocked Davidson hits ZHEGV failures, while
    # the single-shot exact diagonalization converges cleanly
    (prim_nscf / "INCAR").write_text(NSCF_INCAR.format(
        tag="primitive cell", natoms=1, nbands=NBANDS,
        extra="ALGO = Exact\n"))

    (sc_scf / "KPOINTS").write_text(AUTO_KPOINTS.format(n=4))
    (prim_scf / "KPOINTS").write_text(AUTO_KPOINTS.format(n=12))

    lines = [f"Primitive {PATH} path mapped by K_sc = k_prim @ M.T, "
             f"M = {MATRIX.tolist()} (row convention); {len(ksc)} points"]
    lines += [f"{len(ksc)}", "Reciprocal"]
    # this VASP build requires the weight column on explicit k-point lines
    lines += [" ".join(f"{v:.10f}" for v in k) + " 1.0" for k in ksc]
    (sc_nscf / "KPOINTS").write_text("\n".join(lines) + "\n")

    lines = [f"Primitive {PATH} path, {len(kprim)} points", f"{len(kprim)}", "Reciprocal"]
    lines += [" ".join(f"{v:.10f}" for v in k) + " 1.0" for k in kprim]
    (prim_nscf / "KPOINTS").write_text("\n".join(lines) + "\n")

    np.savez(base.parent / "path_axis.npz", kprim=kprim, x=x, X=X, labels=np.array(labels))
    print("wrote", base)
    print("path labels:", labels)


if __name__ == "__main__":
    main()
