#!/usr/bin/env python
"""Generate the GPAW LCAO Si fixtures (data/*.gpw) for this bundle.

Produces the three ``mode='all'`` LCAO restarts the figures need:

    si_prim_lcao.gpw   primitive 2-atom cell, 16x16x16 gamma-centered grid
    si_sc_lcao.gpw     8-atom conventional cell, 8x8x8 gamma-centered grid
    si7p_lcao.gpw      Si7P conventional cell, 8x8x8 gamma-centered grid

Run from the unpacked bundle directory (this file's directory):

    python generate_fixtures.py          # ~2-3 h serial for all three

Requires a recent GPAW (>= 25; tested with gpaw 26). PBE, h=0.17,
symmetry='off'. Existing files are skipped, so the script resumes.

Grid choices (why not coarser): the real-space tables are the inverse
lattice Fourier transform of the k-grid data; a uniform grid cannot
disentangle its +N/2 and -N/2 shells, so the primitive reference needs a
grid resolving the real-space range of its tables (16^3 for the 2-atom
cell keeps the interpolation error ~1e-11 eV; 4^3 leaves ~eV wobble).
The supercell runs use 8^3: their folded mesh coincides exactly with the
16^3 primitive mesh, so SCF densities -- and the eigenvalue reference --
match between the supercell and the primitive overlay.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

A = 5.43  # Angstrom
PRIM_CELL = A * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
CONV_CELL = A * np.eye(3)
SC_SCALED = [
    (0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
    (0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75),
    (0.25, 0.75, 0.75),
]


def primitive_atoms():
    from ase import Atoms

    return Atoms(
        "Si2",
        scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
        cell=PRIM_CELL,
        pbc=True,
    )


def sc_atoms(symbols="Si8"):
    from ase import Atoms
    from ase.symbols import string2symbols

    if len(string2symbols(symbols)) != 8:
        raise ValueError(f"{symbols} does not name 8 atoms")
    return Atoms(symbols, scaled_positions=SC_SCALED, cell=CONV_CELL, pbc=True)


def run_lcao(atoms, kpts, txt):
    from gpaw import GPAW

    atoms.calc = GPAW(
        mode="lcao",
        h=0.17,
        xc="PBE",
        kpts={"size": kpts, "gamma": True},
        symmetry="off",
        txt=txt,
    )
    energy = atoms.get_potential_energy()
    return atoms.calc, energy


def main():
    os.makedirs(DATA, exist_ok=True)

    lcao_jobs = [
        ("si_prim_lcao", "primitive 2-atom LCAO 16x16x16",
         primitive_atoms(), (16, 16, 16)),
        ("si_sc_lcao", "8-atom conventional LCAO 8x8x8",
         sc_atoms(), (8, 8, 8)),
        ("si7p_lcao", "Si7P LCAO 8x8x8", sc_atoms("Si7P"), (8, 8, 8)),
    ]
    for name, label, atoms, kpts in lcao_jobs:
        out = os.path.join(DATA, f"{name}.gpw")
        if os.path.exists(out):
            print(f"[skip] {name}")
            continue
        print(f"[run ] {name}: {label}", flush=True)
        calc, energy = run_lcao(atoms, kpts, os.path.join(DATA, f"{name}.txt"))
        calc.write(out, mode="all")
        print(f"[done] {name}: E={energy:.6f} eV, "
              f"efermi={calc.get_fermi_level():.6f}", flush=True)

    print("fixtures written to", DATA)


if __name__ == "__main__":
    main()
