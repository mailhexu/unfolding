#!/usr/bin/env python
"""Generate the synthetic multi-shell Si k-grid fixture (story 010).

Writes ``si_prim.HSX`` / ``si_sc.HSX`` (plus fdf stubs) for a
Si-diamond tight-binding model (1 orbital per atom, analytic H and S
with a 4.0 A cutoff) on the 2-atom primitive cell and the 8-atom conventional
supercell. The HSX files carry a full 3x3x3 shell set (27 translations
for the SC), so the multi-shell parsing, relabeling and generic-k
ideal-weight paths are exercised on real binary-format data without a
SIESTA license. Re-run to regenerate byte-identical data.
"""
import os

import numpy as np
import sisl

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "si_synth_kgrid")
os.makedirs(HERE, exist_ok=True)
A0 = 5.43


def geometry(which):
    if which == "prim":
        cell = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * A0
        xyz = [[0, 0, 0], [A0 / 4, A0 / 4, A0 / 4]]
    else:
        cell = np.diag([A0, A0, A0])
        xyz = [np.array(o) * A0 for o in
               [[0, 0, 0], [.25, .25, .25], [0, .5, .5], [.25, .75, .75],
                [.5, 0, .5], [.75, .25, .75], [.5, .5, 0], [.75, .75, .25]]]
    g = sisl.Geometry(np.array(xyz), sisl.Atom("Si", R=2.0), lattice=cell)
    g.lattice.set_nsc([3, 3, 3])
    return g


def fill(H):
    pos, cell, n = H.geometry.xyz, H.geometry.cell, len(H.geometry)
    for i in range(n):
        H[i, i, 0] = -2.0
        H[i, i, 1] = 1.0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                isc = (dx, dy, dz)
                off = np.array(isc) @ cell
                for i in range(n):
                    for j in range(n):
                        if i == j and isc == (0, 0, 0):
                            continue
                        d = np.linalg.norm(pos[j] + off - pos[i])
                        if 1e-6 < d < 2.5:  # first-neighbor shell only: complete in both cells
                            w = np.exp(-d / 2.0)
                            H[i, j, 0, isc] = 0.8 * w
                            H[i, j, 1, isc] = 0.12 * w
    return H


FDF_PRIM = """SystemName     Si diamond primitive (synthetic TB model)
SystemLabel    si_prim
NumberOfAtoms  2
NumberOfSpecies 1
%block ChemicalSpeciesLabel
  1  14  Si  Si.psf
%endblock ChemicalSpeciesLabel
LatticeConstant 5.430 Ang
%block LatticeVectors
  0.000  0.500  0.500
  0.500  0.000  0.500
  0.500  0.500  0.000
%endblock LatticeVectors
AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
  0.000  0.000  0.000  1
  0.250  0.250  0.250  1
%endblock AtomicCoordinatesAndAtomicSpecies
"""

FDF_SC = """SystemName     Si diamond 8-atom supercell (synthetic TB model)
SystemLabel    si_sc
NumberOfAtoms  8
NumberOfSpecies 1
%block ChemicalSpeciesLabel
  1  14  Si  Si.psf
%endblock ChemicalSpeciesLabel
LatticeConstant 5.430 Ang
%block LatticeVectors
  1.000  0.000  0.000
  0.000  1.000  0.000
  0.000  0.000  1.000
%endblock LatticeVectors
AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
  0.000  0.000  0.000  1
  0.250  0.250  0.250  1
  0.000  0.500  0.500  1
  0.250  0.750  0.750  1
  0.500  0.000  0.500  1
  0.750  0.250  0.750  1
  0.500  0.500  0.000  1
  0.750  0.750  0.250  1
%endblock AtomicCoordinatesAndAtomicSpecies
"""


def main():
    Hp = fill(sisl.Hamiltonian(geometry("prim"), orthogonal=False))
    Hs = fill(sisl.Hamiltonian(geometry("sc"), orthogonal=False))
    Hp.write(os.path.join(HERE, "si_prim.HSX"))
    Hs.write(os.path.join(HERE, "si_sc.HSX"))
    with open(os.path.join(HERE, "si_prim.fdf"), "w") as f:
        f.write(FDF_PRIM)
    with open(os.path.join(HERE, "si_sc.fdf"), "w") as f:
        f.write(FDF_SC)
    print("fixture written:", HERE)


if __name__ == "__main__":
    main()
