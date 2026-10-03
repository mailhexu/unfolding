#!/usr/bin/env python
"""Generate the pristine 8-atom GPAW PW path restart for this bundle."""
from pathlib import Path

import numpy as np

from reproduce import A, B, band_path

HERE = Path(__file__).resolve().parent
SC_SCALED = [
    (0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
    (0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75),
    (0.25, 0.75, 0.75),
]


def path_kpoints():
    """Return the primitive path mapped to GPAW supercell coordinates."""
    kpoints, _, _, _ = band_path()
    return [tuple(k) for k in kpoints @ B.T]


def main():
    from ase import Atoms
    from gpaw import GPAW, PW

    data = HERE / "data"
    data.mkdir(exist_ok=True)
    restart = data / "si8_pw.gpw"
    atoms = Atoms(
        "Si8", scaled_positions=SC_SCALED, cell=A * np.eye(3), pbc=True
    )
    atoms.calc = GPAW(
        mode=PW(340.0),
        xc="PBE",
        kpts=path_kpoints(),
        nbands=24,
        symmetry="off",
        txt=str(HERE / "si8_pw.txt"),
    )
    energy = atoms.get_potential_energy()
    atoms.calc.write(str(restart), mode="all")
    print(f"wrote {restart} (E={energy:.6f} eV)")
    return restart


if __name__ == "__main__":
    main()
