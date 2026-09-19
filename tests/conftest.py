"""Shared fixtures: pristine Cu phonon round-trip and toy LCAO reference data."""
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# the 27 reciprocal-lattice triples {0,1,2}^3: the complete set of unfolding
# sectors for a 3x3x3 supercell at commensurate q-points
G_GRID_333 = np.array([[a, b, c] for a in range(3) for b in range(3) for c in range(3)], dtype=float)


@pytest.fixture(scope="session")
def cu_pristine():
    """Pristine 3x3x3 fcc Cu supercell phonons from the in-repo FORCE_CONSTANTS.

    Returns a dict with qpoints (27 commensurate points), frequencies
    (27, 81), eigenvectors (27, 81, 81) and a ready `unfolder`
    (phonon_unfolder) built with the same conventions as the working
    examples (supercell-as-unitcell, supercell_matrix=3I, phase=False).
    """
    pytest.importorskip("phonopy")
    from phonopy import Phonopy
    from phonopy.file_IO import parse_FORCE_CONSTANTS
    from phonopy.structure.atoms import PhonopyAtoms

    from unfolding.phonon_unfolder import phonon_unfolder

    spos = [[i / 3, j / 3, k / 3] for i in range(3) for j in range(3) for k in range(3)]
    scell = PhonopyAtoms(
        symbols=["Cu"] * 27,
        cell=[[0, 5.415, 5.415], [5.415, 0, 5.415], [5.415, 5.415, 0]],
        scaled_positions=spos,
    )
    ph = Phonopy(scell, supercell_matrix=np.eye(3), primitive_matrix=np.eye(3))
    ph.force_constants = parse_FORCE_CONSTANTS(REPO_ROOT / "examples" / "phonopy" / "FORCE_CONSTANTS")
    qs = np.array([np.array([i, j, k]) / 3.0 for i in range(3) for j in range(3) for k in range(3)])
    ph.run_qpoints(qs, with_eigenvectors=True)
    d = ph.get_qpoints_dict()

    from ase.atoms import Atoms

    sc_ase = Atoms(symbols=scell.symbols, cell=scell.cell, scaled_positions=scell.scaled_positions)
    uf = phonon_unfolder(sc_ase, np.diag([3, 3, 3]), d["eigenvectors"], qs, phase=False)
    return {
        "qpoints": qs,
        "frequencies": d["frequencies"],
        "eigenvectors": d["eigenvectors"],
        "unfolder": uf,
    }
