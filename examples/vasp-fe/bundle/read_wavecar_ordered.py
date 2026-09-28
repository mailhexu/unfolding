"""WAVECAR reader with G-lists enumerated in VASP's stored k-frame.

HamiltonIO.vasp.read_wavecar delegates to pymatgen's Wavecar, which
enumerates each k-point's G-vectors around the *wrapped* fractional k.
VASP itself iterates its plane-wave lists around the k exactly as stored
in the WAVECAR, and this build (KPAR writer) stores unwrapped components
(>1).  The two lists contain the same physical plane waves but pair them
to the stored coefficients in different orders, which silently corrupts
any overlap that involves the augmentation charge on high-energy bands.

This reader parses the same records but enumerates each sphere around
the stored k with VASP's fold-lexicographic iteration order, verifies
the plane-wave count against the stored record, and returns a
``VaspPWData`` for ``unfolding.vasp_paw``.  Coefficients are read as the
complex64 values VASP writes; eigenvalues are the stored band energies.
"""
from pathlib import Path

import numpy as np

from HamiltonIO.vasp import VaspPWData


def _fold_lex_indices(bound):
    """(3, M) integer grid in VASP/pymatgen iteration order (x fastest)."""
    rng = np.arange(2 * bound + 1)
    fold = np.where(rng <= bound, rng, rng - 2 * bound - 1)
    i3, j2, k1 = np.meshgrid(fold, fold, fold, indexing="ij")
    return np.stack([k1.ravel(), j2.ravel(), i3.ravel()], axis=0).T


def read_wavecar_ordered(path, poscar, *, kpoint_indices=None):
    """Parse a standard (rtag 45200/53300, spin-polarized) WAVECAR.

    ``kpoint_indices`` selects a zero-based subset; the records are still
    streamed through the full file.
    """
    from pymatgen.io.vasp.inputs import Poscar

    path = Path(path)
    fin = open(path, "rb")
    recl, spin, rtag = np.fromfile(fin, dtype=np.float64, count=3).astype(int)
    if rtag not in (45200, 45210, 53300, 53310):
        raise ValueError(f"unsupported WAVECAR rtag {rtag}")
    recl8 = int(recl / 8)
    np.fromfile(fin, dtype=np.float64, count=recl8 - 3)
    nk, nb = np.fromfile(fin, dtype=np.float64, count=2).astype(int)
    encut = float(np.fromfile(fin, dtype=np.float64, count=1)[0])
    lattice = np.fromfile(fin, dtype=np.float64, count=9).reshape(3, 3)
    efermi = float(np.fromfile(fin, dtype=np.float64, count=1)[0])
    np.fromfile(fin, dtype=np.float64, count=recl8 - 13)
    # reciprocal vectors in VASP's convention: rows of 2*pi*inv(A).T
    bcell = 2 * np.pi * np.linalg.inv(lattice).T

    # generous per-axis enumeration bound
    bmag = np.linalg.norm(bcell, axis=1)
    gmax = np.sqrt(encut * 0.262465831) / bmag + 2.0
    structure = Poscar.from_file(str(poscar)).structure

    gvecs, coefficients, eigenvalues, kpoints = [], [], [], []
    for ispin in range(spin):
        for ik in range(nk):
            head = np.fromfile(fin, dtype=np.float64, count=4)
            nplane = int(head[0])
            kpt = head[1:4]
            enocc = np.fromfile(
                fin, dtype=np.float64, count=3 * nb
            ).reshape(nb, 3)[:, 0]
            skip = (recl8 - 4 - 3 * nb) % recl8
            if skip:
                np.fromfile(fin, dtype=np.float64, count=skip)
            rows = np.empty((nb, nplane), dtype=complex)
            for ib in range(nb):
                rows[ib] = np.fromfile(fin, dtype=np.complex64, count=nplane)
                np.fromfile(fin, dtype=np.float64, count=recl8 - nplane)
            if ispin == 0:
                kpoints.append(kpt)
                bound = int(np.ceil(gmax.max() + np.abs(kpt).max()))
                grid = _fold_lex_indices(bound)
                shifted = kpt + grid
                kin = np.einsum("ij,ij->i", shifted @ bcell, shifted @ bcell)
                keep = np.flatnonzero(kin / 0.262465831 <= encut)
                if len(keep) != nplane:
                    raise ValueError(
                        f"{path.name}: k-point {ik} enumerates {len(keep)} "
                        f"plane waves, WAVECAR stores {nplane}"
                    )
                gvecs.append(grid[keep])
                eigenvalues.append(enocc[None, :])
                coefficients.append(rows[None, :, None, :])
            else:
                if not np.allclose(kpt, kpoints[ik], atol=1e-7):
                    raise ValueError(
                        f"{path.name}: spin-1 k-point {ik} differs: {kpt}"
                    )
                if len(gvecs[ik]) != nplane:
                    raise ValueError(
                        f"{path.name}: spin-1 plane-wave count {nplane} "
                        f"differs from spin 0 ({len(gvecs[ik])})"
                    )
                eigenvalues[ik] = np.concatenate([eigenvalues[ik], enocc[None, :]])
                coefficients[ik] = np.concatenate(
                    [coefficients[ik], rows[None, :, None, :]], axis=0
                )
    fin.close()
    if kpoint_indices is None:
        selected = range(nk)
    else:
        selected = [int(i) for i in kpoint_indices]
        if not selected or min(selected) < 0 or max(selected) >= nk:
            raise ValueError("kpoint_indices outside WAVECAR range")
    return VaspPWData(
        lattice,
        np.asarray([kpoints[i] for i in selected]),
        tuple(gvecs[i] for i in selected),
        tuple(coefficients[i] for i in selected),
        tuple(eigenvalues[i] for i in selected),
        efermi,
        tuple(str(site.specie.symbol) for site in structure),
        np.asarray(structure.frac_coords),
    )
