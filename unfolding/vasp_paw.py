"""PAW-metric projector for standard VASP WAVECAR + matching POTCAR.

HamiltonIO parses both licensed files; this module computes on-site
``S = I + sum_a |p_a> dS_a <p_a|`` and primitive-band reference weights.
The optional primitive calculation must use the same PAW datasets and
commensurate cell. No POTCAR bytes are distributed with unfolding.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from HamiltonIO.vasp import read_potcar_paw, read_wavecar
from .abinit_paw import _s_overlap


@dataclass(frozen=True)
class VaspPawWeights:
    kpoints: np.ndarray
    eigenvalues: np.ndarray  # eV
    weights: np.ndarray
    pseudo_weights: np.ndarray
    norm_residuals: np.ndarray


def _projector_blocks(data, paw, ik):
    """VASP reciprocal nonlocal projector tables in the real-Y convention."""
    from pypao.spherical_harmonics import compute_ylm

    reduced = data.gvecs[ik] + data.kpoints[ik]
    cart = 2 * np.pi * (reduced @ np.linalg.inv(data.lattice).T)  # 1/Angstrom
    q = np.linalg.norm(cart, axis=1)
    volume = abs(np.linalg.det(data.lattice))
    blocks, corrections, radial_cache = [], [], {}
    for symbol, position in zip(data.atom_symbols, data.atom_positions):
        if symbol not in radial_cache:
            dataset = paw[symbol]
            harmonics = compute_ylm(cart, int(dataset.angular_momenta.max()))
            columns = []
            for l, samples in zip(dataset.angular_momenta, dataset.projectors_q):
                radial = np.zeros_like(q)
                inside = q <= dataset.q_grid[-1]
                radial[inside] = CubicSpline(dataset.q_grid, samples,
                                              bc_type="natural")(q[inside])
                for m in range(-int(l), int(l)+1):
                    columns.append((1j ** int(l)) * radial
                                   * harmonics[:, int(l)**2 + int(l) + m])
            radial_cache[symbol] = np.column_stack(columns) / np.sqrt(volume)
        phase = np.exp(2j * np.pi * (reduced @ position))
        blocks.append(radial_cache[symbol] * phase[:, None])
        corrections.append(paw[symbol].overlap_correction)
    return blocks, corrections


def _embed_primitive(primitive, ik, supercell, isk, matrix, spin):
    mapped = (primitive.gvecs[ik] + primitive.kpoints[ik]) @ matrix.T - supercell.kpoints[isk]
    rounded = np.rint(mapped).astype(int)
    if np.max(np.abs(mapped-rounded)) > 1e-7:
        raise ValueError("primitive VASP k-point does not fold into supercell")
    lookup = {tuple(g): j for j, g in enumerate(supercell.gvecs[isk])}
    try:
        positions = [lookup[tuple(g)] for g in rounded]
    except KeyError as exc:
        raise ValueError("primitive VASP G vector absent from supercell cutoff") from exc
    embedded = np.zeros((len(primitive.coefficients[ik][spin]),
                         len(supercell.gvecs[isk])), dtype=complex)
    embedded[:, positions] = primitive.coefficients[ik][spin, :, 0, :]
    return embedded, positions


def unfold_vasp_paw(supercell, primitive, potcar, matrix, *, spin=0,
                    resolve_degenerate=None):
    """PAW reference-band weights for VASP calculations.

    Each ``supercell`` / ``primitive`` is a ``(WAVECAR, POSCAR)`` pair or
    a HamiltonIO ``VaspPWData``. ``potcar`` is the licensed matching POTCAR
    path; it is never persisted. The primitive reference may be a pristine
    host, evaluated in the defect supercell PAW metric. ``resolve_degenerate``
    is an eV tolerance; resolved branch labels can permute independently
    between primitive folds.
    """
    if not hasattr(supercell, "coefficients"):
        supercell = read_wavecar(*supercell)
    if not hasattr(primitive, "coefficients"):
        primitive = read_wavecar(*primitive)
    matrix = np.asarray(matrix, dtype=int)
    if matrix.shape != (3, 3) or not np.allclose(
        matrix @ primitive.lattice, supercell.lattice, atol=1e-5
    ):
        raise ValueError("VASP cell relation does not match supercell matrix")
    if len(supercell.coefficients[0]) != len(primitive.coefficients[0]):
        raise ValueError("VASP spin channels differ")
    paw = {sym: read_potcar_paw(potcar, sym)
           for sym in set(supercell.atom_symbols)}
    weights, pseudo, energies, residuals = [], [], [], []
    cache = {}
    for ik, k in enumerate(primitive.kpoints):
        K = np.mod(k @ matrix.T, 1)
        delta = np.abs((np.mod(supercell.kpoints, 1) - K + .5) % 1 - .5).max(axis=1)
        isk = int(np.argmin(delta))
        if delta[isk] > 1e-7:
            raise ValueError(f"supercell WAVECAR lacks folded k-point {K}")
        bra, sectors = _embed_primitive(primitive, ik, supercell, isk, matrix, spin)
        ket = supercell.coefficients[isk][spin, :, 0, :]
        if isk not in cache:
            cache[isk] = _projector_blocks(supercell, paw, isk)
        blocks, corrections = cache[isk]
        cross = _s_overlap(bra, ket, blocks, corrections)
        gram = _s_overlap(bra, bra, blocks, corrections)
        operator = cross.conj().T @ np.linalg.solve(gram, cross)
        wk = operator.diagonal().real.copy()
        e = supercell.eigenvalues[isk][spin]
        if resolve_degenerate is not None:
            if resolve_degenerate < 0:
                raise ValueError("resolve_degenerate must be nonnegative")
            bounds = np.r_[0, np.flatnonzero(np.diff(e) > resolve_degenerate)+1, len(e)]
            for start, stop in zip(bounds[:-1], bounds[1:]):
                if stop-start > 1:
                    wk[start:stop] = np.linalg.eigvalsh(operator[start:stop,start:stop])
        weights.append(wk)
        pseudo.append(np.sum(np.abs(ket[:, sectors])**2, axis=1)
                      / np.sum(np.abs(ket)**2, axis=1))
        residuals.append(np.diag(_s_overlap(ket, ket, blocks, corrections)).real-1)
        energies.append(e)
    return VaspPawWeights(primitive.kpoints.copy(), np.asarray(energies),
                          np.asarray(weights), np.asarray(pseudo),
                          np.asarray(residuals))


def _fold_lex_indices(bound):
    """(3, M) integer grid in VASP/pymatgen iteration order (x fastest)."""
    rng = np.arange(2 * bound + 1)
    fold = np.where(rng <= bound, rng, rng - 2 * bound - 1)
    i3, j2, k1 = np.meshgrid(fold, fold, fold, indexing="ij")
    return np.stack([k1.ravel(), j2.ravel(), i3.ravel()], axis=0).T


def read_wavecar_ordered(path, poscar, *, kpoint_indices=None):
    """Parse a standard (rtag 45200/53300, spin-polarized) WAVECAR.

    Identical record parsing to HamiltonIO's pymatgen-based reader, but
    each k-point's G-vectors are enumerated around the k *as stored* in
    VASP's own fold-lexicographic order (KPAR builds store unwrapped
    components >1, and pymatgen's wrapped enumeration pairs the same
    plane waves to different coefficients, silently corrupting overlaps
    that involve the augmentation charge on high-energy bands). The
    plane-wave count of every spin-0 k-point is verified against the
    stored record. Returns a ``HamiltonIO.vasp.VaspPWData``; this is the
    reader the vasp-paw route uses (ported from
    ``examples/vasp_fe/read_wavecar_ordered.py``).
    """
    from pathlib import Path

    from pymatgen.io.vasp.inputs import Poscar

    path = Path(path)
    with open(path, "rb") as fin:
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
    if kpoint_indices is None:
        selected = range(nk)
    else:
        selected = [int(i) for i in kpoint_indices]
        if not selected or min(selected) < 0 or max(selected) >= nk:
            raise ValueError("kpoint_indices outside WAVECAR range")
    from HamiltonIO.vasp import VaspPWData

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
