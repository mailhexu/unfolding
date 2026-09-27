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
