"""PAW-metric band unfolding from ABINIT ETSF WFKs and JTH PAW XML.

HamiltonIO owns both WFK and XML parsing. Here a primitive-cell pseudo
reference band is embedded into the supercell plane-wave basis, and its
all-electron overlap with every supercell state is evaluated through
``S = 1 + sum_a |p_a> dS_a <p_a|``. The primitive-band Gram accounts for
incomplete/nonorthogonal reference sets. This is a reference-band spectral
projection, not an assertion that a finite primitive band bank is complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from HamiltonIO.abinit import AbinitPawData, read_paw_wfk


@dataclass(frozen=True)
class PawWeights:
    kpoints: np.ndarray
    eigenvalues: np.ndarray  # supercell Hartree, indexed by requested fold
    weights: np.ndarray     # S-metric reference-band projector
    pseudo_weights: np.ndarray  # pseudo-L2 reciprocal-coset fractions
    norm_residuals: np.ndarray  # S-norm of each SC band minus one


def _projector_blocks(data: AbinitPawData, ik: int):
    """Return one (npw,nproj_site) projector block and dS per atom."""
    from abinao.orbital_builder import build_paw_projector_form_factors
    from pypao.libpsp.augmentation import compute_delta_s, compute_reduced_moments
    from pypao.spherical_harmonics import compute_ylm

    wfk = data.wavefunctions
    reduced = wfk.gvecs[ik] + wfk.kpoints[ik]
    cart = reduced @ np.linalg.inv(wfk.rprimd).T  # cycles / Bohr
    q = 2 * np.pi * np.linalg.norm(cart, axis=1)
    volume = abs(np.linalg.det(wfk.rprimd))
    matrices = {}
    blocks, corrections = [], []
    for symbol, position in zip(data.species, data.positions):
        if symbol not in matrices:
            pseudo = data.datasets[symbol]
            ls = [l for _, l in pseudo.channel_labels]
            moment = compute_reduced_moments(
                pseudo.partial_waves_ae, pseudo.partial_waves_pseudo,
                pseudo.grid, 0,
            )
            correction = compute_delta_s(moment, ls)
            channels = build_paw_projector_form_factors(pseudo, q)
            harmonics = compute_ylm(cart, max(ls))
            radial = np.column_stack([
                (4 * np.pi / np.sqrt(volume)) * (1j ** ch["l"])
                * ch["f_q"] * harmonics[:, ch["l"] ** 2 + ch["l"] + ch["m"]]
                for ch in channels
            ])
            matrices[symbol] = radial, correction
        radial, correction = matrices[symbol]
        phase = np.exp(2j * np.pi * (reduced @ position))
        blocks.append(radial * phase[:, None])
        corrections.append(correction)
    return blocks, corrections


def _s_overlap(left, right, projector_blocks, corrections):
    """S overlap with row-major band coefficients (left† S right)."""
    result = left.conj() @ right.T
    for projector, ds in zip(projector_blocks, corrections):
        l = left @ projector
        r = right @ projector
        result += l.conj() @ ds @ r.T
    return result


def _embedded_primitive(primitive, ik, supercell, isk, matrix, spin):
    """Map primitive G+k into the stored supercell plane-wave basis."""
    p, s = primitive.wavefunctions, supercell.wavefunctions
    k = p.kpoints[ik]
    K = s.kpoints[isk]
    reduced = (p.gvecs[ik] + k) @ matrix.T - K
    mapped = np.rint(reduced).astype(int)
    if np.max(np.abs(reduced - mapped)) > 1e-7:
        raise ValueError("primitive k-point does not fold to stored supercell K")
    lookup = {tuple(g): j for j, g in enumerate(s.gvecs[isk])}
    try:
        positions = [lookup[tuple(g)] for g in mapped]
    except KeyError as exc:
        raise ValueError("primitive reference plane wave absent from supercell G sphere") from exc
    coeff = p.coefficients[ik][spin, :, 0, :]
    embedded = np.zeros((len(coeff), len(s.gvecs[isk])), dtype=complex)
    embedded[:, positions] = coeff
    return embedded, positions


def unfold_abinit_paw(supercell, primitive, datasets, matrix, *, spin=0,
                      resolve_degenerate=None):
    """PAW-corrected reference-band weights for every primitive WFK k-point.

    ``supercell`` and ``primitive`` are WFK paths or parsed
    :class:`HamiltonIO.abinit.AbinitPawData`; ``datasets`` maps chemical
    symbols to matching JTH XML paths. The primitive reference may be
    pristine Si while the supercell has a substitutional P: the host
    pseudo states are embedded geometrically, then all overlaps are
    evaluated in the *supercell's* PAW metric. Both WFKs need compatible
    spin channels and commensurate cells. Energies remain Hartree.
    `resolve_degenerate` is a Hartree tolerance; the eigenvalues of
    the reference projector inside each degenerate eigenspace remove
    arbitrary eigenvector-gauge mixing. Their order need not correspond
    across different primitive folds.
    """
    if not isinstance(supercell, AbinitPawData):
        supercell = read_paw_wfk(Path(supercell), datasets)
    if not isinstance(primitive, AbinitPawData):
        primitive = read_paw_wfk(Path(primitive), datasets)
    s, p = supercell.wavefunctions, primitive.wavefunctions
    matrix = np.asarray(matrix, dtype=int)
    if matrix.shape != (3, 3) or not np.allclose(matrix @ p.rprimd, s.rprimd, atol=1e-6):
        raise ValueError("supercell primitive_vectors must equal matrix @ primitive_vectors")
    if s.nspin != p.nspin or not 0 <= spin < s.nspin:
        raise ValueError("incompatible WFK spin channels")
    all_w, all_pseudo, all_e, all_norm = [], [], [], []
    projector_cache = {}
    for ik, k in enumerate(p.kpoints):
        K = np.mod(k @ matrix.T, 1.0)
        distances = np.abs((np.mod(s.kpoints, 1.0) - K + .5) % 1.0 - .5).max(axis=1)
        isk = int(np.argmin(distances))
        if distances[isk] > 1e-7:
            raise ValueError(f"no supercell WFK state at folded momentum {K}")
        bra, sectors = _embedded_primitive(primitive, ik, supercell, isk, matrix, spin)
        ket = s.coefficients[isk][spin, :, 0, :]
        if isk not in projector_cache:
            projector_cache[isk] = _projector_blocks(supercell, isk)
        blocks, corrections = projector_cache[isk]
        gram = _s_overlap(bra, bra, blocks, corrections)
        cross = _s_overlap(bra, ket, blocks, corrections)
        operator = cross.conj().T @ np.linalg.solve(gram, cross)
        weights = operator.diagonal().real.copy()
        if resolve_degenerate is not None:
            if resolve_degenerate < 0:
                raise ValueError("resolve_degenerate must be nonnegative")
            energies = s.eigenvalues[isk][spin]
            boundaries = np.r_[0, np.flatnonzero(np.diff(energies) > resolve_degenerate) + 1,
                               len(energies)]
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                if stop - start > 1:
                    weights[start:stop] = np.linalg.eigvalsh(operator[start:stop, start:stop])
        norm = np.diag(_s_overlap(ket, ket, blocks, corrections)).real
        pseudo = np.abs(ket[:, sectors]) ** 2
        pseudo_weight = pseudo.sum(axis=1) / np.sum(np.abs(ket) ** 2, axis=1)
        all_w.append(weights)
        all_pseudo.append(pseudo_weight)
        all_e.append(s.eigenvalues[isk][spin])
        all_norm.append(norm - 1.0)
    return PawWeights(p.kpoints.copy(), np.asarray(all_e), np.asarray(all_w),
                      np.asarray(all_pseudo), np.asarray(all_norm))
