"""Licensed VASP Fe fixture: assert PAW overlap restores physical normalization.

The POTCAR/WAVECAR are NOT redistributed. Set UNFOLDING_VASP_FE_SEED to a
private directory with matching POSCAR, POTCAR and WAVECAR to exercise this.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from HamiltonIO.vasp import VaspPWData, read_potcar_paw, read_wavecar
from unfolding.vasp_paw import unfold_vasp_paw

SEED = Path(os.environ.get("UNFOLDING_VASP_FE_SEED", ""))
pytestmark = pytest.mark.skipif(
    not os.environ.get("UNFOLDING_VASP_FE_SEED") or
    not all((SEED / file).is_file() for file in ("POSCAR", "POTCAR", "WAVECAR")),
    reason="private licensed VASP fixture unavailable",
)


def test_potcar_projector_augmentation_restores_wavecar_norm():
    paw = read_potcar_paw(SEED / "POTCAR", "Fe")
    assert len(paw.angular_momenta) == 7
    np.testing.assert_allclose(paw.overlap_correction, paw.overlap_correction.T)
    wave = read_wavecar(SEED / "WAVECAR", SEED / "POSCAR",
                        kpoint_indices=[0, 1], band_indices=list(range(8)))
    raw_norm = np.sum(np.abs(wave.coefficients[0][0, :, 0, :])**2, axis=-1)
    assert np.max(np.abs(raw_norm - 1)) > .5  # pseudo-L2 is not PAW normalization
    result = unfold_vasp_paw(wave, wave, SEED / "POTCAR", np.eye(3, dtype=int))
    assert np.max(np.abs(result.norm_residuals)) < 1e-5
    np.testing.assert_allclose(result.weights, 1, atol=1e-5)


def test_elevenfold_embedding_preserves_real_fe_paw_metric():
    """Fold real Fe coefficients onto an 11-site synthetic supercell.

    This is a geometry/metric test, not a claim of an independent VASP
    supercell calculation: one real primitive eigenstate is embedded.
    """
    primitive = read_wavecar(SEED / "WAVECAR", SEED / "POSCAR",
                             kpoint_indices=list(range(11)), band_indices=[0])
    matrix = np.diag([11, 1, 1])
    mapped = [np.rint((g+k) @ matrix.T).astype(int)
              for g, k in zip(primitive.gvecs, primitive.kpoints)]
    vectors = np.unique(np.concatenate(mapped), axis=0)
    indices = {tuple(g): j for j, g in enumerate(vectors)}
    coeff = np.zeros((2, 1, 1, len(vectors)), dtype=complex)
    target = [indices[tuple(g)] for g in mapped[0]]
    for spin in range(2):
        coeff[spin, 0, 0, target] = primitive.coefficients[0][spin, 0, 0]
    supercell = VaspPWData(
        matrix @ primitive.lattice, np.zeros((1, 3)), (vectors,),
        (coeff,), (primitive.eigenvalues[0],), primitive.fermi_energy,
        ("Fe",) * 11,
        np.column_stack([np.arange(11)/11, np.zeros((11, 2))]),
    )
    result = unfold_vasp_paw(supercell, primitive, SEED / "POTCAR", matrix)
    assert np.max(np.abs(result.norm_residuals)) < 1e-6
    assert result.weights[0, 0] == pytest.approx(1, abs=1e-6)
    assert np.max(result.weights[1:, 0]) < 1e-8


def test_full_bank_eye3_weights_and_norms_binary_for_both_spins():
    """A primitive cell folded onto itself has unit sector weights everywhere.

    Beyond the eight-band spin-0 norm check above: every band of both spins
    at sampled k-points recovers weight one in its own sector, and the PAW
    S-norm residual stays at machine-restoration level over s/p/d/f channels.
    """
    wave = read_wavecar(SEED / "WAVECAR", SEED / "POSCAR",
                        kpoint_indices=[0, 1], band_indices=list(range(24)))
    for spin in (0, 1):
        result = unfold_vasp_paw(wave, wave, SEED / "POTCAR",
                                 np.eye(3, dtype=int), spin=spin)
        assert np.max(np.abs(result.weights - 1)) < 1e-4
        assert np.max(np.abs(result.norm_residuals)) < 1e-4


def test_spectral_completeness_over_full_reference_bank():
    """Parseval: projecting the full 24-band bank sums to one per band.

    An 11-site synthetic chain containing every band of one real primitive
    eigenstate copy per fold; the reference projector's column sums must
    reproduce the unit spectral weight of each primitive band.
    """
    primitive = read_wavecar(SEED / "WAVECAR", SEED / "POSCAR",
                             kpoint_indices=list(range(11)),
                             band_indices=list(range(24)))
    matrix = np.diag([11, 1, 1])
    mapped = [np.rint((g + k) @ matrix.T).astype(int)
              for g, k in zip(primitive.gvecs, primitive.kpoints)]
    vectors = np.unique(np.concatenate(mapped), axis=0)
    indices = {tuple(g): j for j, g in enumerate(vectors)}
    coeff = np.zeros((2, 24, 1, len(vectors)), dtype=complex)
    for spin in range(2):
        for b in range(24):
            coeff[spin, b, 0, [indices[tuple(g)] for g in mapped[0]]] = \
                primitive.coefficients[0][spin, b, 0]
    supercell = VaspPWData(
        matrix @ primitive.lattice, np.zeros((1, 3)), (vectors,),
        (coeff,), (primitive.eigenvalues[0],), primitive.fermi_energy,
        ("Fe",) * 11,
        np.column_stack([np.arange(11) / 11, np.zeros((11, 2))]),
    )
    result = unfold_vasp_paw(supercell, primitive, SEED / "POTCAR", matrix)
    np.testing.assert_allclose(result.weights[0], 1, atol=1e-5)


def test_conventional_cell_embedding_preserves_paw_weights():
    """Non-diagonal B fold into a 4-site conventional cell keeps unit weights.

    bcc primitive folded with B = [[-1,1,1],[1,-1,1],[1,1,-1]] (det 4) into
    a cell with Fe at (0,0,0), (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0): the
    same conventional-cell decomposition used for fcc Si elsewhere. General
    stored K representatives (including BZ-negative ones) must embed exactly.
    """
    primitive = read_wavecar(SEED / "WAVECAR", SEED / "POSCAR",
                             kpoint_indices=[133 * t for t in (0, 1, 5, 10)],
                             band_indices=list(range(24)))
    matrix = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    lattice = matrix @ primitive.lattice
    kpts = primitive.kpoints
    folded = kpts @ matrix.T
    positions = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], float)
    gvecs, coeffs, eigs = [], [], []
    for ik in range(len(kpts)):
        mapped = np.rint((primitive.gvecs[ik] + kpts[ik]) @ matrix.T
                         - np.mod(folded[ik], 1)).astype(int)
        gvecs.append(mapped)
        coeff = np.zeros((2, 24, 1, len(mapped)), dtype=complex)
        for spin in range(2):
            for b in range(24):
                coeff[spin, b, 0] = primitive.coefficients[ik][spin, b, 0]
        coeffs.append(coeff)
        eigs.append(primitive.eigenvalues[ik])
    supercell = VaspPWData(lattice, np.mod(folded, 1.0), tuple(gvecs), tuple(coeffs),
                           tuple(eigs), primitive.fermi_energy, ("Fe",) * 4, positions)
    for spin in (0, 1):
        result = unfold_vasp_paw(supercell, primitive, SEED / "POTCAR", matrix, spin=spin)
        assert np.max(np.abs(result.weights - 1)) < 1e-5


def test_resolution_removes_degenerate_gauge_scrambling():
    """Raw diagonal weights scatter under an arbitrary gauge; resolved are binary.

    Two primitive momenta fold onto supercell Gamma with identical band
    energies. Interleaving them into exactly degenerate pairs and rotating
    each pair by a real orthogonal gauge leaves the raw projector diagonal
    fractional, while the eigenvalues of the reference operator inside each
    degenerate eigenspace recover the exact {1, 0} sector populations.
    """
    primitive = read_wavecar(SEED / "WAVECAR", SEED / "POSCAR",
                             kpoint_indices=[0, 1], band_indices=list(range(6)))
    matrix = np.diag([11, 1, 1])
    mapped = [np.rint((g + k) @ matrix.T).astype(int)
              for g, k in zip(primitive.gvecs, primitive.kpoints)]
    vectors = np.unique(np.concatenate(mapped), axis=0)
    indices = {tuple(g): j for j, g in enumerate(vectors)}
    nb = 6
    coeff = np.zeros((2, 2 * nb, 1, len(vectors)), dtype=complex)
    target = [[indices[tuple(g)] for g in mapped[0]],
              [indices[tuple(g)] for g in mapped[1]]]
    rng = np.random.default_rng(3)
    for spin in range(2):
        for b in range(nb):
            coeff[spin, 2 * b, 0, target[0]] = primitive.coefficients[0][spin, b, 0]
            coeff[spin, 2 * b + 1, 0, target[1]] = primitive.coefficients[1][spin, b, 0]
            theta = rng.uniform(0, np.pi / 2)
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
            coeff[spin, [2 * b, 2 * b + 1], 0, :] = \
                rotation @ coeff[spin, [2 * b, 2 * b + 1], 0, :]
    energies = np.empty((2, 2 * nb))
    for spin in range(2):
        energies[spin, 0::2] = primitive.eigenvalues[0][spin]
        energies[spin, 1::2] = primitive.eigenvalues[0][spin]
    supercell = VaspPWData(
        matrix @ primitive.lattice, np.zeros((1, 3)), (vectors,),
        (coeff,), (energies,), primitive.fermi_energy,
        ("Fe",) * 11,
        np.column_stack([np.arange(11) / 11, np.zeros((11, 2))]),
    )
    raw = unfold_vasp_paw(supercell, primitive, SEED / "POTCAR", matrix, spin=0)
    resolved = unfold_vasp_paw(supercell, primitive, SEED / "POTCAR", matrix,
                               spin=0, resolve_degenerate=1e-3)
    scrambled = np.max(np.minimum(np.abs(raw.weights), np.abs(1 - raw.weights)))
    assert scrambled > 0.05
    assert np.max(np.minimum(np.abs(resolved.weights),
                             np.abs(1 - resolved.weights))) < 1e-4


PATH_RUNS = Path(os.environ.get("UNFOLDING_VASP_FE_PATH", ""))
PATH_FILES = (
    "sc16_nscf/WAVECAR", "sc16_nscf/POSCAR",
    "prim_nscf/WAVECAR", "prim_nscf/POSCAR",
    "sc16_scf/OUTCAR", "prim_scf/OUTCAR",
)


def _run_efermi(outcar):
    match = re.search(r"E-fermi\s*:\s*([-\d.]+)", Path(outcar).read_text())
    return float(match.group(1))


@pytest.mark.skipif(
    not os.environ.get("UNFOLDING_VASP_FE_PATH")
    or not os.environ.get("UNFOLDING_VASP_FE_SEED")
    or not all((PATH_RUNS / file).is_file() for file in PATH_FILES),
    reason="private VASP supercell-path fixtures unavailable",
)
def test_supercell_path_unfold_is_sector_binary_and_matches_primitive():
    """Real 16-fold bcc-Fe NSCF along the folded GHNGPH path.

    UNFOLDING_VASP_FE_PATH points at the run tree of examples/vasp_fe
    (make_inputs.py + vasp: sc16_scf/sc16_nscf supercell and prim_scf/
    prim_nscf primitive reference, same POTCAR and ENCUT). The example's
    dense-path reader parses the WAVECAR with G lists enumerated in
    VASP's stored k frame, which pymatgen's wrapped-k assumption gets
    wrong for unwrapped KPAR k-points. Checks: the PAW S-metric restores
    real supercell state norms, resolved weights are sector binary away
    from degeneracies, and weight-1 branches reproduce the directly
    computed primitive dispersion after E_F alignment.
    """
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "examples" / "vasp_fe")
    )
    from make_inputs import MATRIX
    from read_wavecar_ordered import read_wavecar_ordered

    subset = list(range(0, 250, 25))  # ten points spanning GHNGPH
    sc = read_wavecar_ordered(
        PATH_RUNS / "sc16_nscf/WAVECAR", PATH_RUNS / "sc16_nscf/POSCAR",
        kpoint_indices=subset,
    )
    prim = read_wavecar_ordered(
        PATH_RUNS / "prim_nscf/WAVECAR", PATH_RUNS / "prim_nscf/POSCAR",
        kpoint_indices=subset,
    )
    ef_sc = _run_efermi(PATH_RUNS / "sc16_scf/OUTCAR")
    ef_prim = _run_efermi(PATH_RUNS / "prim_scf/OUTCAR")
    for spin in (0, 1):
        result = unfold_vasp_paw(sc, prim, SEED / "POTCAR", np.asarray(MATRIX),
                                 spin=spin, resolve_degenerate=1e-3)
        assert np.max(np.abs(result.norm_residuals)) < 1e-4
        weights = np.clip(result.weights, 0.0, 1.0)
        energies = result.eigenvalues - ef_sc
        window = (energies > -8.0) & (energies < 8.0)
        gap = np.minimum(weights, np.abs(1 - weights))
        assert np.mean(gap[window] > 0.05) < 0.01
        primitive = np.asarray(prim.eigenvalues)[:, spin, :] - ef_prim
        counted = 0
        for ik in range(len(energies)):
            for ib in np.flatnonzero((weights[ik] > 0.9) & window[ik]):
                nearest = np.argmin(np.abs(primitive[ik] - energies[ik, ib]))
                deviation = abs(energies[ik, ib] - primitive[ik, nearest])
                assert deviation < 0.15
                counted += 1
        # ~6 d/sp branches per sampled path point have their fold partner
        # at that momentum; the remaining supercell states belong to the
        # other star points visited elsewhere along the path
        assert counted > 40
