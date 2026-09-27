"""Licensed VASP Fe fixture: assert PAW overlap restores physical normalization.

The POTCAR/WAVECAR are NOT redistributed. Set UNFOLDING_VASP_FE_SEED to a
private directory with matching POSCAR, POTCAR and WAVECAR to exercise this.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from HamiltonIO.vasp import read_potcar_paw, read_wavecar
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
    from HamiltonIO.vasp import VaspPWData

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
