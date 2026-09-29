"""Non-diagonal supercell matrices in the phonopy unfolding route.

Regression: ``unf`` passed the same matrix to the q-point map (which needs
``sc_mat`` under phonopy's ``A_sc = sc_mat^T A_prim`` convention) and to
``phonon_unfolder``'s translation maps (built with ase ``make_supercell``,
which uses the transposed convention).  For non-diagonal matrices the two
differ, the translation orbits get scrambled, and pristine-supercell weights
come out fractional instead of binary.  ``unf`` now passes ``sc_mat.T`` to
the maps.

Case: 1-atom simple-cubic primitive, supercell matrix
M = [[2, 1, 0], [0, 1, 0], [0, 0, 1]]  (det 2, non-symmetric), analytic
nearest-neighbour spring force constants -> flat two-band model.  For a
pristine supercell every mode must unfold with weight 0 or 1: exactly
3 * (det M - 1) zeros and 3 ones per q-point.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from unfolding.phonopy_unfolder import unf

M_NS = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]])


def _make_phonon():
    """2-atom calc cell (M_NS supercell of sc primitive) + spring FC."""
    prim = PhonopyAtoms(symbols=["C"], cell=np.eye(3),
                        scaled_positions=[[0, 0, 0]])
    builder = Phonopy(prim, supercell_matrix=M_NS)
    sc = builder.supercell                      # phonopy ordering, as SPOSCAR
    phonon = Phonopy(PhonopyAtoms(symbols=sc.symbols, cell=sc.cell,
                                  scaled_positions=sc.scaled_positions),
                     supercell_matrix=np.eye(3), primitive_matrix=np.eye(3))
    k = 1.0
    fc = np.zeros((2, 2, 3, 3))
    fc[:, :, range(3), range(3)] = 0.0
    for i in range(3):
        fc[0, 0, i, i] = 2 * k
        fc[1, 1, i, i] = 2 * k
        fc[0, 1, i, i] = -k
        fc[1, 0, i, i] = -k
    phonon.force_constants = fc
    return phonon


def test_nonsymmetric_supercell_matrix_binary_weights():
    phonon = _make_phonon()
    qpoints = np.array([[0.1, 0.2, 0.3], [0.0, 0.25, 0.0], [0.0, 0.0, 0.0]])
    x = np.linspace(0.0, 1.0, len(qpoints))
    ax, result = unf(phonon, sc_mat=M_NS, qpoints=qpoints,
                     knames=["G", "X", "L"], x=list(x), xpts=[0.0, 0.5, 1.0],
                     return_result=True)
    w = result.weights
    assert w.shape == (3, 6)
    assert np.all(np.isfinite(w))
    # pristine supercell: binary weights, 3 ones (target sector) + 3 zeros
    for iq in range(len(qpoints)):
        assert np.allclose(np.sort(w[iq]), [0, 0, 0, 1, 1, 1], atol=1e-8), \
            np.sort(w[iq])
