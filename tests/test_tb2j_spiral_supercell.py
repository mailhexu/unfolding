"""Stationary supercell helices through TB2J and the public unfolding adapter."""

import sys
from pathlib import Path

import numpy as np
import pytest

TB2J_SUPERCELL = Path("/home/hexu/projects/TB2J_dev/TB2J-supercell")
if not (TB2J_SUPERCELL / "TB2J" / "magnon" / "magnon_math.py").exists():
    pytest.skip("TB2J-supercell worktree not available", allow_module_level=True)
# force the supercell worktree even if a sibling test imported another TB2J
for _name in [n for n in sys.modules if n == "TB2J" or n.startswith("TB2J.")]:
    del sys.modules[_name]
sys.path.insert(0, str(TB2J_SUPERCELL))

magnon_math = pytest.importorskip("TB2J.magnon.magnon_math")
if not hasattr(magnon_math, "_cholesky_magnon_hamiltonian"):
    pytest.skip("TB2J build lacks the shared Cholesky frame", allow_module_level=True)

from TB2J.magnon.magnon3 import Magnon  # noqa: E402
from TB2J.magnon.magnon_math import get_rotation_arrays  # noqa: E402
from unfolding import MagnonUnfolder
from unfolding.tb2j_unfold import magnon_eigendata_from_tb2j


J = 1e-3  # eV, pair exchange in each direction; moment 2 mu_B means S=1


def chain(n, j1, j2, order):
    """Two-sided exchanges in the cell-origin gauge, with Cartesian positions."""
    rs = np.arange(-2, 3)
    jr = np.zeros((len(rs), n, n, 3, 3))
    for i in range(n):
        for distance, coupling in ((1, j1), (2, j2)):
            for d in (-distance, distance):
                cell, j = divmod(i + d, n)
                jr[cell + 2, i, j] += coupling * np.eye(3)
    if order == "helix":
        theta = np.arange(n) * np.pi / 3
        moments = 2 * np.column_stack((np.cos(theta), np.sin(theta), np.zeros(n)))
    else:
        signs = np.ones(n) if order == "fm" else (-1.0) ** np.arange(n)
        moments = np.column_stack((np.zeros(n), np.zeros(n), 2 * signs))
    model = Magnon(
        nspin=n, magmom=moments, Rlist=np.column_stack((rs, 0 * rs, 0 * rs)),
        JR=jr, cell=np.diag([n, 1, 1]), _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]), _n=np.array([0.0, 0.0, 1.0]),
        positions=np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))),
    )
    model.set_reference([0, 0, 0], [[0, 0, 1]], [0, 0, 1], magmoms=moments)
    return model


def spiral_dispersion(k):
    # J(k)=2J[cos(k)-cos(2k)/2], J(Q)=3J/2, Q=pi/3.
    # omega=2 sqrt((J(Q)-J(k))*(J(Q)-(J(k+Q)+J(k-Q))/2))/S.
    c = np.cos(k)
    return 2 * J * np.sqrt(2 * (c - 0.5) ** 2 * (1 - c) * (2 + c))


def classical_ring_frequencies(n):
    """Independent real-space Hessian of -2 sum_i,d J_d e_i.e_(i+d)."""
    hp = np.zeros((n, n))
    hz = np.zeros((n, n))
    for i in range(n):
        for d, coupling in ((1, J), (2, -J / 2)):
            j = (i + d) % n
            diagonal = 2 * coupling * np.cos(d * np.pi / 3)
            for h in (hp, hz):
                h[i, i] += diagonal
                h[j, j] += diagonal
            hp[i, j] -= diagonal
            hp[j, i] -= diagonal
            hz[i, j] -= 2 * coupling
            hz[j, i] -= 2 * coupling
    hessian = np.block([[hp, np.zeros_like(hp)], [np.zeros_like(hz), hz]])
    assert np.linalg.eigvalsh(hessian).min() > -1e-14
    dynamics = np.block([[np.zeros_like(hp), np.eye(n)],
                         [-np.eye(n), np.zeros_like(hp)]]) @ hessian
    eigenvalues = np.linalg.eigvals(dynamics)
    assert np.max(np.abs(eigenvalues.real)) < 1e-9
    return np.sort(eigenvalues.imag)[n:]


def test_stable_helix_spectrum_frames_and_unfolding():
    model = chain(6, J, -J / 2, "helix")
    theta = np.arange(6) * np.pi / 3
    moments = model.magmom / 2
    fields = sum(coupling * (np.roll(moments, d, axis=0) + np.roll(moments, -d, axis=0))
                 for d, coupling in ((1, J), (2, -J / 2)))
    np.testing.assert_allclose(np.cross(moments, fields), 0, atol=1e-17)
    u, v = get_rotation_arrays(moments, u=np.array([[0.0, 0.0, 1.0]]))
    tangent = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(6)))
    expected_u = np.exp(1j * theta)[:, None] * (1j * tangent - [0, 0, 1])
    np.testing.assert_allclose(u, expected_u, atol=1e-14)
    np.testing.assert_allclose(v, moments, atol=1e-14)

    # Five magnetic momenta contain every mode of a 30-site ring.
    kpoints = np.column_stack((np.arange(5) / 5, np.zeros((5, 2))))
    states = model.get_magnon_eigenstates(kpoints, include_wavefunctions=True)
    k = 2 * np.pi * (kpoints[:, :1] + np.arange(6)) / 6
    expected = np.sort(spiral_dispersion(k), axis=1)
    np.testing.assert_allclose(states.energies, expected, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(np.sort(states.energies.ravel()),
                               classical_ring_frequencies(30), atol=2e-8)

    matrix = np.diag([6, 1, 1])
    data = magnon_eigendata_from_tb2j(model, kpoints)
    assert data.wavefunctions.shape == (5, 6, 12)
    single = magnon_eigendata_from_tb2j(model, kpoints[1:2])
    overlap = np.sum(single.wavefunctions[0].conj() * data.wavefunctions[1], axis=1)
    np.testing.assert_allclose(np.abs(overlap), 1, atol=1e-7)
    folds = np.column_stack(((kpoints[1, 0] + np.arange(6)) / 6,
                             np.zeros((6, 2))))
    weights = MagnonUnfolder(data, matrix).compute(folds).weights
    np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-10)


def test_stationary_but_unstable_ferromagnetic_helix_is_rejected():
    model = chain(6, J, 0, "helix")
    with pytest.raises(ValueError, match="not positive semidefinite"):
        model.get_magnon_eigenstates([[0, 0, 0]], include_wavefunctions=True)


@pytest.mark.parametrize("order,j1,primitive_sites", [("fm", J, 1), ("afm", -J, 2)])
def test_replicated_collinear_supercells_preserve_bands_and_weights(order, j1, primitive_sites):
    primitive = chain(primitive_sites, j1, 0, order)
    supercell = chain(6, j1, 0, order)
    copies = 6 // primitive_sites
    matrix = np.diag([copies, 1, 1])
    qpoints = np.column_stack(([0.11, 0.23, 0.37], np.zeros((3, 2))))
    super_k = np.mod(qpoints @ matrix.T, 1)
    data = magnon_eigendata_from_tb2j(supercell, super_k)
    result = MagnonUnfolder(data, matrix).compute(qpoints, resolve_degenerate=1e-9)
    primitive_states = primitive.get_magnon_eigenstates(qpoints)
    for i, q in enumerate(qpoints):
        folds = q + np.column_stack((np.arange(copies) / copies, np.zeros((copies, 2))))
        folded_bands = primitive.get_magnon_eigenstates(folds).energies.ravel()
        np.testing.assert_allclose(np.sort(data.energies[i]), np.sort(folded_bands), atol=1e-12)
        on_shell = np.any(np.isclose(data.energies[i, :, None],
                                     primitive_states.energies[i], atol=1e-10, rtol=0), axis=1)
        np.testing.assert_allclose(result.weights[i, on_shell].sum(), primitive_sites, atol=1e-8)
        np.testing.assert_allclose(result.weights[i, ~on_shell], 0, atol=1e-8)
