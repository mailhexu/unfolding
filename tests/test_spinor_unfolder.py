"""Story-009 tests: spinor (noncollinear/SOC) unfolding.

The spinor toy interlaces spin within each orbital pair
(``o = 4 i + 2 a + sigma``: SIESTA/HamiltonIO ``::2`` up/down order),
with a spin-mixing on-site term (SOC-like) and a spin-diagonal overlap
``S = S_orbital kron I2``. The unfolder treats orbitals opaquely, so
the sigma,sigma' sums run over coefficient indices only (ADR-003).
"""
import numpy as np
import pytest
from ase import Atoms
from scipy.linalg import eigh

from unfolding.lcao_unfolder import LCAOUnfolder
from unfolding.mapping import RelabelMap
from unfolding.toy_models import toy_lcao_model

N_CELLS = 4
N_SPIN = 2


def _spinor_tables():
    """Doubled primitive tables: interlaced (a, sigma) 4x4 blocks."""
    m = toy_lcao_model()
    H, S = m["H"], m["S"]
    lam = 0.07  # on-site spin mixing (SOC-like): i*lam*sigma_y per orbital
    mix = np.array([[0.0, 1j * lam], [-1j * lam, 0.0]])
    Hs, Ss = {}, {}
    for T in (-1, 0, 1):
        Hs[T] = np.kron(H[T], np.eye(N_SPIN)).astype(complex)
        Ss[T] = np.kron(S[T], np.eye(N_SPIN)).astype(complex)
    # on-site spin mixing on every orbital (block-diagonal in a)
    for a in range(2):
        Hs[0][2 * a:2 * a + 2, 2 * a:2 * a + 2] += mix
    return Hs, Ss


class _SpinorModel:
    """Supercell spinor model: HR/SR dicts + hs_and_eigen (toy form)."""

    def __init__(self, Hs, Ss, atoms):
        L = N_CELLS
        n_sc = 4 * L  # 4 spinor orbitals (2 orbital x 2 spin) per cell
        self.HR, self.SR = {}, {}
        for T in (-1, 0, 1):
            Mh = np.zeros((n_sc, n_sc), complex)
            Ms = np.zeros((n_sc, n_sc), complex)
            for i in range(L):
                for j in range(L):
                    d = T * L + j - i
                    if abs(d) <= 1:
                        Mh[4 * i:4 * i + 4, 4 * j:4 * j + 4] = Hs[d]
                        Ms[4 * i:4 * i + 4, 4 * j:4 * j + 4] = Ss[d]
            if np.any(Mh):
                self.HR[T] = Mh
            if np.any(Ms):
                self.SR[T] = Ms
        self.atoms = atoms

    def hs_and_eigen(self, k):
        k = np.atleast_1d(np.asarray(k, dtype=float))
        n = self.SR[0].shape[0]
        H = np.zeros((n, n), complex)
        S = np.zeros((n, n), complex)
        for T, block in self.HR.items():
            H += block * np.exp(2j * np.pi * k[0] * T)
        for T, block in self.SR.items():
            S += block * np.exp(2j * np.pi * k[0] * T)
        return H, S, eigh(H, S, eigvals_only=True)


@pytest.fixture()
def spinor_unfolder():
    Hs, Ss = _spinor_tables()
    prim = Atoms(
        "Si", positions=[[0.0, 0, 0]], cell=[[1.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]
    )
    sc = prim.repeat((N_CELLS, 1, 1))
    # doubled orbital counts flow through the SAME relabel-map path
    rm = RelabelMap.from_atoms(
        sc, prim, np.diag([N_CELLS, 1, 1]),
        orb_counts_sc=[4] * N_CELLS, orb_counts_prim=[4],
    )
    model = _SpinorModel(Hs, Ss, sc)
    return LCAOUnfolder(model, rm), model


def _independent_spinor_weights(unf, kx):
    """Independent interlaced primitive-table oracle (story-7 formula x2).

    The sigma,sigma' structure: A[(m sigma), j] with the interlace
    o = 4 r' + 2 a' + sigma'; S is spin diagonal so
    Sp_spinor = kron(Sp_orbital, I2) and the weight is the spin sum of
    the spinless dual-form weights.
    """
    Hs, Ss = _spinor_tables()
    K = N_CELLS * kx
    n = 4 * N_CELLS
    Hsc = np.zeros((n, n), complex)
    Ssc = np.zeros((n, n), complex)
    for T in (-1, 0, 1):
        ph = np.exp(2j * np.pi * K * T)
        if T in unf._model.HR:
            Hsc += unf._model.HR[T] * ph
        if T in unf._model.SR:
            Ssc += unf._model.SR[T] * ph
    eps, C = eigh(Hsc, Ssc)
    # interlaced bra overlaps: A[(m sigma), 4 r' + 2 a' + sigma]
    from unfolding.toy_models import toy_lcao_model as _tlm

    S_orb = _tlm()["S"]
    A = np.zeros((2 * N_SPIN, n), complex)
    for m in range(2):
        for sig in range(N_SPIN):
            for j in range(n):
                cp, rest = divmod(j, 4)
                a, sig2 = divmod(rest, 2)
                if sig != sig2:
                    continue  # spin-diagonal overlap
                for r in range(N_CELLS):
                    d = (cp - r + N_CELLS // 2) % N_CELLS - N_CELLS // 2
                    if abs(d) <= 1:
                        A[2 * m + sig, j] += (
                            np.exp(-2j * np.pi * kx * r)
                            / np.sqrt(N_CELLS)
                            * S_orb[d][m, a]
                        )
    # Sp_spinor = kron(Sp_orbital, I2) built from the same classes
    Sp_orb = np.zeros((2, 2), complex)
    for d in (-1, 0, 1):
        Sp_orb += np.exp(2j * np.pi * kx * d) * S_orb[d]
    Spinv = np.kron(np.linalg.inv(Sp_orb), np.eye(N_SPIN))
    Aband = A @ C
    return eps, np.real(np.conj(Aband).T @ Spinv @ Aband).diagonal()


def test_spinor_weights_match_independent_oracle(spinor_unfolder):
    """TEST-003: interlaced sigma,sigma' dual-form oracle at several k."""
    unf, _ = spinor_unfolder
    worst = 0.0
    for kx in (0.25, 0.1, 0.375):
        res = unf.compute(np.array([[kx, 0, 0]]))
        eps_o, W_o = _independent_spinor_weights(unf, kx)
        order = np.lexsort((res.weights[0], np.round(res.eigenvalues[0], 9)))
        order_o = np.lexsort((W_o, np.round(eps_o, 9)))
        worst = max(worst, np.abs(res.weights[0][order] - W_o[order_o]).max())
    assert worst < 1e-10, worst


def test_spinor_pristine_roundtrip_and_sum_rule(spinor_unfolder):
    """TEST-001/002: pristine group weights 0/1; total grid Parseval.

    With SOC every band pair is Kramers degenerate, so individual band
    columns are basis-dependent across k-points (same caveat as the
    story-7 roundtrip test): the basis-invariant statements are the
    degenerate-group weights (exactly 0 or 1) and the total trace of
    the weight matrix over the grid.
    """
    unf, _ = spinor_unfolder
    ks = np.array([[i / N_CELLS, 0, 0] for i in range(N_CELLS)])
    res = unf.compute(ks)
    assert np.abs(res.weights.sum() - res.weights.size / N_CELLS) < 1e-8
    tol_e = 1e-8
    for ik in range(len(ks)):
        e, w = res.eigenvalues[ik], res.weights[ik]
        start = 0
        for i in range(1, len(e) + 1):
            if i == len(e) or e[i] - e[i - 1] > tol_e:
                g = w[start:i].sum()
                assert abs(g - round(g)) < 1e-8, (ik, start, i, g)
                assert g > -1e-10
                start = i
