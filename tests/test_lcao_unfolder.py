"""LCAOUnfolder tests: toy supercell model vs independent dual-basis oracle.

The toy is the story-2 two-orbital LCAO chain folded onto an L-cell
supercell. The reference weights are computed with the independent
full-sector-Gram dual-basis machinery of story 4 (ket vectors over the
AO basis), so the comparison seals Eq. 25's implementation end to end.
"""
import numpy as np
import pytest
from ase import Atoms
from scipy.linalg import eigh

from unfolding.lcao_unfolder import LCAOUnfolder
from unfolding.mapping import RelabelMap
from unfolding.toy_models import toy_lcao_model

N_CELLS = 4
N_ORB = 2


def _prim_tables():
    """Story-2 real-space primitive tables: R in {-1, 0, 1}."""
    m = toy_lcao_model()
    return m["H"], m["S"]


def _toy_supercell_model():
    """Fold the story-2 primitive tables onto an N_CELLS supercell.

    Supercell orbital o = 2*i + a (atom i = 0..L-1, orbital a = 0..1).
    SR[T][o, o'] = S_prim[T*L + i' - i][a, a'] restricted to |displ| <= 1.
    """
    H, S = _prim_tables()
    L = N_CELLS
    HR, SR = {}, {}
    for T in (-1, 0, 1):
        Mh = np.zeros((2 * L, 2 * L))
        Ms = np.zeros((2 * L, 2 * L))
        for i in range(L):
            for j in range(L):
                d = T * L + j - i
                if abs(d) <= 1:
                    Mh[2 * i:2 * i + 2, 2 * j:2 * j + 2] = H[d]
                    Ms[2 * i:2 * i + 2, 2 * j:2 * j + 2] = S[d]
        if np.any(Mh):
            HR[T] = Mh
        if np.any(Ms):
            SR[T] = Ms

    a0 = 1.0
    prim = Atoms(
        "Si", positions=[[0.0, 0, 0]], cell=[[a0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]
    )
    sc = prim.repeat((L, 1, 1))

    class ToyModel:
        """Thin model wrapper: HR/SR dicts + hs_and_eigen (no mutation)."""

        def __init__(self, HR, SR):
            self.HR = HR
            self.SR = SR
            self.atoms = sc

        def hs_and_eigen(self, k):
            k = np.atleast_1d(np.asarray(k, dtype=float))
            H = np.zeros((2 * L, 2 * L), complex)
            S = np.zeros((2 * L, 2 * L), complex)
            for T, block in HR.items():
                H += block * np.exp(2j * np.pi * k[0] * T)
            for T, block in SR.items():
                S += block * np.exp(2j * np.pi * k[0] * T)
            return H, S, eigh(H, S, eigvals_only=True)

    return ToyModel(HR, SR)


@pytest.fixture()
def unfolder():
    model = _toy_supercell_model()
    prim = Atoms(
        "Si", positions=[[0.0, 0, 0]], cell=[[1.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]
    )
    sc = prim.repeat((N_CELLS, 1, 1))
    rm = RelabelMap.from_atoms(
        sc, prim, np.diag([N_CELLS, 1, 1]),
        orb_counts_sc=[2] * N_CELLS, orb_counts_prim=[2],
    )
    snap = {T: b.copy() for T, b in model.SR.items()}
    unf = LCAOUnfolder(model, rm)
    yield unf, model
    # no mutation of the model object
    for T, b in model.SR.items():
        assert np.array_equal(snap[T], b)


def test_weights_match_independent_primitive_table_oracle(unfolder):
    """End-to-end seal: SR-fold + relabel + phases vs direct primitive tables.

    The reference evaluates W(k) = A^dag S_p(k)^{-1} A directly from the
    story-2 primitive S(R) tables (no supercell folding, no rep map), a
    fully independent data path from the LCAOUnfolder machinery.
    """
    unf, _ = unfolder
    ks = np.array([[i / N_CELLS, 0, 0] for i in range(N_CELLS)])
    res = unf.compute(ks)
    H, S = _prim_tables()

    worst = 0.0
    for ik, k in enumerate(ks):
        K = np.diag([N_CELLS, 1, 1]).T @ k
        Hsc, Ssc, _ = unf._model.hs_and_eigen(K)
        eps, C = eigh(Hsc, Ssc)
        # direct primitive-table path (k-state normalization 1/sqrt(L))
        A = np.zeros((N_ORB, 2 * N_CELLS), complex)
        Sp = np.zeros((N_ORB, N_ORB), complex)
        for m in range(N_ORB):
            for j in range(2 * N_CELLS):
                jj, a = divmod(j, 2)
                for r in range(N_CELLS):
                    d = (jj - r + N_CELLS // 2) % N_CELLS - N_CELLS // 2
                    if abs(d) <= 1:
                        A[m, j] += (
                            np.exp(-2j * np.pi * k[0] * r)
                            / np.sqrt(N_CELLS)
                            * S[d][m, a]
                        )
        for n in range(N_ORB):
            for m in range(N_ORB):
                for d in range(-1, 2):
                    Sp[n, m] += np.exp(2j * np.pi * k[0] * d) * S[d][n, m]
        A = A @ C  # band-projected overlaps <k m|psi_J>
        w_ref = np.real(np.conj(A).T @ np.linalg.inv(Sp) @ A).diagonal()
        worst = max(worst, np.abs(w_ref - res.weights[ik]).max())
    assert worst < 1e-10, worst


def test_pristine_roundtrip_and_sum_rule(unfolder):
    unf, _ = unfolder
    ks = np.array([[i / N_CELLS, 0, 0] for i in range(N_CELLS)])
    res = unf.compute(ks)
    # pristine: per degenerate eigenspace, the group weight over the
    # unfolding grid is exactly one 1 (the folded state) and 0 elsewhere.
    # (Individual band columns are basis-dependent when eigenvalues from
    # different k's cross; the group weight is rotation invariant.)
    tol_e = 1e-8
    grouped = np.zeros_like(res.weights)
    n_groups_per_k = []
    for ik in range(len(ks)):
        e = res.eigenvalues[ik]
        w = res.weights[ik]
        groups = []
        start = 0
        for i in range(1, len(e) + 1):
            if i == len(e) or e[i] - e[i - 1] > tol_e:
                groups.append((start, i))
                start = i
        n_groups_per_k.append(len(groups))
        for gi, (a, b) in enumerate(groups):
            grouped[ik, gi] = w[a:b].sum()
    # each (k, group) weight is exactly 0 or 1 (pristine folding)
    assert np.abs(grouped - np.round(grouped)).max() < 1e-8
    assert grouped.min() > -1e-10
    # at every k at least one eigenspace folds with weight 1
    for ik in range(len(ks)):
        assert grouped[ik].max() > 1 - 1e-8, ik


def test_degenerate_averaging(unfolder):
    unf, model = unfolder
    ks = np.array([[0.25, 0, 0]])
    # build a model with an exactly degenerate manifold:
    # H' = S + |v><v| shifts one state; the orthogonal complement stays at eps=1
    H, S, _ = model.hs_and_eigen(np.array([1.0, 0, 0]))
    v = np.zeros(2 * N_CELLS, complex)
    v[0] = 1.0
    H2 = S + np.outer(v, np.conj(v))

    class DegModel:
        def __init__(self, HR, SR, H2):
            self.HR = HR
            self.SR = SR
            self._H2 = H2

        def hs_and_eigen(self, k):
            return self._H2, S, None

    deg_unf = LCAOUnfolder(DegModel(model.HR, model.SR, H2), unf._rm)
    res = deg_unf.compute(ks)
    raw = res.weights[0]
    # without averaging the complement bands all carry weight ~1/|complement|
    # with degenerate averaging they collapse into one entry holding their sum
    avg = res.average_degenerate(tol_e=1e-8)
    # entries: shifted band (weight ~1 at its slot) + one collapsed manifold
    n_groups = avg.weights.shape[1]
    assert n_groups < 2 * N_CELLS
    # the collapsed manifold weight equals the sum of the raw members
    e0 = res.eigenvalues[0]
    deg = np.abs(e0 - 1.0) < 1e-8
    assert avg.weights[0].max() <= raw[deg].sum() + 1e-10
    # total spectral weight conserved by the collapse
    assert abs(avg.weights[0].sum() - raw.sum()) < 1e-10


def test_fold_row_convention_nonsymmetric():
    # K = k @ scmat.T (row convention): each axis folds with its own
    # replication factor; scmat.T @ k would be wrong for nonsymmetric
    # supercell matrices
    prim = Atoms("Si", positions=[(0, 0, 0)], cell=np.eye(3))
    rm = RelabelMap.from_atoms(prim.repeat((2, 1, 1)), prim, np.diag([2, 1, 1]))
    k = np.array([0.1, 0.3, 0.0])
    assert np.allclose(unf_fold(rm, k), k @ rm.scmat.T)


def unf_fold(rm, k):
    from unfolding.lcao_unfolder import LCAOUnfolder

    class _Stub:
        SR = {0: np.eye(2)}
        HR = {0: np.eye(2)}

    return LCAOUnfolder(_Stub(), rm)._fold(k)


def test_hamiltonio_adapter():
    """The (Rlist, SR-array, HS_and_eigen)->4-values HamiltonIO surface
    adapts to the backend-neutral interface and produces identical
    weights."""
    from unfolding.lcao_unfolder import HamiltonIOModel

    model = _toy_supercell_model()
    rlist = list(model.SR.keys())
    sr_arr = np.stack([model.SR[T] for T in rlist])

    class FakeHamiltonIOChannel:
        """HamiltonIO/SislParser-shaped single spin channel."""

        def __init__(self):
            self.Rlist = rlist
            self.SR = sr_arr

        def HS_and_eigen(self, kpts):
            k = np.asarray(kpts, dtype=float).ravel()
            H = np.zeros((2 * N_CELLS, 2 * N_CELLS), complex)
            S = np.zeros((2 * N_CELLS, 2 * N_CELLS), complex)
            for T, block in model.HR.items():
                H += block * np.exp(2j * np.pi * k[0] * T)
            for T, block in model.SR.items():
                S += block * np.exp(2j * np.pi * k[0] * T)
            return H, S, eigh(H, S, eigvals_only=True), None

    prim = Atoms(
        "Si", positions=[[0.0, 0, 0]], cell=[[1.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]
    )
    sc = prim.repeat((N_CELLS, 1, 1))
    rm = RelabelMap.from_atoms(
        sc, prim, np.diag([N_CELLS, 1, 1]),
        orb_counts_sc=[2] * N_CELLS, orb_counts_prim=[2],
    )
    adapted = HamiltonIOModel(FakeHamiltonIOChannel())
    unf = LCAOUnfolder(adapted, rm)
    ks = np.array([[0.25, 0, 0], [0.5, 0, 0]])
    res = unf.compute(ks)
    ref = LCAOUnfolder(model, rm).compute(ks)
    assert np.abs(res.weights - ref.weights).max() < 1e-12


def test_defect_supercell_sum_rule(unfolder):
    """A symmetry-broken overlap (defect) still satisfies the per-state
    grid sum rule through the position-resolved sector Gram."""
    model = _toy_supercell_model()
    prim = Atoms(
        "Si", positions=[[0.0, 0, 0]], cell=[[1.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]
    )
    sc = prim.repeat((N_CELLS, 1, 1))
    rm = RelabelMap.from_atoms(
        sc, prim, np.diag([N_CELLS, 1, 1]),
        orb_counts_sc=[2] * N_CELLS, orb_counts_prim=[2],
    )
    # break primitive translation symmetry: one supercell orbital's onsite
    # overlap shifted (a local defect in the position-resolved S)
    model.SR[0][4, 4] += 0.15
    model.HR[0][4, 4] += 0.1

    unf = LCAOUnfolder(model, rm)
    ks = np.array([[i / N_CELLS, 0, 0] for i in range(N_CELLS)])
    res = unf.compute(ks)
    # per-state grid sum rule (each column = one supercell eigenstate)
    assert np.abs(res.weights.sum(axis=0) - 1.0).max() < 1e-8


def test_sector_enumeration_skew_matrix():
    """Progressive box enumeration must find all |det scmat| quotient
    representatives for skew supercell matrices (round-2 finding)."""
    unf = LCAOUnfolder.__new__(LCAOUnfolder)
    unf._scmat = np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]], dtype=int)
    unf._n_cells = 16
    members = unf._sector(np.array([0.1, 0.3, 0.0]))
    assert len(members) == 16
    # members are distinct primitive k-points mod 1
    keys = {tuple(np.round(np.asarray(m) % 1.0, 8)) for m in members}
    assert len(keys) == 16
