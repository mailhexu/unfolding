"""Production-core behavioral fixture: nearest-neighbour TB supercell (story 048).

Port of the shared-core prototype's toy objective onto the production
core: a one-orbital cubic tight-binding supercell unfolded through
``unfolding.phonon_unfolder`` (the character-sum engine) and the shared
``unfolding.cosets`` helpers — no prototype code is imported. Seals:

- pristine robust weights are binary on a commensurate path;
- the supercell spectrum at each path point equals the folded primitive
  dispersion eps(k + G) over the full coset;
- the degenerate group containing eps(k) carries total weight 1
  (assignments inside a degenerate group are presentation-only);
- an onsite defect produces fractional weights.

Physics conventions encoded here (see devdoc/adding-an-unfolding.md):
supercell shell keys are true integer displacements T_sc = (D - d)/L
with D the minimum-image displacement; the robust-projector phase uses
k* = fold(k) - K_stored WITHOUT mod 1 (the integer part distinguishes
coset members).
"""
import numpy as np
import pytest
from ase import Atoms

from unfolding import cosets
from unfolding.phonon_unfolder import phonon_unfolder


class TBSupercell:
    """Nearest-neighbour cubic TB, one s-orbital, optional onsite defect."""

    def __init__(self, L, t=-0.25, defect=0.0):
        self.L, self.t, self.defect = L, t, defect
        self.cells = np.array([[i, j, k] for i in range(L) for j in range(L)
                               for k in range(L)])
        self.n = L ** 3
        self.atoms = Atoms(symbols=["H"] * self.n,
                           scaled_positions=np.mod(self.cells / L, 1.0),
                           cell=np.eye(3) * L)

    def HR(self):
        """Supercell-lattice hopping tables keyed by true integer keys."""
        d = {}
        for i, ci in enumerate(self.cells):
            for j, cj in enumerate(self.cells):
                if i == j:
                    continue
                delta = cj - ci
                D = delta - self.L * np.round(delta / self.L)  # min image
                if np.abs(D).sum() == 1:                       # nn only
                    Tsc = tuple(int(x) for x in np.round((D - delta) / self.L))
                    d.setdefault(Tsc, np.zeros((self.n, self.n), complex))[i, j] = self.t
        d.setdefault((0, 0, 0), np.zeros((self.n, self.n), complex))[0, 0] += self.defect
        return d

    def bands(self, kpoints_sc):
        """Eigendata on stored supercell momenta; evecs rows = dof."""
        HR, out_e, out_v = self.HR(), [], []
        for K in np.atleast_2d(kpoints_sc):
            H = np.zeros((self.n, self.n), complex)
            for T, b in HR.items():
                H += b * np.exp(2j * np.pi * (np.array(T) @ K))
            eps, C = np.linalg.eigh(H)
            out_e.append(eps)
            out_v.append(C)
        return np.array(out_e), np.array(out_v)

def _eps_prim(k, t=-0.25):
    return 2 * t * np.cos(2 * np.pi * np.atleast_2d(k)).sum(axis=1)


@pytest.fixture(scope="module")
def pristine():
    tb = TBSupercell(4)
    grid = np.mod(np.array([[i, j, l] for i in range(4) for j in range(4)
                            for l in range(4)]) / 4.0, 1.0)
    eps, vecs = tb.bands(grid)
    return tb, grid, eps, vecs


def _engine(tb, grid, vecs):
    return phonon_unfolder(tb.atoms, np.diag([4, 4, 4]), vecs, grid, ndim=1)


def _robust_weights(tb, grid, eps, vecs, path):
    """Robust projector weights on `path` (production engine + cosets).

    Mirrors get_weights_robust with the folded-target phase: k* keeps
    its integer part so coset members are distinguished.
    """
    eng = _engine(tb, grid, vecs)
    R = np.asarray(eng._trans_rs, dtype=float)
    N = len(R)
    home = eng._home_orbit_columns()
    from unfolding.degeneracy import assign_eigvalsh_groups
    out_e = np.zeros((len(path), eps.shape[1]))
    out_w = np.zeros_like(out_e)
    for i, k in enumerate(path):
        ik = cosets.match_stored_k(grid, k, np.diag([4, 4, 4]), 1e-8)
        kstar = cosets.fold(k, np.diag([4, 4, 4])) - np.mod(grid[ik], 1.0)
        ph = np.exp(-2j * np.pi * (R @ kstar))
        A = np.tensordot(ph, np.asarray(vecs)[ik][eng._trans_indices[:, home], :],
                         axes=1) / np.sqrt(N)
        H = A.conj().T @ A
        out_e[i] = eps[ik]
        w0 = np.asarray(np.diag(H)).real.astype(float).copy()
        out_w[i] = np.clip(
            assign_eigvalsh_groups(w0, H, eps[ik], 1e-8, clip=(0.0, 1.0)),
            0.0, 1.0)
    return out_e, out_w


def test_pristine_weights_binary(pristine):
    tb, grid, eps, vecs = pristine
    path = grid[[0, 5, 21, 63]]
    _, w = _robust_weights(tb, grid, eps, vecs, path)
    assert np.all((w < 1e-9) | (w > 1 - 1e-9))


def test_spectrum_equals_folded_primitive_dispersion(pristine):
    tb, grid, eps, vecs = pristine
    path = grid[[0, 5, 21, 63]]
    e, _ = _robust_weights(tb, grid, eps, vecs, path)
    coset = cosets.fold_kpoints(np.diag([4, 4, 4]))
    for ik, k in enumerate(path):
        ref = np.sort(_eps_prim(np.mod(k + coset, 1.0)))
        assert np.allclose(np.sort(e[ik]), ref, atol=1e-10)


def test_eps_group_carries_weight_one(pristine):
    tb, grid, eps, vecs = pristine
    path = grid[[0, 5, 21, 63]]
    e, w = _robust_weights(tb, grid, eps, vecs, path)
    for ik, k in enumerate(path):
        target = _eps_prim(k[None])[0]
        group = np.abs(e[ik] - target) < 1e-8
        assert group.any()
        assert abs(w[ik][group].sum() - 1.0) < 1e-9


def test_character_sum_sector_sum_rule(pristine):
    """Bare G=0 character sum over bands = 1 per stored point (Parseval)."""
    tb, grid, eps, vecs = pristine
    eng = _engine(tb, grid, vecs)
    w = eng.get_weights()
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)


def test_onsite_defect_fractional_weights():
    tb = TBSupercell(4, defect=0.4)
    grid = np.mod(np.array([[i, j, l] for i in range(4) for j in range(4)
                            for l in range(4)]) / 4.0, 1.0)
    eps, vecs = tb.bands(grid)
    path = grid[[0, 5, 21, 63]]
    _, w = _robust_weights(tb, grid, eps, vecs, path)
    n_frac = int(((w > 1e-6) & (w < 1 - 1e-6)).sum())
    assert n_frac > 0


def test_recipe_dataset_roundtrip(pristine):
    """The recipe's consumer seam: any (kpoints, eigenvalues, weights)
    triple exports and re-loads through Dataset.from_result unchanged."""
    from types import SimpleNamespace
    from unfolding.dataset import Dataset, save_dataset, load_dataset
    import tempfile, os
    tb, grid, eps, vecs = pristine
    path = grid[[0, 5, 21]]
    e, w = _robust_weights(tb, grid, eps, vecs, path)
    result = SimpleNamespace(kpoints=np.asarray(path, dtype=float),
                             eigenvalues=e, weights=w)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "tb.json")
        save_dataset(result, out, route="tb-toy",
                     provenance={"engine": "phonon_unfolder/robust"})
        back = load_dataset(out)
    assert np.allclose(back.eigenvalues, e, atol=1e-12)
    assert np.allclose(back.weights, w, atol=1e-12)
    assert np.allclose(back.kpoints, path, atol=1e-12)
    assert back.route == "tb-toy"


def test_shell_keys_are_true_integers(pristine):
    tb = pristine[0]
    HR = tb.HR()
    for T in HR:
        assert all(isinstance(x, int) for x in T)
    # hopping row sums: exactly 6 nn per site at Gamma
    H0 = sum(HR[(0, 0, 0)])
    Htot = np.zeros((tb.n, tb.n), complex)
    for b in HR.values():
        Htot += b
    assert np.allclose(Htot.real.sum(axis=1), 6 * tb.t, atol=1e-12)
