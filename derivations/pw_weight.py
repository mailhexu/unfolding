"""Planewave unfolding weight: translation-operator projection and the
orthonormal limit of the Eq. 25 dual-basis formula (story 016, PRD Rev 2).

Three statements are derived and sealed:

1. **Sector decomposition is exact integer arithmetic.** For a supercell
   matrix M (integer, det N) and supercell k-point K (supercell reduced),
   every supercell plane wave |K+G> carries the primitive-lattice
   translation eigenvalue

       T_{R_j} |K+G> = exp(2*pi*i*t_j) |K+G>,   t = M^-T (K + n_G),

   and t mod Z^3 takes exactly N distinct values -- the coset
   representatives of Z^3 / M^T Z^3 (for the diamond M below the group
   is Z2 x Z2, not Z4). Plane waves therefore ARE the eigenbasis of the
   primitive translation operator restricted to the K-Bloch sector (the
   structural difference vs LCAO, where the translation operator must
   be diagonalized on the AO Bloch sums and the weight carries the S
   metric).

2. **Weight = sector projector norm.** With P_m = sum_{G in sector m}
   |K+G><K+G|, the unfolding weight is W_m = <psi|P_m|psi> =
   sum_{G in m} |c_G|^2 and sum_m P_m = 1 on the plane-wave span
   (Parseval: sum_m W_m = sum_G |c_G|^2 = 1 for normalized states).
   The projector is built independently from translation characters,

       (P_m psi)(G) = [ (1/N) sum_R exp(2*pi*i (w - k_m).R) ] c_G,

   with w = M^-T (K + n_G) the primitive fractional wavevector, k_m the
   primitive k-images, and R running over the N primitive-lattice
   translations modulo the supercell lattice -- no sector labels enter
   this evaluation.

3. **Orthonormal limit of Eq. 25 (Lee et al.).** The primitive basis
   functions at wavevector k are the primitive plane waves
   phi_{m,p} = |k_m + G_p> normalized on the supercell torus (a
   primitive plane wave restricted to the supercell IS one supercell
   plane wave), so the sector Gram is exactly

       <phi_{m,p} | phi_{n,p'}> = delta_mn delta_p'

   (character sum over the actual quotient translations, Z2 x Z2 for
   the fixture M), the dual overlaps are A_{m,p} = <phi_{m,p}|psi> =
   c(G_s), and the Eq. 25 weight with the AO-overlap metric replaced by
   the identity collapses to the binning form of statement 2:
   W_m = sum_p |A_{m,p}|^2 -- sealed numerically through an independent
   real-space quadrature on the supercell-fractional grid (the correct
   domain: on the primitive-cell cube the half-integer fold
   wavevectors are NOT orthogonal -- two-sublattice structure).

Numeric cross-checks (random unitary coefficients over a G ball): the
character-projector, binning, and quadrature evaluations agree to
~1e-15, the sum rule holds, and a folded state built through the
independent forward map (primitive G_p -> supercell G_s) has binary
weights in every sector. A reference fixture for the story-017 tests is
dumped to tests/data/pw_synth/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sympy as sp

OUT = Path(__file__).resolve().parent / "out"
DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "pw_synth"

# The diamond conventional-cubic supercell matrix in fcc primitive units
# (same M as the ABINIT Si fixtures, research memo 2026-09-21).
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int)
K = np.zeros(3)                     # supercell Gamma
GCUT = 3                            # G-ball radius in integer units
SEED = 41


# ---------------------------------------------------------------------------
# Exact integer sector machinery
# ---------------------------------------------------------------------------

def coset_reps(M: np.ndarray) -> np.ndarray:
    """Representatives of Z^3 / M^T Z^3 as fractions of the primitive cell."""
    M = np.asarray(M, dtype=int)
    n = int(round(abs(np.linalg.det(M))))
    reps: list[tuple[int, ...]] = []
    rng = int(np.abs(M).max()) * 2 + 2
    for i in range(-rng, rng + 1):
        for j in range(-rng, rng + 1):
            for k in range(-rng, rng + 1):
                v = np.linalg.solve(M.T, np.array([i, j, k], dtype=float))
                f = np.mod(v, 1.0)
                key = tuple(np.round(f * 1e9).astype(np.int64) % 10**9)
                if key not in reps:
                    reps.append(key)
    assert len(reps) == n, (len(reps), n)
    return np.array([np.array(r) / 1e9 for r in reps])


def translation_reps(M: np.ndarray) -> np.ndarray:
    """Integer primitive translations representing Z^3 / (Z^3 M).

    In the row convention A_sc = M A_prim, two integer primitive
    translations r and r' are equivalent when (r-r') M^-1 is integer.
    The fractional key frac(r M^-1) therefore enumerates their N=|det M|
    cosets; the returned vectors themselves remain integer primitive
    translation coordinates for the character projector.
    """
    M = np.asarray(M, dtype=int)
    n = int(round(abs(np.linalg.det(M))))
    inv_M = np.linalg.inv(M)
    by_key: dict[tuple[int, ...], tuple[int, ...]] = {}
    rng = int(np.abs(M).max()) * 2 + 2
    for i in range(rng + 1):
        for j in range(rng + 1):
            for k in range(rng + 1):
                r = np.array([i, j, k], dtype=int)
                f = np.mod(r @ inv_M, 1.0)
                key = tuple(np.round(f * 1e9).astype(np.int64) % 10**9)
                by_key.setdefault(key, tuple(r))
    assert len(by_key) == n, (len(by_key), n)
    return np.array(list(by_key.values()), dtype=int)


def sector_labels(M: np.ndarray, K: np.ndarray, gvecs: np.ndarray) -> np.ndarray:
    """Sector index of each plane wave frac(M^-T (K + n)) -> nearest coset."""
    reps = coset_reps(M)
    t = np.mod(np.linalg.solve(M.T, (K + gvecs).T).T, 1.0)
    d = ((t[:, None, :] - reps[None, :, :] + 0.5) % 1.0 - 0.5)
    return np.argmin((d ** 2).sum(-1), axis=1)


def g_ball(gcut: int) -> np.ndarray:
    n = np.arange(-gcut, gcut + 1)
    i, j, k = np.meshgrid(n, n, n, indexing="ij")
    return np.column_stack([i.ravel(), j.ravel(), k.ravel()])


def _rational(vec: np.ndarray) -> sp.Matrix:
    """Exact sympy Rational column from a float vector on the 1e-9 grid."""
    return sp.Matrix([sp.Rational(int(round(x * 1e9)), 10**9) for x in vec])


def _symbolic_fold_reps(Msym: sp.Matrix, Ksym: sp.Matrix) -> list[tuple[sp.Expr, ...]]:
    """Exact representatives of Z^3 / M^T Z^3 for the symbolic seals."""
    MinvT = Msym.T.inv()
    reps: list[tuple[sp.Expr, ...]] = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                t = MinvT * (Ksym + sp.Matrix([i, j, k]))
                f = tuple(sp.Mod(x, 1) for x in t)
                if f not in reps:
                    reps.append(f)
    n = int(abs(Msym.det()))
    assert len(reps) == n, (len(reps), n)
    zero = (sp.Integer(0), sp.Integer(0), sp.Integer(0))
    assert zero in reps
    return [zero] + [rep for rep in reps if rep != zero]

# ---------------------------------------------------------------------------
# Symbolic seals (sympy, exact)
# ---------------------------------------------------------------------------

def symbolic_seals() -> dict:
    """Statements 1-3 on exact rational arithmetic for the fixture M."""
    Msym = sp.Matrix(M.tolist())
    N = int(abs(Msym.det()))
    assert N == 4
    Ksym = sp.Matrix([sp.Integer(int(x)) for x in K])

    # Statement 1: t = M^-T (K + n) mod 1 partitions Z^3 into N classes.
    kappas = _symbolic_fold_reps(Msym, Ksym)
    sector_partition_exact = len(kappas) == N
    MinvT = Msym.T.inv()

    # eigenvalue constancy on a sector: exp(2*pi*i t_j) depends only on
    # the sector, since t mod 1 is the sector coordinate. Two members of
    # one sector differ by M^T Z^3, so the phase difference is
    # exp(2*pi*i * integer) = 1.
    n1 = sp.Matrix([2, 0, 1])
    n2 = n1 + Msym.T * sp.Matrix([1, 1, 1])      # same sector by construction
    t1 = MinvT * (Ksym + n1)
    t2 = MinvT * (Ksym + n2)
    eigenvalue_exact = sp.simplify(
        sp.exp(2 * sp.pi * sp.I * (t1[0] - t2[0]))
        * sp.exp(2 * sp.pi * sp.I * (t1[1] - t2[1]))
        * sp.exp(2 * sp.pi * sp.I * (t1[2] - t2[2]))
    ) == 1

    # Statement 3a: character orthogonality over the ACTUAL quotient
    # translations (Z2 x Z2 for this M): pairing each fold rep kappa
    # with the translation reps R, sum_R exp(2*pi*i kappa.R) is N for
    # the trivial rep and 0 for every nontrivial one.
    Rtrs = [sp.Matrix([sp.Integer(int(x)) for x in r])
            for r in translation_reps(M)]
    char_sums = []
    for kap in kappas:
        kapvec = sp.Matrix(kap)
        tot = sp.Integer(0)
        for R in Rtrs:
            tot += sp.exp(2 * sp.pi * sp.I * kapvec.dot(R))
        char_sums.append(sp.simplify(tot))
    bloch_orthogonality_exact = (
        char_sums[0] == N and all(s == 0 for s in char_sums[1:])
    )

    # Statement 3b: Eq. 25 with S = identity -- the Gram of the
    # supercell-normalized primitive plane waves is delta, and the
    # dual-basis weight reduces to the sector amplitude norm
    # sum_p |a_{m,p}|^2.
    a0, a1 = sp.symbols("a0 a1")
    avec = sp.Matrix([a0, a1])
    W_dual = (avec.H * sp.eye(2).inv() * avec)[0]
    dual_limit_exact = sp.expand(
        W_dual - (sp.conjugate(a0) * a0 + sp.conjugate(a1) * a1)
    ) == 0

    return {
        "sector_partition_exact": sector_partition_exact,
        "eigenvalue_exact": eigenvalue_exact,
        "bloch_orthogonality_exact": bloch_orthogonality_exact,
        "dual_limit_exact": dual_limit_exact,
    }


# ---------------------------------------------------------------------------
# Numeric cross-checks
# ---------------------------------------------------------------------------

def weights_binning(cg: np.ndarray, M: np.ndarray, K: np.ndarray,
                    gvecs: np.ndarray, nsec: int) -> np.ndarray:
    """Statement 2: W_m = sum_{G in m} |c_G|^2 via reciprocal labels."""
    labels = sector_labels(M, K, gvecs)
    W = np.zeros((cg.shape[0], nsec))
    for m in range(nsec):
        W[:, m] = (np.abs(cg[:, labels == m]) ** 2).sum(axis=1)
    return W


def weights_character_projector(cg: np.ndarray, M: np.ndarray, K: np.ndarray,
                                gvecs: np.ndarray) -> np.ndarray:
    """Statement 2 via translation characters -- label-free.

    (P_m psi)(G) = [ (1/N) sum_R exp(2*pi*i (w - k_m).R) ] c_G with w =
    M^-T (K + n_G) the primitive fractional wavevector, k_m = frac(M^-T
    K) + kappa_m the primitive k-images, and R over the primitive
    translations modulo the supercell lattice. The bracket is the sector
    indicator constructed from the operator side; no reciprocal-space
    label enters.
    """
    reps = coset_reps(M)
    Rtrs = translation_reps(M)
    N = len(Rtrs)
    nsec = len(reps)
    kfrac = np.mod(np.linalg.solve(M.T, K), 1.0)
    w = np.linalg.solve(M.T, (K + gvecs).T).T
    W = np.zeros((cg.shape[0], nsec))
    for m, kap in enumerate(reps):
        char = np.exp(2j * np.pi * (w - (kfrac + kap)) @ Rtrs.T).sum(axis=1) / N
        assert np.allclose(char.imag, 0.0, atol=1e-12), "complex character indicator"
        indicator = np.rint(char.real)
        assert np.allclose(char.real, indicator, atol=1e-12), "character not integer"
        assert np.all((indicator == 0) | (indicator == 1)), "character not a projector"
        W[:, m] = ((np.abs(cg) ** 2) * indicator).sum(axis=1)
    return W


def weights_quadrature(cg: np.ndarray, M: np.ndarray, K: np.ndarray,
                       gvecs: np.ndarray, ng: int | None = None) -> np.ndarray:
    """Eq. 25 (S = identity) weights via real-space quadrature.

    psi is sampled on a supercell-fractional integer grid y = m/N (the
    sparse frequencies K + n are integers for K on the commensurate
    grid, here K = 0) and transformed back; the amplitudes A_{m,p} come
    out of the grid transform -- no coefficient index is reused -- and
    are binned by frac(M^-T (K + n)).
    """
    reps = coset_reps(M)
    N = ng if ng is not None else 2 * (int(np.abs(gvecs).max()) + 1)
    freq = K + gvecs                                   # SC-fractional frequencies (integers)
    r = np.arange(N)
    grid = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    phase = np.exp(2j * np.pi * freq @ grid.T / N)     # (npw, ngrid)
    psi = cg @ phase                                   # (nband, ngrid)
    A = (psi @ phase.conj().T) / N ** 3                # (nband, npw)
    tfrac = np.mod(np.linalg.solve(M.T, freq.T).T, 1.0)
    d = ((tfrac[:, None, :] - reps[None, :, :] + 0.5) % 1.0 - 0.5)
    lab = np.argmin((d ** 2).sum(-1), axis=1)
    W = np.zeros((cg.shape[0], len(reps)))
    for m in range(len(reps)):
        W[:, m] = (np.abs(A[:, lab == m]) ** 2).sum(axis=1)
    return W


def forward_folded_state(M: np.ndarray, K: np.ndarray, m: int,
                         primitive_gcut: int = 2, sc_gcut: int | None = None):
    """Build a sector-m state through the FORWARD map, label-free.

    Primitive G_p candidates map to supercell G_s through
    K + G_s = M^T (kappa_m + G_p), with integer n_s asserted. When
    ``sc_gcut`` is set, retain the valid primitive-PW subset whose image
    lives in that SC G ball; this remains a genuine primitive state and
    avoids silently clipping a constructed oracle during embedding.
    """
    Msym = sp.Matrix(np.asarray(M, dtype=int).tolist())
    kappa = _rational(coset_reps(M)[m])
    Ksym = sp.Matrix([sp.Integer(int(x)) for x in K])
    n_s = []
    for g in g_ball(primitive_gcut):
        ns = Msym.T * (kappa + sp.Matrix(g.tolist())) - Ksym
        assert all(x == int(x) for x in ns), f"non-integer SC G for G_p={g}"
        ns_int = np.array([int(x) for x in ns])
        if sc_gcut is None or np.abs(ns_int).max() <= sc_gcut:
            n_s.append(ns_int)
    assert n_s, f"no forward images inside SC G cutoff for sector {m}"
    rng = np.random.default_rng(SEED + m)
    c_s = rng.normal(size=len(n_s)) + 1j * rng.normal(size=len(n_s))
    c_s /= np.linalg.norm(c_s)
    return np.array(n_s), c_s


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    sym = symbolic_seals()
    assert all(sym.values()), sym

    rng = np.random.default_rng(SEED)
    gvecs = g_ball(GCUT)
    reps = coset_reps(M)
    nsec = len(reps)

    # random orthonormal states (rows unitary)
    nband, npw = 6, len(gvecs)
    Araw = rng.normal(size=(nband, npw)) + 1j * rng.normal(size=(nband, npw))
    Q, _ = np.linalg.qr(Araw.T)            # columns orthonormal
    cg = Q.conj().T                        # rows orthonormal

    W_bin = weights_binning(cg, M, K, gvecs, nsec)
    W_proj = weights_character_projector(cg, M, K, gvecs)
    W_quad = weights_quadrature(cg, M, K, gvecs)
    binning_vs_projector = np.abs(W_bin - W_proj).max()
    dual_vs_binning = np.abs(W_quad - W_bin).max()
    sum_rule_err = np.abs(W_bin.sum(1) - 1.0).max()

    # folded states through the forward map: coefficients on the SC
    # G-vectors that are images of one sector's primitive G_p ball ->
    # binary weights (pristine seal, construction independent of labels)
    W_folded_err = 0.0
    key = {tuple(g): i for i, g in enumerate(gvecs)}
    for m in range(nsec):
        n_s, c_s = forward_folded_state(M, K, m, sc_gcut=GCUT)
        c = np.zeros((1, npw), dtype=complex)
        for g, cc in zip(n_s, c_s):
            assert tuple(g) in key, f"forward image outside SC G ball: {g}"
            c[0, key[tuple(g)]] = cc
        assert np.isclose(np.linalg.norm(c), 1.0, atol=1e-12)
        Wf = weights_binning(c, M, K, gvecs, nsec)
        target = np.zeros(nsec)
        target[m] = 1.0
        W_folded_err = max(W_folded_err, np.abs(Wf - target).max())

    fixture_path = DATA / "pw_synth_fixture.npz"
    np.savez(fixture_path, M=M, K=K, gvecs=gvecs,
             labels=sector_labels(M, K, gvecs), reps=reps, cg=cg,
             W_ref=W_bin, seed=SEED)

    tex = r"""% Auto-generated by derivations/pw_weight.py -- do not edit.
\section{Planewave unfolding weight}
\begin{equation}
T_{\mathbf{R}}\,\ket{\mathbf{K}+\mathbf{G}} =
  e^{2\pi i\,\mathbf{t}\cdot\mathbf{r}},\qquad
  \mathbf{t} = M^{-T}(\mathbf{K}+\mathbf{n}_\mathbf{G}) \bmod \mathbb{Z}^3 .
\end{equation}
\begin{equation}
W_m = \braket{\psi|P_m|\psi} = \sum_{\mathbf{G}\in m}|c_\mathbf{G}|^2,
  \qquad \sum_m W_m = \sum_\mathbf{G}|c_\mathbf{G}|^2 = 1 .
\end{equation}
With supercell-normalized primitive plane waves
$\phi_{m,p} = \ket{\mathbf{k}_m+\mathbf{G}_p}$ (a primitive plane wave
restricted to the supercell torus is one supercell plane wave):
\begin{equation}
\braket{\phi_{m,p}}{\phi_{n,p'}} = \delta_{mn}\delta_{\mathbf{G}_p\mathbf{G}_p'}
  \;\Rightarrow\;
  W^{\text{Eq.25}}_{S=\mathbb{1}} = \sum_p|A_{m,p}|^2 = W_m .
\end{equation}
"""
    (OUT / "pw_weight.tex").write_text(tex)

    return {
        "symbolic_seals": sym,
        "binning_vs_projector_err": float(binning_vs_projector),
        "dual_vs_binning_err": float(dual_vs_binning),
        "sum_rule_err": float(sum_rule_err),
        "folded_binary_err": float(W_folded_err),
        "n_sectors": nsec,
        "npw": npw,
        "fixture": str(fixture_path),
    }


if __name__ == "__main__":
    print(run())
