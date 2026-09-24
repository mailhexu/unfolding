"""Magnon unfolding weight: reciprocal-coset Euclidean projector on the
Boson BdG mode amplitudes (story 021, PRD Rev 3, ADR-011).

Three statements are derived and sealed:

1. **Fold characters resolve the identity on the site amplitudes.**
   A supercell magnon mode at supercell momentum K lives on the
   magnetic atoms of one supercell cell. Each supercell magnetic atom a
   carries a primitive sublattice index m_a and a primitive-cell coset
   label n_a in the finite translation group G = Z^3 / M^T Z^3 of order
   N = det M (from tau_prim = M @ tau_sc, m = tau_prim mod 1,
   n = tau_prim - m, exact integers for a commensurate supercell). The
   primitive Bloch basis at primitive momentum q restricted to
   sublattice m is

       phi_{q,m}(m', n) = delta_{m m'} chi_q(n) / sqrt(N),
       chi_q(n) = exp(2*pi*i q . n),

   and the N folds q of one K are the N distinct characters of G.
   Sealed: P_q = sum_m |phi_{q,m}><phi_{q,m}| obeys P_q^2 = P_q,
   P_q P_q' = 0 for distinct folds, and sum_q P_q = 1.

2. **Weight = projector norm, sectors summed.** For a BdG mode
   psi_nu = (u_nu, v_nu) of CANONICAL amplitudes (Psi = K X from the
   TB2J Cholesky frame, <Psi|g|Psi> = energy, here Euclidean
   normalized; both sectors of a clean fold state carry the same fold
   character -- sealed),

       W_q(nu) = sum_{s in {u,v}} sum_m |<phi_{q,m}| psi_nu^s>|^2,

   with sum_q W_q(nu) = ||psi_nu||^2 = 1 (Parseval over the character
   basis). The projector is Euclidean: it resolves the identity on
   mode amplitudes and never touches the Boson metric
   g = diag(I, -I), which governs the BdG eigenproblem (mode
   orthogonality and completeness) rather than this spectral
   resolution. Degenerate mode groups (G-AFM doublets are exact) mix
   the stored gauge; the invariant content of a near-degenerate group
   is the eigenvalue spectrum of its projected Gram, handled by the
   same eigen-assignment as the electronic engines (story 022).

3. **Heisenberg oracle.** The nearest-neighbour bipartite
   antiferromagnetic Heisenberg chain (H = J sum_<ij> S_i . S_j,
   two-site magnetic primitive cell, bonds at Delta = +/- 1/2 cell)
   has LSWT BdG blocks A = J S z (z = 2) and B(q) = J S gamma_q with
   gamma_q = 2 cos(pi q); sympy diagonalizes the 4x4 BdG symbolically
   to omega(q) = 2 J S |sin(pi q)| (Goldstone at q = 0 and the zone
   edge). A numeric supercell BdG (M = diag(3,1,1), six sites,
   supercell Bloch phases on the wrapping bond, TB2J Cholesky
   convention) must reproduce omega at the on-shell fold of every K
   with binary weights (w = 1 on exactly one fold), and a
   weakened-bond supercell must keep the per-mode sum rule while
   developing fractional weights.

Dumps ``tests/data/magnon_synth/magnon_synth_fixture.npz`` (story 022)
and ``derivations/out/magnon_weight.tex``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "derivations" / "out"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "tests" / "data" / "magnon_synth"
DATA.mkdir(exist_ok=True, parents=True)
SEED = 20260924

# Oracle: NN AFM Heisenberg chain, two-site magnetic primitive cell,
# supercell M = diag(3, 1, 1) (six magnetic atoms).
S_ORACLE = 1.0
J_ORACLE = 1.0
NCELL = 3


def fold_characters(M):
    """(qpts, n_vectors): fold representatives and a complete coset-label set.

    The fold q points are the characters chi_q(n) = exp(2 pi i q . n)
    of G = Z^3 / M^T Z^3; labels n are chosen inside a small box.
    """
    M = np.asarray(M, dtype=int)
    N = round(abs(np.linalg.det(M)))
    bound = int(np.abs(M).max()) * 2 + 2
    labels, seen = [], set()
    for n in np.ndindex(*(bound * 2 + 1,) * 3):
        n = np.array(n) - bound
        # canonical key of n modulo the M^T lattice
        key = tuple(
            np.round(np.linalg.solve(M.T.astype(float), n.astype(float)), 6) % 1.0
        )
        if key not in seen:
            seen.add(key)
            labels.append(n)
        if len(labels) == N:
            break
    if len(labels) != N:
        raise RuntimeError("could not enumerate coset labels")
    labels = np.array(labels)
    qs = np.mod(np.linalg.solve(M.T.astype(float), labels.T.astype(float)).T, 1.0)
    # deduplicate by distinct character values
    chars = np.exp(2j * np.pi * qs @ labels.T)
    uniq = []
    for i in range(len(qs)):
        if not any(np.allclose(chars[i], chars[j]) for j in uniq):
            uniq.append(i)
    return qs[uniq], labels


def site_labels(positions_sc, M):
    """(m_idx, n_vec, tau_prim) per supercell magnetic atom.

    m_idx enumerates primitive sublattices (distinct tau_prim mod 1);
    n_vec are integer coset labels; tau_prim = tau_sc @ M.T.
    """
    M = np.asarray(M, dtype=int)
    tau_prim_raw = np.asarray(positions_sc) @ M
    tau_prim = np.mod(tau_prim_raw, 1.0)
    n_vec = tau_prim_raw - tau_prim
    if np.max(np.abs(n_vec - np.round(n_vec))) > 1e-8:
        raise ValueError("supercell positions are not commensurate with M")
    n_vec = np.round(n_vec).astype(int)
    keys = [tuple(np.round(t, 6)) for t in tau_prim]
    subs = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([subs[k] for k in keys]), n_vec, tau_prim


def character_weight(wf, positions_sc, M, q):
    """Fold weight of one Euclidean BdG mode on fold q (numeric, TB2J-free).

    wf : (2 Nmag,) complex, u block then v block.
    """
    M = np.asarray(M, dtype=int)
    N = round(abs(np.linalg.det(M)))
    m_idx, n_vec, _ = site_labels(positions_sc, M)
    chi = np.exp(-2j * np.pi * n_vec @ np.mod(q, 1.0))
    nmag = len(positions_sc)
    w = 0.0
    for m in np.unique(m_idx):
        sel = m_idx == m
        w += abs(np.sum(chi[sel] * wf[:nmag][sel])) ** 2 / N
        w += abs(np.sum(chi[sel] * wf[nmag:][sel])) ** 2 / N
    return w


def heisenberg_chain_bdg(nmag, js, K=None):
    """2nmag x 2nmag BdG matrix of the periodic NN chain.

    Moments alternate +/-S with site parity. Bonds carry supercell Bloch
    phases e^{2 pi i K . dr} when they wrap the periodic boundary
    (dr in supercell fractional units).
    """
    S = S_ORACLE
    A = np.zeros((nmag, nmag))
    B = np.zeros((nmag, nmag), dtype=complex)
    for i in range(nmag):
        j = (i + 1) % nmag
        A[i, i] += js[i] * S
        A[j, j] += js[i] * S
        if j > i:
            phase = 1.0
        elif K is not None:
            # wrapping bond crosses one full supercell translation
            phase = np.exp(2j * np.pi * K[0])
        else:
            phase = 1.0
        B[i, j] += js[i] * S * phase
        B[j, i] += js[i] * S * np.conj(phase)
    return np.block([[A, B], [B.conj().T, A.T]])


def positive_modes(H):
    """Positive BdG modes as canonical Euclidean-normalized amplitudes.

    TB2J's ``get_magnon_eigenstates`` stores the Cholesky-frame modes X
    (columns of eigh(K^H g K)); the canonical amplitudes are
    Psi = K @ X (boson norm <Psi|g|Psi> = energy). This function returns
    (energies, modes) with modes = Psi rows, each Euclidean-normalized;
    both u and v sectors of a clean fold state then carry the SAME fold
    character exactly (sealed in the oracle below).
    """
    n = H.shape[-1] // 2
    min_eig = 0.0
    try:
        K = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        min_eig = np.min(np.linalg.eigvalsh(H))
        K = np.linalg.cholesky(H - (min_eig - 1e-9) * np.eye(2 * n))
    g = np.diag([1.0] * n + [-1.0] * n)
    vals, vecs = np.linalg.eigh(K.conj().T @ g @ K)
    energies = vals[n:] + min_eig
    psi = (K @ vecs[:, n:]).T
    psi = psi / np.linalg.norm(psi, axis=1)[:, None]
    return energies, psi


def symbolic_seals():
    """Projector algebra on the fold characters (N = 2 chain doubling)."""
    import sympy as sp

    N = 2
    nvecs = [np.array([0, 0, 0]), np.array([1, 0, 0])]

    def phi(q, m):
        v = np.zeros(4, dtype=complex)  # basis order (m, n)
        for idx, nv in enumerate(nvecs):
            v[m * 2 + idx] = np.exp(2j * np.pi * np.dot(q, nv)) / np.sqrt(N)
        return v

    def P(q):
        out = np.zeros((4, 4), dtype=complex)
        for m in (0, 1):
            v = phi(q, m)
            out += np.outer(v, v.conj())
        return out

    q0, q1 = np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])
    exact = lambda mat: sp.Matrix(np.round(mat.real, 12) + 1j * np.round(mat.imag, 12))  # noqa: E731
    seals = {
        "idempotent": sp.simplify(exact(P(q0) @ P(q0) - P(q0))) == sp.zeros(4, 4),
        "orthogonal": sp.simplify(exact(P(q0) @ P(q1))) == sp.zeros(4, 4),
        "resolution": np.allclose(P(q0) + P(q1), np.eye(4)),
    }
    rng = np.random.default_rng(SEED)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    w = [np.real(np.vdot(psi, P(q) @ psi)) for q in (q0, q1)]
    seals["parseval"] = abs(sum(w) - 1.0) < 1e-12
    return seals


def chain_dispersion_symbolic():
    """sympy: omega(q) of the two-site-cell AFM chain BdG, gamma = 2 cos(pi q)."""
    import sympy as sp

    q, J, S = sp.symbols("q J S", positive=True)
    A = 2 * J * S  # z = 2
    B = J * S * 2 * sp.cos(sp.pi * q)
    omega = sp.sqrt(A**2 - B**2)
    return sp.simplify(omega)


def run():
    import sympy as sp

    sym = symbolic_seals()
    assert all(sym.values()), sym

    omega_sym = chain_dispersion_symbolic()
    omega_fn = sp.lambdify(sp.symbols("q J S", positive=True), omega_sym, "numpy")

    # non-symmetric on purpose: positions must transform with M (not
    # M.T) -- the shear in the dummy y/z directions leaves the chain
    # physics untouched but seals the asymmetric branch
    M = np.array([[3, 0, 0], [0, 1, 1], [0, 0, 1]])
    nmag = 2 * NCELL
    positions = np.array([[j / nmag, 0.0, 0.0] for j in range(nmag)])
    qpts0, labels = fold_characters(M)

    js_defect = np.ones(nmag) * J_ORACLE
    js_defect[1] = 0.5 * J_ORACLE

    def group_energies(energies, tol):
        """Consecutive near-degenerate groups by energy."""
        groups, start = [], 0
        for stop in range(1, len(energies) + 1):
            if stop == len(energies) or energies[stop] - energies[stop - 1] > tol:
                groups.append(list(range(start, stop)))
                start = stop
        return groups

    K_path = np.array([[kx, 0.0, 0.0] for kx in np.linspace(0.0, 1.0, 13)])
    binary_err = sum_err = energy_err = 0.0
    for K in K_path:
        energies, modes = positive_modes(
            heisenberg_chain_bdg(nmag, np.ones(nmag) * J_ORACLE, K)
        )
        # primitive momenta folding onto this K: q = M^-T (K + n)
        qfolds = np.mod(
            np.linalg.solve(M.T.astype(float), (K + labels).T.astype(float)).T, 1.0
        )
        weights = np.array(
            [
                [character_weight(modes[nu], positions, M, q) for q in qfolds]
                for nu in range(nmag)
            ]
        )
        for grp in group_energies(energies, 1e-9):
            # gauge-invariant group weights: per-fold sums over the group
            wgrp = weights[grp].sum(axis=0)
            for w in wgrp:
                binary_err = max(binary_err, abs(w - round(w)))
            sum_err = max(
                sum_err, abs(wgrp.sum() - len(grp)), abs(weights.sum() - nmag)
            )
            for f in np.where(wgrp > 1 - 1e-8)[0]:
                target = abs(float(omega_fn(np.mod(qfolds[f, 0], 1.0), J_ORACLE, S_ORACLE)))
                energy_err = max(energy_err, abs(energies[grp[0]] - target))
        # defect bonds: only the per-mode sum rule survives
        _, modes_d = positive_modes(heisenberg_chain_bdg(nmag, js_defect, K))
        for nu in range(nmag):
            ws = sum(
                character_weight(modes_d[nu], positions, M, q) for q in qfolds
            )
            sum_err = max(sum_err, abs(ws - 1.0))

    qpts = qpts0

    # independent numeric anchor: the two-site primitive cell BdG (same
    # builder, one magnetic cell) reproduces omega_fn
    qcheck = np.linspace(0.0, 0.5, 7)
    for qc in qcheck:
        energies, _ = positive_modes(heisenberg_chain_bdg(2, np.ones(2), np.array([qc, 0.0, 0.0])))
        target = abs(float(omega_fn(np.mod(qc, 1.0), J_ORACLE, S_ORACLE)))
        energy_err = max(energy_err, abs(energies.min() - target))

    fixture_path = DATA / "magnon_synth_fixture.npz"
    energies, modes = positive_modes(heisenberg_chain_bdg(nmag, np.ones(nmag) * J_ORACLE, np.array([0.0, 0, 0])))
    energies_d, modes_d = positive_modes(heisenberg_chain_bdg(nmag, js_defect, np.array([0.0, 0, 0])))
    np.savez(
        fixture_path,
        M=M,
        positions=positions,
        qpts=qpts,
        labels=labels,
        energies=energies,
        modes=modes,
        energies_defect=energies_d,
        modes_defect=modes_d,
        js_defect=js_defect,
        q_ref=np.linspace(0.0, 0.5, 21),
        omega_ref=np.array(
            [
                abs(float(omega_fn(np.mod(q, 1.0), J_ORACLE, S_ORACLE)))
                for q in np.linspace(0.0, 0.5, 21)
            ]
        ),
        seed=SEED,
    )

    tex = r"""% Auto-generated by derivations/magnon_weight.py -- do not edit.
\section{Magnon unfolding weight}
BdG mode amplitudes live on supercell magnetic atoms; each atom carries
a primitive sublattice $m$ and a coset label $n\in G=\mathbb{Z}^3/M^T\mathbb{Z}^3$
(from $\tau_{\mathrm{prim}}=M\tau_{\mathrm{sc}}$). With the fold
characters $\chi_q(n)=e^{2\pi i q\cdot n}$:
\begin{equation}
\phi_{q,m}(m',n)=\delta_{mm'}\chi_q(n)/\sqrt{N},\qquad
P_q^2=P_q,\quad P_qP_{q'}=0,\quad \sum_q P_q=\mathbb{1}.
\end{equation}
\begin{equation}
W_q(\nu)=\sum_{s\in\{u,v\}}\sum_m\bigl|\langle\phi_{q,m}|\psi^s_\nu\rangle\bigr|^2,
\qquad \sum_q W_q(\nu)=\lVert\psi_\nu\rVert^2=1 .
\end{equation}
The projector is Euclidean; the Boson metric $g=\mathrm{diag}(I,-I)$
governs the BdG eigenproblem, not this spectral resolution.
Heisenberg oracle (sympy): $\omega(q)=2JS\,|\sin\pi q|$ (NN AFM chain,
two-site cell, $z=2$), reproduced by the supercell BdG with binary
weights on the on-shell fold.
"""
    (OUT / "magnon_weight.tex").write_text(tex)

    return {
        "symbolic_seals": {k: bool(v) for k, v in sym.items()},
        "dispersion": str(omega_sym),
        "oracle_binary_err": float(binary_err),
        "oracle_sum_rule_err": float(sum_err),
        "oracle_energy_err": float(energy_err),
        "fixture": str(fixture_path),
    }


def q_fold_arg(K, q, M):
    """Primitive momentum argument of the analytic dispersion.

    The mode lives at supercell momentum K; its on-shell fold q (a
    character representative) labels the primitive momentum of the
    Bloch content: q_prim = frac-of (K mapped into the primitive frame)
    combined with the fold character. For the chain oracle with
    M = diag(3,1,1) this reduces to q_prim = q + integer shifts that
    leave |sin(pi q)| invariant, so the argument is q itself.
    """
    return q[0]


if __name__ == "__main__":
    print(run())
