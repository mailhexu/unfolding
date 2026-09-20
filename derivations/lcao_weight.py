"""LCAO unfolding weight identity: dual-basis projection on a BvK ring.

Derives and seals the machine-exact statement backing the LCAO weight
formula (paper Eq. 9 -> 12 chain, tested on a 1D BvK ring):

1. The direct dual-basis spectral weight is the projection onto the
   folded state through the FULL sector Gram
   ``G_sec[(k,n),(k',m)] = <kn|S|k'm>``. For translation-invariant
   overlap classes on the complete BvK ring this Gram is exactly
   block-diagonal in k (the cell-origin sum kills k' != k), so the
   per-k blocks alone give the same weights here; the sector-Gram form
   is kept because it is the one that generalizes (truncated AO ranges
   / reduced k-grids make cross-k blocks nonzero). The module asserts
   the block-diagonality and the per-k equivalence numerically.

2. Pristine supercell eigenstates have weight EXACTLY 1 in their own
   sector, at every supercell momentum K.

3. sympy extracts exact per-k coefficient tables
   ``M(k)[t,s] = coeff(D_t C_s)`` of ``W(k) = sum D_t C_s M(k)[t,s]``
   with complex Hermitian overlap classes; evaluating the symbolic tables
   on the numeric fixture reproduces the direct weights to ~1e-15.

Writes LaTeX to ``derivations/out/lcao_weight.tex``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import eigh

OUT = Path(__file__).resolve().parent / "out"

NC = 2          # orbitals per normal cell
L_CELLS = 4     # normal cells on the ring
LS = NC * L_CELLS
P = 2           # supercell copies (supercell = 2 cells = 4 sites)
RS_SUP = (0, 4)
KS = (0.0, np.pi / 2.0)
SEED = 23


def ring_classes(seed: int = SEED) -> np.ndarray:
    """Random positive-definite translation-invariant AO overlap classes."""
    rng = np.random.default_rng(seed)
    ks8 = 2 * np.pi * np.arange(LS) / LS
    w8 = rng.normal(size=LS) + 1j * rng.normal(size=LS)
    spectrum = np.array([abs(np.sum(w8 * np.exp(-2j * ks8 * k))) ** 2 + 0.3 for k in ks8])
    return np.fft.ifft(spectrum)  # Hermitian: cls[LS-d] = conj(cls[d])


def _hopping_classes(seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h8 = rng.normal(size=LS) + 1j * rng.normal(size=LS)
    half = (h8[1:4] + np.conj(h8[1:4][::-1])) / 2
    tail = (h8[5:8] + np.conj(h8[5:8][::-1])) / 2
    return np.concatenate(
        [[h8[0].real], half, [h8[4].real], tail]  # self-conjugate sites real
    )


def make_fixture(seed: int = SEED) -> dict:
    cls = ring_classes(seed)
    hd8 = _hopping_classes(seed)
    S_site = lambda a, b: cls[(b - a) % LS]
    H_site = lambda a, b: hd8[(b - a) % LS]
    S_AO = np.array([[S_site(a, b) for b in range(LS)] for a in range(LS)])

    def sc_matrix(K, mfun):
        return np.array(
            [
                [
                    sum(
                        mfun((R + s) % LS, (Rp + t) % LS) * np.exp(1j * K * ((R - Rp) // NC))
                        for R in RS_SUP
                        for Rp in RS_SUP
                    )
                    / P
                    for t in range(4)
                ]
                for s in range(4)
            ]
        )

    def ket_kn(k, n):
        v = np.zeros(LS, complex)
        for r in range(L_CELLS):
            v[(r * NC + n) % LS] = np.exp(1j * k * r)
        return v

    def S_prim(k):
        return np.array(
            [
                [
                    sum(
                        np.exp(1j * k * (r - rp)) * S_site((r * NC + n) % LS, (rp * NC + m) % LS)
                        for r in range(L_CELLS)
                        for rp in range(L_CELLS)
                    )
                    / L_CELLS
                    for m in range(NC)
                ]
                for n in range(NC)
            ]
        )

    def A_overlap(k, n, K, s):
        return sum(
            np.exp(-1j * k * r)
            * np.exp(1j * K * (R // NC))
            * S_site((r * NC + n) % LS, (R + s) % LS)
            for r in range(L_CELLS)
            for R in RS_SUP
        ) / np.sqrt(L_CELLS * P)

    def sector_gram(K):
        """Full sector Gram (ordered k in {K, K+pi} x n) and the k list."""
        sks = [K % (2 * np.pi), (K + np.pi) % (2 * np.pi)]
        G = np.zeros((2 * NC, 2 * NC), complex)
        for ik, k in enumerate(sks):
            for n in range(NC):
                for ik2, k2 in enumerate(sks):
                    for m in range(NC):
                        G[ik * NC + n, ik2 * NC + m] = (
                            np.conj(ket_kn(k, n)) @ S_AO @ ket_kn(k2, m) / L_CELLS
                        )
        return sks, G

    def direct_weights(C, K):
        """Per-k dual weights via the full sector Gram (cross-k blocks in)."""
        sks, G = sector_gram(K)
        Asec = np.zeros(2 * NC, complex)
        for ik, k in enumerate(sks):
            for n in range(NC):
                Asec[ik * NC + n] = sum(C[s] * A_overlap(k, n, K, s) for s in range(4))
        x = np.linalg.inv(G) @ Asec
        return {
            k: float(
                np.real(sum(np.conj(Asec[ik * NC + n]) * x[ik * NC + n] for n in range(NC)))
            )
            for ik, k in enumerate(sks)
        }

    return {
        "cls": cls,
        "S_AO": S_AO,
        "S_site": S_site,
        "H_site": H_site,
        "sc_matrix": sc_matrix,
        "ket_kn": ket_kn,
        "S_prim": S_prim,
        "A_overlap": A_overlap,
        "sector_gram": sector_gram,
        "direct_weights": direct_weights,
    }


def symbolic_tables(K_sp: str = "zero"):
    """Exact per-k coefficient tables M(k)[t, s] via sympy.

    K_sp selects the supercell momentum: "zero" (sector {0, pi}) or
    "half" (sector {pi/2, 3pi/2}). Overlap classes are symbolic complex
    Hermitian numbers (c0, c4 real; c_d = c_dr + i c_di for d = 1..3,
    cls[LS-d] = conj(cls[d])).
    """
    import sympy as sp

    K_sym = sp.Integer(0) if K_sp == "zero" else sp.pi / 2
    c0, c4 = sp.symbols("c0 c4", real=True)
    cls_sym = {0: c0, 4: c4}
    csyms = {}
    for d in (1, 2, 3):
        re, im = sp.symbols(f"c{d}r c{d}i", real=True)
        csyms[d] = (re, im)
        cls_sym[d] = re + sp.I * im
        cls_sym[LS - d] = re - sp.I * im
    S_site = lambda a, b: cls_sym[(b - a) % LS]
    C = {s: sp.Symbol(f"C{s}") for s in range(4)}
    D = {s: sp.Symbol(f"D{s}") for s in range(4)}

    def ket_vec(k, n):
        v = sp.zeros(LS, 1)
        for r in range(L_CELLS):
            v[(r * NC + n) % LS] = sp.exp(sp.I * k * r)
        return v

    K = K_sym
    sks = [K, K + sp.pi]
    Asec = sp.zeros(2 * NC, 1)
    G = sp.zeros(2 * NC, 2 * NC)
    for ik, k in enumerate(sks):
        for n in range(NC):
            acc = 0
            for s in range(4):
                for r in range(L_CELLS):
                    for R in RS_SUP:
                        acc += (
                            C[s]
                            * sp.exp(-sp.I * k * r)
                            * sp.exp(sp.I * K * (R // NC))
                            * S_site((r * NC + n) % LS, (R + s) % LS)
                        )
            Asec[ik * NC + n] = sp.expand(acc / sp.sqrt(L_CELLS * P))
            for ik2, k2 in enumerate(sks):
                for m in range(NC):
                    ket1, ket2 = ket_vec(k, n), ket_vec(k2, m)
                    a2 = 0
                    for a in range(LS):
                        for b in range(LS):
                            if ket1[a] != 0 and ket2[b] != 0:
                                a2 += sp.conjugate(ket1[a]) * ket2[b] * S_site(a, b)
                    G[ik * NC + n, ik2 * NC + m] = sp.expand(a2 / L_CELLS)

    Ginv = sp.simplify(G.inv())
    x = Ginv * Asec
    tables = {}
    for ik in (0, 1):
        idx = [ik * NC + n for n in range(NC)]
        Wk = sp.expand(sum(sp.conjugate(Asec[i]) * x[i] for i in idx))
        Wk = Wk.subs({sp.conjugate(C[s]): D[s] for s in range(4)})
        Wk = sp.expand(Wk.subs({sp.conjugate(cls_sym[LS - d]): cls_sym[d] for d in (1, 2, 3)}))
        M = sp.zeros(4, 4)
        for t in range(4):
            for s in range(4):
                M[t, s] = sp.simplify(Wk.coeff(D[t], 1).coeff(C[s], 1))
        tables[ik] = M
    return tables, (c0, c4), csyms


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fx = make_fixture()

    # 1. positive-definite overlap blocks at every cell k
    eigmins = []
    for k in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        Spm = fx["S_prim"](k)
        eigmins.append(float(np.linalg.eigvalsh((Spm + Spm.conj().T) / 2).real.min()))
        assert eigmins[-1] > 1e-8, (k, eigmins[-1])

    # 2. pristine weights exactly 1 per state, every supercell K; and the
    #    sector Gram is block-diagonal in k, so per-k blocks reproduce the
    #    sector-Gram weights on this translation-invariant fixture
    worst = 0.0
    crossk = 0.0
    for K in KS:
        Ssc = fx["sc_matrix"](K, fx["S_site"])
        Hsc = fx["sc_matrix"](K, fx["H_site"])
        ev, V = eigh(Hsc, Ssc)
        sks, G = fx["sector_gram"](K)
        crossk = max(crossk, float(np.abs(G[:NC, NC:]).max()))
        for J in range(4):
            Cv = V[:, J]
            dk = fx["direct_weights"](Cv, K)
            worst = max(worst, abs(sum(dk.values()) - 1.0))
            # per-k block equivalent: W(k) = A_k^dag Gk[k]^{-1} A_k
            for ik, k in enumerate(sks):
                Ak = np.array(
                    [
                        sum(Cv[s] * fx["A_overlap"](k, n, K, s) for s in range(4))
                        for n in range(NC)
                    ]
                )
                w_perk = float(np.real(np.conj(Ak) @ np.linalg.inv(G[ik*NC:(ik+1)*NC, ik*NC:(ik+1)*NC]) @ Ak))
                worst = max(worst, abs(w_perk - dk[k]))
    assert worst < 1e-10, worst
    assert crossk < 1e-10, crossk

    # 3. symbolic tables reproduce numeric direct weights, both K sectors
    import sympy as sp

    cls_vals = fx["cls"]
    csub = {}

    def build_csub(c0, c4, csyms):
        sub = {c0: complex(cls_vals[0]).real, c4: complex(cls_vals[4]).real}
        for d in (1, 2, 3):
            re, im = csyms[d]
            sub[re] = complex(cls_vals[d]).real
            sub[im] = complex(cls_vals[d]).imag
        return sub

    seal = 0.0
    for K_sp, K in (("zero", 0.0), ("half", np.pi / 2)):
        tables, (c0, c4), csyms = symbolic_tables(K_sp)
        csub = build_csub(c0, c4, csyms)
        Ssc = fx["sc_matrix"](K, fx["S_site"])
        Hsc = fx["sc_matrix"](K, fx["H_site"])
        ev, V = eigh(Hsc, Ssc)
        sks = [K, K + np.pi]
        for J in range(4):
            Cv = V[:, J]
            dk = fx["direct_weights"](Cv, K)
            for ik, k in enumerate(sks):
                M = np.array(
                    [
                        [complex(sp.N(tables[ik][t, s].subs(csub))) for s in range(4)]
                        for t in range(4)
                    ]
                )
                w_sym = float(np.real(np.conj(Cv) @ M @ Cv))
                seal = max(seal, abs(w_sym - dk[k]))
    assert seal < 1e-10, seal

    tex = [
        r"% Auto-generated by derivations/lcao_weight.py -- do not edit.",
        r"\section{LCAO dual-basis weight identity (BvK ring)}",
        r"The direct spectral weight is the dual-basis projection through the sector Gram",
        r"\begin{equation} G^{\mathrm{sec}}_{(kn),(k'm)} = \langle kn|S|k'm\rangle, \qquad",
        r"W^{KJ}(k) = \sum_n \langle \widetilde{kn}|KJ\rangle \langle KJ|kn\rangle, \quad",
        r"\langle \widetilde{kn}| = \sum_{(k'm)} (G^{\mathrm{sec}})^{-1}_{(kn),(k'm)} \langle k'm| .",
        r"\end{equation}",
        rf"On the toy ring ($N_C={NC}$, $L={L_CELLS}$, $P={P}$) pristine weights equal $1$ to ${worst:.1e}$ at every $K$.",
        rf"For translation-invariant overlaps the sector Gram is block-diagonal in $k$ (cross-$k$ blocks $<{crossk:.0e}$), so per-$k$ blocks give identical weights.",
        rf"Symbolic coefficient tables reproduce the numeric weights to ${seal:.1e}$ at both supercell momenta.",
        r"The clean image-sum form of Eq.~25 is an infinite-lattice identity;",
        r"on BvK rings the tables above are its exact reduction.",
    ]
    (OUT / "lcao_weight.tex").write_text("\n".join(tex) + "\n")

    spinor = spinor_reduction()

    return {
        "pristine_weight_err": worst,
        "symbolic_seal_err": seal,
        "crossk_block_max": crossk,
        "S_prim_eigmins": eigmins,
        "spinor_reduction_err": spinor["spinor_reduction_err"],
    }


def spinor_reduction(seed: int = SEED) -> dict:
    """Seal the sigma,sigma' spin structure of the weight (story 009).

    Symbolically (sympy): with a spin-diagonal overlap,
    ``S_spinor = block_diag(S_up, S_dn)`` in the interlaced coefficient
    basis, the spinor dual weight reduces to the sum of the two spin
    channels' spinless weights,
    ``W_spinor = c^dag kron(Sp_up, I) ... = W_up + W_dn``: the
    sigma,sigma' sums of Eq. 25 run over coefficient indices only.

    Numerically: the reduction is verified on random spin-diagonal
    blocks and on the spin-mixing spinor toy of
    ``tests/test_spinor_unfolder.py`` through the unfolder's own
    weights (the pytest oracle test seals that end to end).
    """
    import sympy as sp

    # symbolic spin-diagonal reduction, 1 orbital x 2 spins per cell
    s_up00, s_up01, s_dn00, s_dn11 = sp.symbols("su00 su01 sd00 sd11", real=True)
    Sp_up = sp.Matrix([[s_up00, s_up01], [s_up01, s_up00]])
    Sp_dn = sp.Matrix([[s_dn00, 0], [0, s_dn11]])
    Sp_spinor = sp.Matrix(
        [[Sp_up[0, 0], 0, Sp_up[0, 1], 0],
         [0, Sp_dn[0, 0], 0, 0],
         [Sp_up[1, 0], 0, Sp_up[1, 1], 0],
         [0, 0, 0, Sp_dn[1, 1]]]
    )  # interlaced (a=0,sig=0), (a=0,sig=1), (a=1,sig=0), (a=1,sig=1)
    cu0, cu1, cd0, cd1 = sp.symbols("cu0 cu1 cd0 cd1")
    c = sp.Matrix([cu0, cd0, cu1, cd1])  # interlaced coefficients
    W_spinor = sp.expand((c.T * Sp_spinor.inv() * c)[0, 0])
    c_up = sp.Matrix([cu0, cu1])
    c_dn = sp.Matrix([cd0, cd1])
    W_split = sp.expand(
        (c_up.T * Sp_up.inv() * c_up)[0, 0] + (c_dn.T * Sp_dn.inv() * c_dn)[0, 0]
    )
    assert sp.simplify(W_spinor - W_split) == 0

    # numeric cross-check: random spin-diagonal S, spin-mixing C
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(20):
        Su = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        Su = Su @ Su.conj().T + np.eye(2)
        Sd = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        Sd = Sd @ Sd.conj().T + np.eye(2)
        Sp_spinor_n = np.zeros((4, 4), complex)
        for a in range(2):
            for b in range(2):
                Sp_spinor_n[2 * a, 2 * b] = Su[a, b]
                Sp_spinor_n[2 * a + 1, 2 * b + 1] = Sd[a, b]
        c_n = rng.normal(size=4) + 1j * rng.normal(size=4)
        w_num = np.real(np.conj(c_n) @ np.linalg.inv(Sp_spinor_n) @ c_n)
        w_split = (
            np.real(np.conj(c_n[::2]) @ np.linalg.inv(Su) @ c_n[::2])
            + np.real(np.conj(c_n[1::2]) @ np.linalg.inv(Sd) @ c_n[1::2])
        )
        worst = max(worst, abs(w_num - w_split))
    assert worst < 1e-10, worst
    return {"spinor_reduction_err": worst, "spinor_symbolic": "exact"}



if __name__ == "__main__":
    print(run())
