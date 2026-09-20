"""Spinor (nspin=4) end-to-end validation on real SIESTA fixtures.

Fixtures: si_prim_pso.* / si_sc_pso.* -- the scalar Si-diamond runs
regenerated with Spin.Orbit true (scalar Si pseudopotential, so the SO
splitting is numerically ~0: this seals the *spinor machinery* with
trivial physics, i.e. the doubled 2-component basis and the WFSX
coefficient leg, not SOC physics).

Observed shapes (pinned by test_spinor_sc_shapes):

* SC model: 8 atoms x 4 SZ orbitals x 2 spin components = 64 spinor
  orbitals. ``HS_and_eigen`` gives complex Hermitian H and real-overlap
  S, both (64, 64); the real-space tables SR/HR are (125, 64, 64) over
  the 125 shells of the 2x2x2 SCF grid -- the full spinor dimension,
  NOT a (32, 32) spin-blocked form.
* Primitive model: 2 atoms x 4 orbitals x 2 = 16 spinor orbitals,
  H/S (16, 16).

Sealed:

1. Kramers doubling of the spectrum: every spinor Gamma eigenvalue
   equals one scalar Gamma eigenvalue (separate SCFs agree to <0.11 meV;
   sealed at 50 meV).
2. Gamma unfolding group structure via RelabelMap with doubled orbital
   counts (8 per atom): weight sums over exactly-degenerate groups are
   exact integers, pinned to 8x0 + 4x1 + 4x3. The same groups in the
   scalar run give 4x0 + 2x1 + 2x3: Kramers doubling splits each scalar
   group into two noise-separated copies of equal weight sum (the pair
   splitting, ~1e-4 eV SCF noise, exceeds the 1e-6 eV grouping
   tolerance), and the total weight 16 = number of primitive spinor
   orbitals, exactly twice the scalar 8.
3. WFSX eigenvector oracle: weights computed from the SIESTA-native
   spinor WFSX coefficients match the HSX-diagonalization path per
   degenerate group (complex64 precision), which also adjudicates the
   orbital ordering: sisl's spinor state layout (orbital-major, spin
   inside) matches HamiltonIO's H/S table ordering.

Retained from the earlier revision of this file: the FePt real-SO
fixture checks (tests/data/fept_spinor, full-relativistic
pseudopotentials, WFSX leg only -- that branch emitted no HSX).
"""
import os

import numpy as np
import pytest
from scipy.linalg import eigh

DATA = os.path.join(os.path.dirname(__file__), "data", "si_example")
B_DIAMOND = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])


def _read_si_model(fdf_name):
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from siesta_helpers import read_si_model

    return read_si_model(os.path.join(DATA, fdf_name))


def run_sums(e, w, tol=1e-6):
    """Weight sums over consecutive eigenvalue runs (sorted)."""
    order = np.argsort(e)
    e_s, w_s = e[order], w[order]
    out, start = [], 0
    for i in range(1, len(e_s) + 1):
        if i == len(e_s) or e_s[i] - e_s[i - 1] > tol:
            out.append(float(w_s[start:i].sum()))
            start = i
    return out


def make_spinor_unfolder():
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = _read_si_model("si_prim_pso.fdf")
    sc = _read_si_model("si_sc_pso.fdf")
    rm = RelabelMap.from_atoms(
        sc.atoms,
        prim.atoms,
        B_DIAMOND,
        orb_counts_sc=[8] * 8,  # 4 SZ orbitals x 2 spin components
        orb_counts_prim=[8, 8],
    )
    return prim, sc, LCAOUnfolder(HamiltonIOModel(sc), rm)


@pytest.fixture(scope="module")
def spinor():
    pytest.importorskip("HamiltonIO")
    pytest.importorskip("sisl")
    return make_spinor_unfolder()


@pytest.fixture(scope="module")
def scalar_sc():
    pytest.importorskip("HamiltonIO")
    pytest.importorskip("sisl")
    return _read_si_model("si_sc.fdf")


def test_spinor_sc_shapes(spinor):
    """Production path parses nspin=4: (64, 64) complex Hermitian H,
    (64, 64) S, and full-spinor (125, 64, 64) real-space tables."""
    prim, sc, unf = spinor
    out = sc.HS_and_eigen(np.atleast_2d(np.zeros(3)))
    H, S = np.asarray(out[0]), np.asarray(out[1])
    # HamiltonIO batches over k points
    assert H.shape == (1, 64, 64)
    assert S.shape == (1, 64, 64)
    assert np.iscomplexobj(H) and np.iscomplexobj(S)
    assert np.abs(H[0] - H[0].conj().T).max() < 1e-6
    assert np.abs(S[0] - S[0].conj().T).max() < 1e-8
    # real-space tables: 125 shells of the 2x2x2 SCF grid, full spinor dim
    assert np.asarray(sc.Rlist).shape == (125, 3)
    assert np.asarray(sc.SR).shape == (125, 64, 64)
    assert np.asarray(sc.HR).shape == (125, 64, 64)
    # the HamiltonIOModel dict view carries the same blocks
    from unfolding.lcao_unfolder import HamiltonIOModel

    assert len(HamiltonIOModel(sc).SR) == 125
    assert next(iter(HamiltonIOModel(sc).SR.values())).shape == (64, 64)
    # primitive model: 16 spinor orbitals
    outp = prim.HS_and_eigen(np.atleast_2d(np.zeros(3)))
    Hp, Sp = np.asarray(outp[0]), np.asarray(outp[1])
    assert Hp.shape == (1, 16, 16) and Sp.shape == (1, 16, 16)
    assert np.abs(Hp[0] - Hp[0].conj().T).max() < 1e-6


def test_spinor_gamma_spectrum_kramers_pairing(spinor, scalar_sc):
    """Spinor Gamma spectrum = each scalar SC Gamma eigenvalue twice
    (Kramers; SO splitting ~0 with the scalar pseudo, separate SCFs)."""
    from unfolding.lcao_unfolder import HamiltonIOModel

    prim, sc, unf = spinor
    H, S = HamiltonIOModel(sc).hs_and_eigen(np.zeros(3))
    eps_pso = np.sort(eigh(H, S, eigvals_only=True))
    assert len(eps_pso) == 64

    H2, S2 = HamiltonIOModel(scalar_sc).hs_and_eigen(np.zeros(3))
    eps_sc = np.sort(eigh(H2, S2, eigvals_only=True))
    assert len(eps_sc) == 32

    dev = np.abs(eps_pso - np.repeat(eps_sc, 2)).max()
    assert dev < 0.05, f"Kramers/scalar deviation {dev * 1000} meV"


def test_spinor_gamma_group_weights(spinor):
    """Pristine 8-atom spinor SC unfolding at Gamma: group weight sums
    are exact integers, pinned to eight 0s, four 1s, four 3s.

    Mirrors the scalar seal (test_siesta_example.py): the weight-1/3
    groups are the primitive-Gamma bands (s: 1, p: 3, s*: 1, p*: 3 ->
    four groups of each value after Kramers doubling), the 0 groups are
    the X-derived bands folded from the other three sector points.
    Total weight 16 = number of primitive spinor orbitals.
    """
    prim, sc, unf = spinor
    res = unf.compute(np.array([[0.0, 0.0, 0.0]]))
    e, w = res.eigenvalues[0], res.weights[0]
    assert len(e) == 64
    assert abs(w.sum() - 16.0) < 1e-8
    assert np.abs(w - np.round(w)).max() < 1e-8, w
    sums = sorted(run_sums(e, w))
    assert sums == pytest.approx([0.0] * 8 + [1.0] * 4 + [3.0] * 4, abs=1e-8)


def test_spinor_wfsx_gamma_weights_oracle(spinor):
    """Spinor WFSX eigenvector oracle: unfolding weights from the
    SIESTA-native spinor coefficients match the HSX-diagonalization
    path per degenerate eigenvalue group (mirrors the scalar oracle)."""
    import sisl

    from unfolding.lcao_unfolder import HamiltonIOModel

    prim, sc, unf = spinor
    res = unf.compute(np.array([[0.0, 0.0, 0.0]]))

    st = sisl.get_sile(
        os.path.join(DATA, "si_sc_pso.selected.WFSX")
    ).read_eigenstate()
    assert np.allclose(st.info["k"], 0.0)  # entry is Gamma
    C = np.asarray(st.state).T.astype(complex)  # (n_orb=64, n_bands)
    e_st = np.asarray(st.c, dtype=float)
    assert C.shape == (64, 64)

    # WFSX energies are Fermi-shifted; align against the HSX spectrum
    H, S = HamiltonIOModel(sc).hs_and_eigen(np.zeros(3))
    eps_sc = np.sort(eigh(H, S, eigvals_only=True))
    shift = np.sort(eps_sc - np.sort(e_st))[0]
    e_st = e_st + shift
    assert np.abs(np.sort(e_st) - eps_sc).max() < 1e-4

    A = unf._bra_overlap(np.zeros(3))
    G = unf._sector_gram_block(np.zeros(3), np.zeros(3))
    c = A @ C
    x = np.linalg.solve(G, c)
    w_wfsx = np.real(np.sum(np.conj(c) * x, axis=0))

    gr = run_sums(res.eigenvalues[0], res.weights[0])
    gw = run_sums(e_st, w_wfsx)
    # complex64 WFSX coefficients -> ~1e-7 agreement
    assert np.abs(np.array(gr) - np.array(gw)).max() < 1e-6, (gr, gw)
    # the SIESTA-native states are exact folded states: clean 0/1 weights
    assert np.abs(w_wfsx - np.round(w_wfsx)).max() < 1e-6, w_wfsx


def test_spinor_prim_model(spinor, scalar_sc):
    """The primitive spinor model parses and doubles the scalar prim
    Gamma spectrum (Kramers)."""
    from unfolding.lcao_unfolder import HamiltonIOModel

    prim, sc, unf = spinor
    H, S = HamiltonIOModel(prim).hs_and_eigen(np.zeros(3))
    eps_pso = np.sort(eigh(H, S, eigvals_only=True))
    assert len(eps_pso) == 16

    prim_sc = _read_si_model("si_prim.fdf")
    H2, S2 = HamiltonIOModel(prim_sc).hs_and_eigen(np.zeros(3))
    eps_sc = np.sort(eigh(H2, S2, eigvals_only=True))
    assert len(eps_sc) == 8

    dev = np.abs(eps_pso - np.repeat(eps_sc, 2)).max()
    assert dev < 0.05, f"Kramers/scalar prim deviation {dev * 1000} meV"


# -- FePt real-SO fixture: WFSX leg only (no HSX from that branch) ------

FEPT_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "data",
    "fept_spinor",
)


def _first_eig_values(path, nstates):
    """Read the first k-point's energies from a SIESTA EIG file."""
    with open(path) as fh:
        lines = fh.readlines()
    values = []
    for line in lines[2:]:
        fields = line.split()
        if not fields:
            continue
        # A new k-point line starts with its integer index; continuation
        # lines contain only energy values.
        start = 1 if len(values) == 0 and fields[0].isdigit() else 0
        values.extend(float(x) for x in fields[start:])
        if len(values) >= nstates:
            break
    return np.asarray(values[:nstates])


def test_real_spinor_wfsx_is_complex_and_complete():
    sisl = pytest.importorskip("sisl")
    with open(os.path.join(FEPT_DATA, "fept_spinor.fdf")) as fh:
        text = fh.read()
    assert "Spin Spin-Orbit" in text

    wfsx = sisl.get_sile(os.path.join(FEPT_DATA, "fept_spinor.selected.WFSX"))
    kpts, spins, nstates = wfsx.read_info()
    assert kpts.shape == (1, 3)
    assert nstates.tolist() == [[36]]
    state = wfsx.read_eigenstate()
    assert state.state.shape == (36, 36)
    assert np.iscomplexobj(state.state)
    assert np.isfinite(state.state).all()
    assert np.isfinite(state.c).all()


def test_real_spinor_wfsx_eigenvalues_match_eig():
    """The spin-orbit WFSX energies agree with SIESTA's EIG record."""
    sisl = pytest.importorskip("sisl")
    state = sisl.get_sile(
        os.path.join(FEPT_DATA, "fept_spinor.selected.WFSX")
    ).read_eigenstate()
    eig = _first_eig_values(
        os.path.join(FEPT_DATA, "fept_spinor.EIG"), len(state.c)
    )
    assert len(eig) == len(state.c)
    # WFSX.c and EIG carry the same Fermi-shifted absolute values in this
    # spin-orbit branch; the EIG header is not subtracted a second time.
    assert np.abs(np.sort(np.asarray(state.c)) - np.sort(eig)).max() < 1e-4
