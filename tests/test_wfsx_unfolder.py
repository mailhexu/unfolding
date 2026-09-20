"""WFSX-driven unfolding on the Si 8-atom SC path fixture (si_sc_path).

Seals :class:`unfolding.wfsx_unfolder.WFSXUnfolder` against the
HSX-diagonalization reference (:class:`unfolding.lcao_unfolder.LCAOUnfolder`,
``method="ideal"``) on the committed 157-point Gamma-X-W-Gamma-L-W-X path.

Conventions adjudicated here (see also the module docstring):

* Gauge: SIESTA stores WFSX coefficients in the orbital-position gauge.
  Convention 2 (the LCAO overlap machinery) needs the per-orbital factor
  ``c_conv2[s, a] = c_sia[s, a] * exp(+2 pi i K . tau_s)`` with ``K =
  k_prim @ scmat`` the supercell fractional momentum and ``tau_s`` the
  supercell-fractional position of the atom carrying orbital ``s``.
  The ``+`` sign reproduces the HSX reference group weights to ~1e-7 at
  generic k; omitting the factor or flipping the sign deviates by
  1e-2 .. 16 (test_gauge_conversion_is_pinned). At Gamma the factor is 1
  for every choice, so the Gamma-only oracle cannot distinguish it.
* Eigenvalues: WFSX energies are SIESTA-native, ``e_stored = e_abs +
  E_F`` with ``E_F`` of the *writing* run (-3.00800582 eV for
  si_sc_path, its own si_sc_path.EIG header). Subtraction of the header
  value recovers the parsed-HSX eigenvalues to <=9e-4 eV; the residual
  is run-to-run SCF noise between the si_sc.HSX run and the path run --
  a same-run control (si_sc.selected.WFSX vs si_sc.HSX at Gamma) is
  constant to 1e-13 eV (test_energies_fermi_shift).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from siesta_helpers import B_DIAMOND, read_fermi_energy, read_si_model  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "si_example")
SC_FDF = os.path.join(DATA, "si_sc.fdf")
PRIM_FDF = os.path.join(DATA, "si_prim.fdf")
PATH_WFSX = os.path.join(DATA, "si_sc_path.selected.WFSX")
PATH_EIG = os.path.join(DATA, "si_sc_path.EIG")
SC_WFSX = os.path.join(DATA, "si_sc.selected.WFSX")


def group_sums(e, w, tol):
    """Sum weights over consecutive ascending-eigenvalue runs."""
    order = np.argsort(e)
    e_s, w_s = e[order], w[order]
    out, start = [], 0
    for i in range(1, len(e_s) + 1):
        if i == len(e_s) or e_s[i] - e_s[i - 1] > tol:
            out.append(float(w_s[start:i].sum()))
            start = i
    return np.array(out)


@pytest.fixture(scope="module")
def rig():
    pytest.importorskip("HamiltonIO")
    pytest.importorskip("sisl")
    from HamiltonIO.siesta.wfsx import SiestaWFSXParser
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap
    from unfolding.wfsx_unfolder import WFSXUnfolder

    sc = read_si_model(SC_FDF)
    prim = read_si_model(PRIM_FDF)
    B = B_DIAMOND
    relabel = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, B, orb_counts_sc=[4] * 8, orb_counts_prim=[4, 4]
    )
    hs_model = HamiltonIOModel(sc)
    wfsx = SiestaWFSXParser(PATH_WFSX, cell=np.asarray(sc.atoms.cell)).read()
    unf = WFSXUnfolder(wfsx, hs_model, relabel, B)
    lcao = LCAOUnfolder(HamiltonIOModel(sc), relabel)
    # path in primitive fractional coords: row convention K_sc = k_prim @ B
    k_prim = np.asarray(wfsx.kpoints, dtype=float) @ np.linalg.inv(B)
    return dict(unf=unf, lcao=lcao, hs_model=hs_model, wfsx=wfsx,
                k_prim=k_prim, B=B)


def test_weights_and_shapes_full_path(rig):
    """157-point path: WFSX weights == HSX-diagonalization reference.

    Degenerate subspaces may rotate internally, so the seal compares
    degenerate-group weight sums (average_degenerate), as in
    test_wfsx_gamma_weights_oracle.
    """
    from unfolding.wfsx_unfolder import WFSXWeights

    ref = rig["lcao"].compute(rig["k_prim"], method="ideal")
    res = rig["unf"].compute(rig["k_prim"], method="ideal")
    assert isinstance(res, WFSXWeights)
    assert res.kpoints.shape == (157, 3)
    assert res.eigenvalues.shape == (157, 32)
    assert res.weights.shape == (157, 32)
    # kpoints are echoed exactly as passed in
    assert np.array_equal(res.kpoints, np.atleast_2d(rig["k_prim"]))

    rr = ref.average_degenerate(1e-5)
    rw = res.average_degenerate(1e-5)
    assert rr.weights.shape == rw.weights.shape  # identical group counts
    dev = np.abs(rw.weights - rr.weights).max()
    assert dev < 1e-6, dev


def test_energies_fermi_shift(rig):
    """WFSX eigenvalues are SIESTA-native (Fermi-shifted as stored).

    The offset to the si_sc.HSX diagonalization equals the path run's
    own EIG-header Fermi (-3.00800582 eV) up to <=9e-4 eV of run-to-run
    SCF noise (si_sc_path is a separate SCF run from si_sc.HSX).
    """
    ref = rig["lcao"].compute(rig["k_prim"], method="ideal")
    res = rig["unf"].compute(rig["k_prim"], method="ideal")
    ef = read_fermi_energy(PATH_EIG)
    off = res.eigenvalues - ref.eigenvalues  # = +E_F plus cross-run noise
    assert abs(off.mean() - ef) < 1e-4, off.mean()
    assert np.abs(off - ef).max() < 1.5e-3


def test_same_run_energies_exact_at_gamma(rig):
    """Control: same-run WFSX vs parsed HSX differs by a single exact
    constant (the run's Fermi), proving the path-fixture scatter in
    test_energies_fermi_shift is cross-run SCF noise, not a parse
    defect. Returned eigenvalues stay Fermi-shifted; subtract E_F."""
    import sisl
    from scipy.linalg import eigh

    st = sisl.get_sile(SC_WFSX).read_eigenstate()
    e_st = np.asarray(st.eig, dtype=float)
    H, S = rig["hs_model"].hs_and_eigen(np.zeros((1, 3)))
    eps = eigh(np.asarray(H), np.asarray(S), eigvals_only=True)
    off = np.sort(eps) - np.sort(e_st)  # = -E_F of the si_sc run
    assert np.ptp(off) < 1e-8, off


def test_gauge_conversion_is_pinned(rig):
    """The per-orbital phase conversion is adjudicated, not optional.

    At generic k the adjudicated '+' conversion matches the HSX
    reference to 1e-6 while omitting the conversion or flipping the
    sign deviates by >1e-3 in degenerate-group weight sums. (At Gamma
    the factor is 1 for every sign, hence generic k is required.)
    """
    unf, lcao, wfsx = rig["unf"], rig["lcao"], rig["wfsx"]
    B = rig["B"]
    for j in (40, 90):
        k = rig["k_prim"][j]
        K = k @ B
        ref = lcao.compute(k[None, :], method="ideal")
        gr = group_sums(ref.eigenvalues[0], ref.weights[0], 1e-5)

        res = unf.compute(k[None, :], method="ideal")
        gw = group_sums(np.sort(res.eigenvalues[0]), res.weights[0], 1e-5)
        assert len(gw) == len(gr)
        assert np.abs(gw - gr).max() < 1e-6, (j, gw - gr)

        C_raw = np.asarray(wfsx.coefficients[j]).T.astype(complex)
        for sign, label in ((0.0, "no conversion"), (-1.0, "flipped sign")):
            C_bad = C_raw * unf._gauge_phase(K, sign)[:, None]
            w_bad = unf._weights_for_coefficients(k, C_bad, "ideal")
            gb = group_sums(np.sort(res.eigenvalues[0]), w_bad, 1e-5)
            dev = np.abs(gb - gr).max()
            assert dev > 1e-3, (label, j, dev)


def test_ring_method_at_gamma(rig):
    """The ring branch (self sector-Gram block) seals against the LCAO
    ring reference at Gamma, where the multi-atom generic-k gauge is
    exact (the same restriction as test_wfsx_gamma_weights_oracle)."""
    ref = rig["lcao"].compute(np.zeros((1, 3)), method="ring")
    res = rig["unf"].compute(np.zeros((1, 3)), method="ring")
    gr = ref.average_degenerate(1e-5).weights[0]
    gw = res.average_degenerate(1e-5).weights[0]
    assert gr.shape == gw.shape
    dev = np.abs(gr - gw).max()
    assert dev < 1e-6, dev


def test_unmatched_k_raises_keyerror(rig):
    """A primitive k whose supercell momentum is absent from the WFSX
    raises KeyError, listing the unmatched point."""
    k_bad = np.array([0.013, 0.027, 0.041])
    with pytest.raises(KeyError, match="no WFSX entry") as info:
        rig["unf"].compute(k_bad[None, :])
    msg = str(info.value)
    assert "0.013" in msg and "0.027" in msg  # primitive k listed
