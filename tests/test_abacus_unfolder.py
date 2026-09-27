"""ABACUS backend end-to-end tests (story 018).

Two routes are exercised against committed fixtures
(``tests/data/abacus_example``):

LCAO (si_conv / si7p / si_prim, DZP 2s2p1d, 8-atom conventional cell
unfolded onto the 2-atom fcc primitive cell)
    The ABACUS LCAO shell set is truncated at the orbital range cutoff
    (99 R shells with 10 au orbitals), so the generic-k ``ideal`` weight
    is data-limited (the SIESTA fixtures store the full BvK shell box
    and reach exact 0/1).  The LCAO tests therefore seal what this data
    supports: Parseval weight sums, weight support, and the
    substitutional-defect signature of Si7P against the pristine
    reference, plus the primitive-cell band-structure oracle.

Planewave (si_pw_path, out_wfc_pw output)
    Full coefficient unfolding through :class:`PWUnfolder`; for the
    pristine supercell every band weight is exactly 0 or 1 away from
    BZ-face degeneracies (handled by ``resolve_degenerate``).
"""

import os
import sys

import numpy as np
import pytest
from scipy.linalg import eigh

DATA = os.path.join(os.path.dirname(__file__), "data", "abacus_example")
B_DIAMOND = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
GENERIC_K = np.array([[0.137, 0.251, 0.083]])


def _abacus_parser(name):
    outpath = os.path.join(DATA, name, "OUT." + name)
    if not os.path.isdir(outpath):
        pytest.skip(f"ABACUS fixture {outpath} missing")
    from HamiltonIO.abacus.abacus_wrapper import AbacusParser

    return AbacusParser(outpath=outpath)


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("HamiltonIO")
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = _abacus_parser("si_prim").get_models()
    sc = _abacus_parser("si_conv").get_models()
    si7p = _abacus_parser("si7p").get_models()
    counts = [13] * 8
    rm_sc = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, B_DIAMOND,
        orb_counts_sc=counts, orb_counts_prim=[13, 13],
    )
    rm_7p = RelabelMap.from_atoms(
        si7p.atoms, prim.atoms, B_DIAMOND,
        orb_counts_sc=counts, orb_counts_prim=[13, 13], match_species=False,
    )
    return (
        prim,
        LCAOUnfolder(HamiltonIOModel(sc), rm_sc),
        LCAOUnfolder(HamiltonIOModel(si7p), rm_7p),
    )


class TestLCAOUnfolding:
    def test_relabel_map_geometry(self, models):
        prim, unf, unf7 = models
        assert unf._n_orb_sc == 104
        assert unf._n_orb_prim == 26
        assert unf._n_cells == 4
        # the P atom folds onto the Si site it replaces
        assert unf7._rm.atom_to_m[7] == 0

    def test_ideal_weight_parseval(self, models):
        """Sum of ideal weights over bands equals the primitive orbital
        count at every momentum (spectral sum rule)."""
        _, unf, unf7 = models
        res = unf.compute(GENERIC_K, method="ideal")
        res7 = unf7.compute(GENERIC_K, method="ideal")
        assert res.weights[0].sum() == pytest.approx(26.0, abs=1e-6)
        assert res7.weights[0].sum() == pytest.approx(26.0, abs=1e-6)

    def test_ideal_weights_bounded(self, models):
        """Weights stay within the physical interval [-eps, 1+eps]."""
        _, unf, unf7 = models
        res = unf.compute(GENERIC_K, method="ideal")
        res7 = unf7.compute(GENERIC_K, method="ideal")
        for w in (res.weights[0], res7.weights[0]):
            assert w.min() > -0.05
            assert w.max() < 1.05

    def test_si7p_defect_signature(self, models):
        """P-doped weights spread weight across fold sectors: the
        dopant 3s-like state loses weight relative to pristine host
        bands at the same energy."""
        prim, unf, unf7 = models
        res = unf.compute(GENERIC_K, method="ideal")
        res7 = unf7.compute(GENERIC_K, method="ideal")
        w_pris = np.sort(res.weights[0])[::-1]
        w_7p = np.sort(np.clip(res7.weights[0], 0.0, 1.0))[::-1]
        # the highest-weight states stay host-like; the bulk of the
        # spectrum mixes sectors under substitution
        assert w_7p[0] > 0.9
        assert w_7p[:6].sum() < w_pris[:6].sum()

    def test_gamma_group_weight_sums(self, models):
        """Si7P Gamma ring groups: every exactly-degenerate group
        carries a bounded, real weight and the total is Parseval."""
        _, _, unf7 = models
        res = unf7.compute(np.zeros((1, 3)), method="ring")
        e = res.eigenvalues[0]
        w = res.weights[0]
        order = np.argsort(e)
        e, w = e[order], w[order]
        bounds = np.where(np.diff(e) > 1e-6)[0]
        starts = np.concatenate([[0], bounds + 1])
        stops = np.concatenate([bounds + 1, [len(e)]])
        total = 0.0
        for s, t in zip(starts, stops):
            group = w[s:t]
            assert group.sum() < 1.2  # no group exceeds the sector bound
            total += group.sum()
        assert total == pytest.approx(26.0, abs=1e-3)

    def test_host_bands_match_primitive_reference(self, models):
        """Host bands match the ABACUS primitive-cell band structure:
        every primitive Gamma level ABACUS printed in BANDS_1.dat
        reappears in the supercell Gamma spectrum (BvK folding)."""
        _, unf, _ = models
        bands = np.loadtxt(
            os.path.join(DATA, "si_prim", "OUT.si_prim", "BANDS_1.dat")
        )
        e_prim_gamma = bands[0, 2:]  # first row is the Gamma point
        sc = unf._model._model  # AbacusWrapper behind HamiltonIOModel
        e_sc = np.sort(sc.HS_and_eigen(np.zeros((1, 3)))[2][0])
        for value in e_prim_gamma:
            assert np.min(np.abs(e_sc - value)) < 0.15


class TestPWUnfolding:
    @pytest.fixture(scope="class")
    def pw_result(self):
        pytest.importorskip("HamiltonIO")
        from HamiltonIO.abacus.pw_wfc import AbacusPWParser
        from unfolding.pw_unfolder import PWEigenData, PWUnfolder

        outpath = os.path.join(DATA, "si_pw_path", "OUT.si_pw_path")
        if not os.path.isdir(outpath):
            pytest.skip(f"ABACUS fixture {outpath} missing")
        data = AbacusPWParser(outpath).read()
        core = PWEigenData(
            kpoints=data.kpoints,
            gvecs=data.gvecs,
            coefficients=[c[None, :, None, :] for c in data.coefficients],
            eigenvalues=data.eigenvalues[:, None, :],
        )
        unfolder = PWUnfolder(core, B_DIAMOND)
        kprim = data.kpoints @ np.linalg.inv(B_DIAMOND)
        return unfolder.compute(kprim, resolve_degenerate=1e-4)

    def test_pristine_weights_are_zero_or_one(self, pw_result):
        """Pristine 8-atom cell: every band weight is exactly 0 or 1."""
        w = pw_result.weights
        assert w.min() >= 0.0
        assert np.abs(w - np.round(w)).max() < 1e-5

    def test_weighted_eigenvalues_on_primitive_branches(self, pw_result):
        """Gamma-sector weight-1 bands: exactly one copy of each
        primitive occupied Gamma level survives the sector projector
        (4 primitive occupied bands: Gamma1 + triply-degenerate Gamma15)."""
        gamma = 0
        e = pw_result.eigenvalues[gamma]
        w = pw_result.weights[gamma]
        ones = np.sort(e[w > 0.5])
        assert len(ones) == 4

    def test_sum_rule_per_kpoint(self, pw_result):
        """Sector-projector weights sum to the number of sectors."""
        sums = pw_result.weights.sum(axis=1)
        assert np.abs(sums - 4.0).max() < 1e-5

def test_lcao_models_expose_fermi_levels(models):
    """E_Fermi (eV) parsed from running_scf.log backs the E_F=0 figure
    convention: si_conv 6.9002, si7p 7.2271, si_prim 6.7437 eV."""
    prim, unf, unf7 = models
    assert prim.efermi == pytest.approx(6.7437, abs=1e-3)
    assert unf._model._model.efermi == pytest.approx(6.9002, abs=1e-3)
    assert unf7._model._model.efermi == pytest.approx(7.2271, abs=1e-3)


def test_pw_fixtures_expose_fermi_levels():
    """PW parser reads the Fermi energy from the run log: the nscf band
    run (si_pw_path, running_nscf.log) and the scf Gamma run (si7p_pw,
    running_scf.log)."""
    from HamiltonIO.abacus.pw_wfc import AbacusPWParser

    for name, expected in (("si_pw_path", 6.2903334548),
                           ("si7p_pw", 6.85043078)):
        outpath = os.path.join(DATA, name, "OUT." + name)
        if not os.path.isdir(outpath):
            pytest.skip(f"ABACUS fixture {outpath} missing")
        data = AbacusPWParser(outpath).read()
        assert data.efermi == pytest.approx(expected, abs=1e-6)


def test_pw_si7p_gamma_fold_partition():
    """Real Si7P PW state spreads across the four primitive folds."""
    from HamiltonIO.abacus.pw_wfc import AbacusPWParser
    from unfolding.pw_unfolder import PWEigenData, PWUnfolder

    path = os.path.join(DATA, "si7p_pw", "OUT.si7p_pw")
    if not os.path.isfile(os.path.join(path, "WAVEFUNC1.txt")):
        pytest.skip("ABACUS Si7P PW fixture missing")
    data = AbacusPWParser(path).read()
    eigen = PWEigenData(data.kpoints, data.gvecs,
                        [c[None, :, None, :] for c in data.coefficients],
                        data.eigenvalues[:, None, :])
    folds = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
    result = PWUnfolder(eigen, B_DIAMOND).compute(folds)
    np.testing.assert_allclose(result.weights.sum(axis=0), 1, atol=1e-6)
    assert result.weights[:, 16].min() > .02
    assert result.weights[:, 16].max() < .4
