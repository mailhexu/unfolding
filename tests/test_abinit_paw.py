"""Real JTH-PAW WFK unfolding: metric norm, pristine and Si:P defect."""
from pathlib import Path

import numpy as np
import pytest

from HamiltonIO.abinit import HARTREE_TO_EV, read_paw_wfk, read_wfk
from unfolding.abinit_paw import unfold_abinit_paw
from unfolding.abinit_unfold import unfold_abinit
from unfolding.pw_unfolder import PWEigenData, PWUnfolder

DATA = Path(__file__).resolve().parent / "data/abinit_paw"
XML = {symbol: DATA / f"{symbol}.xml" for symbol in ("Si", "P")}
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
PRIM = DATA / "si_primitive_foldso_WFK.nc"
SI = DATA / "si8_gammao_WFK.nc"
DOPED = DATA / "si7p_gammao_WFK.nc"
NC = Path(__file__).resolve().parent / "data/abinit_si"
NC_PRIM = NC / "si_primitive_matchedo_WFK.nc"
NC_DOPED = NC / "si7p_gammao_WFK.nc"
# Primitive momenta folding onto supercell Gamma (fcc X, Y, Z sectors).
KSECT = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
DEGEN_HA = 1e-3 / HARTREE_TO_EV
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in [*XML.values(), PRIM, SI, DOPED]),
    reason="ABINIT PAW WFK and XML fixtures absent",
)


def _degenerate_blocks(energies, tol):
    """Band index arrays of consecutive eigenvalues split by more than tol."""
    boundaries = np.r_[0, np.flatnonzero(np.diff(energies) > tol) + 1, len(energies)]
    return [np.arange(a, b) for a, b in zip(boundaries[:-1], boundaries[1:])]



def test_reader_rejects_nonpaw_wfk():
    norm_conserving = DATA.parent / "abinit_si/si8_gammao_WFK.nc"
    with pytest.raises(ValueError, match="not a PAW calculation"):
        read_paw_wfk(norm_conserving, XML)


def test_pristine_reference_folds_have_paw_unit_norm_and_binary_weights():
    data = read_paw_wfk(SI, XML)
    assert data.wavefunctions.usepaw == 1
    assert data.species == ("Si",) * 8
    with pytest.raises(ValueError, match="requires unfold_abinit_paw"):
        unfold_abinit(data=data.wavefunctions, unfold_sc_mat=M, kpts=[[0, 0, 0]])
    result = unfold_abinit_paw(data, PRIM, XML, M)
    assert np.max(np.abs(result.norm_residuals)) < 2e-6
    np.testing.assert_allclose(result.weights.sum(axis=0), 1, atol=2e-6)
    resolved = unfold_abinit_paw(data, PRIM, XML, M, resolve_degenerate=1e-5)
    assert np.max(np.minimum(np.abs(resolved.weights), np.abs(1-resolved.weights))) < 2e-6
    # Gauge rotation reorders states within each degenerate group independently;
    # raw per-band sums above, not resolved column labels, obey Parseval.

def test_dopant_changes_paw_weights_relative_to_pseudo_cosets():
    result = unfold_abinit_paw(DOPED, PRIM, XML, M)
    assert np.max(np.abs(result.norm_residuals)) < 2e-6
    assert np.max(np.abs(result.weights - result.pseudo_weights)) > 0.005
    donor = result.weights[:, 16]
    assert 0.04 < donor.min() < 0.1
    assert 0.2 < donor.max() < 0.4


def test_resolution_removes_degenerate_gauge_scrambling():
    """Raw Gamma-point PAW weights are gauge-scrambled; resolved are binary.

    ABINIT stores the supercell-Gamma degenerate manifolds in an arbitrary
    unitary gauge, so raw per-band projector diagonals scatter far from the
    binary pristine sectors. Restricting the reference projector and the
    pseudo-coset operator to each degenerate eigenspace yields gauge-invariant
    populations without touching the stored eigenvectors.
    """
    raw = unfold_abinit_paw(SI, PRIM, XML, M)
    resolved = unfold_abinit_paw(SI, PRIM, XML, M, resolve_degenerate=DEGEN_HA)
    assert np.max(np.minimum(np.abs(raw.weights), np.abs(1 - raw.weights))) > 0.1
    for weights in (resolved.weights, resolved.pseudo_weights):
        assert np.max(np.minimum(np.abs(weights), np.abs(1 - weights))) < 1e-2
    blocks = _degenerate_blocks(resolved.eigenvalues[0], DEGEN_HA)
    for block in blocks:
        if len(block) > 1:
            np.testing.assert_allclose(
                raw.weights[:, block].sum(axis=1),
                resolved.weights[:, block].sum(axis=1), atol=1e-8)
    singles = [band for block in blocks if len(block) == 1 for band in block]
    np.testing.assert_allclose(
        raw.weights[:, singles], resolved.weights[:, singles], atol=1e-10)


@pytest.mark.skipif(
    not (NC_PRIM.exists() and NC_DOPED.exists()),
    reason="norm-conserving ABINIT Si7P fixtures absent",
)
def test_paw_route_tracks_norm_conserving_route():
    """PAW eigenvalues trace the NC dispersion; Si:P sector patterns agree.

    Different pseudos shift absolute energies; after removing the median
    offset the unfolded Si7P band positions must match in shape, and the
    gauge-invariant degenerate-block weight patterns must agree sector by
    sector (the two runs put the P on cubic-equivalent sites, so X/Y/Z
    multiplets are comparable block-wise, not band-wise).
    """
    paw = unfold_abinit_paw(DOPED, PRIM, XML, M, resolve_degenerate=DEGEN_HA)
    nc = read_wfk(NC_DOPED)
    nc_result = PWUnfolder(PWEigenData(nc.kpoints, nc.gvecs, nc.coefficients,
                                       nc.eigenvalues), M).compute(
        KSECT, resolve_degenerate=DEGEN_HA)
    defect = read_paw_wfk(DOPED, XML)
    paw_e = np.sort((paw.eigenvalues[0] - defect.wavefunctions.fermi_energy)
                    * HARTREE_TO_EV)
    nc_e = np.sort((nc.eigenvalues[0][0] - nc.fermi_energy) * HARTREE_TO_EV)
    shift = np.median(paw_e - nc_e)
    assert np.abs(paw_e - nc_e - shift).max() < 0.3

    paw_blocks = _degenerate_blocks(paw.eigenvalues[0], DEGEN_HA)
    nc_blocks = _degenerate_blocks(nc.eigenvalues[0][0], DEGEN_HA)
    assert [len(b) for b in paw_blocks] == [len(b) for b in nc_blocks]
    paw_w = paw.weights
    nc_w = np.asarray(nc_result.weights)
    for paw_block, nc_block in zip(paw_blocks, nc_blocks):
        np.testing.assert_allclose(
            paw_w[:, paw_block].sum(axis=1), nc_w[:, nc_block].sum(axis=1),
            atol=0.05)
    # The donor level is a non-degenerate band just above the VBM region in
    # both runs: delocalized donor weight, small on Gamma, ~0.3 on X/Y/Z.
    donor_blocks = [block for block in paw_blocks
                    if len(block) == 1 and -0.1 < paw_e[block[0]] < 0.1]
    assert len(donor_blocks) == 1
    donor = paw_w[:, donor_blocks[0][0]]
    assert donor[0] < 0.2
    assert np.all((donor[1:] > 0.2) & (donor[1:] < 0.4))
    host = [block[0] for block in paw_blocks
            if len(block) == 1 and -13.5 < paw_e[block[0]] < -12.0]
    assert host and paw_w[0, host[0]] > 0.8


# --- dense 305-point GXWGLWX path fixtures (external, gitignored) ---
PATH_PRIM = DATA / "si_prim_paw_patho_DS2_WFK.nc"
PATH_SI = DATA / "si8_paw_patho_DS2_WFK.nc"
PATH_DOPED = DATA / "si7p_paw_patho_DS2_WFK.nc"
NC_DENSE = NC / "si8_gxwglwxo_DS2_WFK.nc"
NPATH = 305

_path_guard = pytest.mark.skipif(
    not all(path.exists() for path in (PATH_PRIM, PATH_SI, PATH_DOPED)),
    reason="dense-path PAW WFKs absent; regenerate with "
           "tests/data/abinit_paw/regenerate_path_on_nic6.sh",
)


def _load_path_example():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "abinit_paw_unfold_path",
        Path(__file__).resolve().parents[1] / "examples/abinit_paw/unfold_path.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dense_path():
    """(example module, primitive, {name: (result, energies eV vs E_F)})."""
    if not all(path.exists() for path in (PATH_PRIM, PATH_SI, PATH_DOPED)):
        pytest.skip("dense-path PAW WFKs absent; regenerate with "
                    "tests/data/abinit_paw/regenerate_path_on_nic6.sh")
    example = _load_path_example()
    primitive = read_paw_wfk(PATH_PRIM, XML)
    results = {
        name: example.unfold_system(path, primitive)
        for name, path in (("si8", PATH_SI), ("si7p", PATH_DOPED))
    }
    return example, primitive, results


@_path_guard
def test_dense_path_axes_and_fixture_shape(dense_path):
    example, primitive, results = dense_path
    prim_k = primitive.wavefunctions.kpoints
    assert prim_k.shape == (NPATH, 3)
    x = example.path_positions(prim_k)
    ticks = example.path_ticks(prim_k, x)
    assert len(ticks) == len(example.PATH) == 7
    assert ticks[0] == 0.0 and ticks[-1] == x[-1]
    assert np.all(np.diff(ticks) > 0)
    for result, energies in results.values():
        assert result.weights.shape == (NPATH, 32)
        assert energies.shape == (NPATH, 32)
        assert np.max(np.abs(result.norm_residuals)) < 1e-5


@_path_guard
def test_dense_path_pristine_weights_stay_binary(dense_path):
    """Away from Gamma the 4 sectors no longer share an eigenspace, but the
    resolved pristine weights must still be binary at every generic point.

    Scoped to the converged window: the top of the 32-band manifold
    (above ~7 eV) is Davidson-tail and not weight-converged.
    """
    example, _, results = dense_path
    result, energies = results["si8"]
    window = (example.YRANGE[0], example.CONVERGED_EV)
    # Measured worst 9.9e-4 (w = 0.999): every non-degenerate band point
    # below 6 eV is binary to within the 1e-3 classifier tolerance.
    assert example.binary_error(result, energies, window=window) < 2e-3
    assert example.binary_fraction(result, energies, window=window) > 0.99
    frac_e, frac_w = example.fractional_weights(result, energies, window=window)
    assert frac_w.size == 0


@_path_guard
def test_dense_path_donor_sector_fractions(dense_path):
    """Si7P: the donor window carries fractional single-sector weights.

    The substitutional P breaks the primitive-cell translations that
    define the sectors, so sector states mix along the whole path: the
    donor window is densely fractional (donor charge spread over the four
    sectors, median ~0.36 as in the Gamma-fold fixture), and even host
    bands anticross into fractional states at isolated crossings.
    """
    example, _, results = dense_path
    result, energies = results["si7p"]
    window = (example.YRANGE[0], example.CONVERGED_EV)
    frac_e, frac_w = example.fractional_weights(result, energies, window=window)
    assert frac_w.size > 2000
    assert np.all((frac_w > 0.05) & (frac_w < 0.95))
    de, dw = example.fractional_weights(result, energies, window=(-1.5, 2.5))
    assert de.size > 300
    assert np.all((de > -1.5) & (de < 2.5))
    assert 0.2 < np.median(dw) < 0.45
    # The pristine reference system has no such mixing below 6 eV.
    result8, energies8 = results["si8"]
    _, frac8 = example.fractional_weights(
        result8, energies8, window=window)
    assert frac8.size == 0


@pytest.mark.skipif(
    not (NC_DENSE.exists() and PATH_PRIM.exists() and PATH_SI.exists()),
    reason="dense PAW or norm-conserving path WFKs absent",
)
def test_dense_path_paw_bands_track_norm_conserving_bands():
    """PAW and NC unfolded band positions agree on the shared 305-point path.

    Both runs used a Gamma-only SCF density and the same Si8 geometry, so
    after the global median shift the high-weight (binary) band positions
    differ only by pseudo/PAW convergence (ecut 10 vs 25). Sorted-energy
    matching can mispair conduction bands at crossings, so the median
    deviation is pinned much tighter than the max.
    """
    example = _load_path_example()
    primitive = read_paw_wfk(PATH_PRIM, XML)
    result, energies = example.unfold_system(PATH_SI, primitive)
    max_dE, shift, med_dE = example.compare_norm_conserving(result, energies, NC_DENSE)
    assert np.abs(shift) < 0.2
    assert max_dE < 0.25
    assert med_dE < 0.05
