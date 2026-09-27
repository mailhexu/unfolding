"""OpenMX-backend unfolding tests on the committed Si/Si:P fixtures.

Fixtures (tests/data/si_example/openmx_*.scfout) are genuine OpenMX
3.9 .scfout runs (Si7.0-s2p2d1 / P7.0-s2p2d1, spin unpolarized):
2-atom fcc primitive cell and 8-atom conventional cells (Si8 and
Si7P). The parser itself is validated in HamiltonIO's
tests/test_openmx_parser.py against OpenMX's own eigenvalue output;
these tests exercise the LCAOUnfolder through the
HamiltonIOModel adapter on top of that parser.

Note the OpenMX overlap matrix is mildly ill-conditioned at generic
momenta (cond ~ 6e3), so the ideal-method computes run with
``atol_orth=1e-6`` instead of the 1e-8 default.
"""

import os

import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(__file__), "data", "si_example")
B_DIAMOND = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
N_ORB = 13  # Si7.0-s2p2d1 / P7.0-s2p2d1 per atom


def _read_model(name):
    from HamiltonIO.openmx import OpenmxParser

    return OpenmxParser(os.path.join(DATA, name)).get_model()


def _make_unfolder(sc_name="openmx_si_sc.scfout", match_species=True):
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = _read_model("openmx_si_prim.scfout")
    sc = _read_model(sc_name)
    rm = RelabelMap.from_atoms(
        sc.atoms,
        prim.atoms,
        B_DIAMOND,
        orb_counts_sc=[N_ORB] * 8,
        orb_counts_prim=[N_ORB, N_ORB],
        match_species=match_species,
    )
    return LCAOUnfolder(HamiltonIOModel(sc), rm)


def _group_sums(e, w, tol=1e-6):
    order = np.argsort(e)
    es, ws = e[order], w[order]
    out, start = [], 0
    for i in range(1, len(es) + 1):
        if i == len(es) or es[i] - es[i - 1] > tol:
            out.append(float(ws[start:i].sum()))
            start = i
    return out


@pytest.fixture(scope="module")
def pristine():
    pytest.importorskip("HamiltonIO")
    return _make_unfolder()


@pytest.fixture(scope="module")
def doped():
    pytest.importorskip("HamiltonIO")
    return _make_unfolder("openmx_si_sc_p.scfout", match_species=False)


def test_openmx_model_adapter_shapes(pristine):
    from unfolding.lcao_unfolder import HamiltonIOModel

    hm = pristine._model
    assert len(hm.SR) == 343  # symmetric {-3..3}^3 image cube
    block = next(iter(hm.SR.values()))
    assert block.shape == (104, 104)
    assert hm.atoms.get_chemical_formula() == "Si8"


def test_pristine_gamma_ring_weights_are_integers(pristine):
    """Pristine supercell at Gamma: exact 0/1 ring projection per band.

    Degenerate-group sums must be integers (Parseval-exact torus
    projection on the commensurate momentum) and non-negative.
    """
    res = pristine.compute(np.atleast_2d(np.zeros(3)), method="ring")
    sums = _group_sums(res.eigenvalues[0], res.weights[0])
    assert np.abs(np.asarray(sums) - np.round(sums)).max() < 1e-6
    assert min(sums) > -1e-6
    # every primitive orbital family is recovered somewhere
    assert np.round(sum(sums)) == 26  # one 2-atom primitive cell worth


def test_pristine_generic_k_ideal_weights(pristine):
    """Ideal Popescu-Zunger weight is exactly 0/1 at a generic momentum."""
    res = pristine.compute(
        np.array([[0.13, 0.27, 0.41]]), method="ideal", atol_orth=1e-6
    )
    w = res.weights[0]
    assert np.isfinite(w).all()
    assert np.abs(w - np.round(w)).max() < 1e-6


def test_doped_gamma_ring_weights(doped):
    """Si7P at Gamma: host-like weight clusters just below 1 and dopant
    states carry strictly fractional weight (SCF reorganization around
    the donor smears the pristine 0/1 projection)."""
    res = doped.compute(np.atleast_2d(np.zeros(3)), method="ring")
    sums = np.asarray(_group_sums(res.eigenvalues[0], res.weights[0]))
    # host-like groups survive just under unit weight
    assert ((sums > 0.9) & (sums < 1.0)).sum() >= 8
    # dopant-derived groups carry strictly fractional weight
    fractional = sums[(sums > 0.02) & (sums < 0.9)]
    assert fractional.size >= 3
    assert sums.min() > -0.05
    assert sums.max() <= 2.0


def test_doped_generic_k_ideal_finite(doped):
    """The ideal definition stays finite and bounded for the dopant."""
    res = doped.compute(
        np.array([[0.13, 0.27, 0.41]]), method="ideal", atol_orth=1e-6
    )
    w = res.weights[0]
    assert np.isfinite(w).all()
    assert w.min() > -1e-2
    assert w.max() <= 1.0 + 1e-6


def test_unfold_openmx_references_energies_to_fermi():
    """unfold_openmx shifts eigenvalues by -E_F so 0 in the figure is E_F.

    The OpenMX scfout stores absolute Hartree energies (ChemP ~ -2.69 eV
    for the pristine run); without the shift, the dashed E_F line drawn
    at y=0 sits ~2.7 eV below the true Fermi level.
    """
    from ase import Atoms
    from ase.units import Ha

    from unfolding import unfold_openmx

    a = 5.43
    prim = Atoms(
        "Si2",
        scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
        cell=[[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]],
        pbc=True,
    )
    kpts = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.25], [0.5, 0.0, 0.5]])
    kw = dict(
        prim_atoms=prim,
        unfold_sc_mat=B_DIAMOND,
        kpts=kpts,
        method="ideal",
        atol_orth=1e-6,
    )

    # raw absolute energies: a parsed model bypasses the ChemP lookup
    sc_model = _read_model("openmx_si_sc.scfout")
    from HamiltonIO.openmx import OpenmxParser

    ef = OpenmxParser(os.path.join(DATA, "openmx_si_sc.scfout")).efermi
    ax_raw = unfold_openmx(model=sc_model, efermi=0.0, **kw)
    # default path: scfout is parsed here and ChemP is picked up
    ax_ef = unfold_openmx(
        scfout=os.path.join(DATA, "openmx_si_sc.scfout"), **kw
    )

    y_raw = np.concatenate(
        [ln.get_ydata() for ln in ax_raw.lines if len(ln.get_ydata()) == len(kpts)]
    )
    y_ef = np.concatenate(
        [ln.get_ydata() for ln in ax_ef.lines if len(ln.get_ydata()) == len(kpts)]
    )
    np.testing.assert_allclose(y_ef - y_raw, -ef, atol=1e-8)

    # shifted valence branches lie below 0 and the lowest conduction
    # branches above: E_F (OpenMX ChemP) sits inside the Si gap
    y_bands = y_ef.reshape(-1, len(kpts))
    assert y_bands[:16].max() < 0.0  # 32 valence electrons / 2
    assert y_bands[16:32].min() > 0.0
    assert abs(ef - -0.09902050121809 * Ha) < 1e-6
