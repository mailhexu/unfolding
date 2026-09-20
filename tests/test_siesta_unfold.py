"""Story-008 tests: unfold_siesta one-call adapter."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from ase import Atoms
from numpy.linalg import eigh

from unfolding import unfold_siesta

N_CELLS = 4


def _prim_tables():
    """Story-2 canonical primitive tables (Hermitian: M[-1] = M[1].T)."""
    from unfolding.toy_models import toy_lcao_model

    m = toy_lcao_model()
    return m["H"], m["S"]


class _RawChannel:
    """HamiltonIO-compatible single spin channel (batch HS_and_eigen)."""

    def __init__(self, scale, atoms, orb_dict):
        H, S = _prim_tables()
        L = N_CELLS
        self._H, self.SR = {}, {}
        for T in (-1, 0, 1):
            Mh = np.zeros((2 * L, 2 * L))
            Ms = np.zeros((2 * L, 2 * L))
            for i in range(L):
                for j in range(L):
                    d = T * L + j - i
                    if abs(d) <= 1:
                        Mh[2 * i:2 * i + 2, 2 * j:2 * j + 2] = H[d] * scale
                        Ms[2 * i:2 * i + 2, 2 * j:2 * j + 2] = S[d]
            if np.any(Mh):
                self._H[T] = Mh
            if np.any(Ms):
                self.SR[T] = Ms
        self.atoms = atoms
        self.orb_dict = orb_dict
        self._n = 2 * len(atoms)

    def HS_and_eigen(self, kpts):
        kpts = np.atleast_2d(np.asarray(kpts, dtype=float))
        Hs = np.zeros((len(kpts), self._n, self._n), complex)
        Ss = np.zeros_like(Hs)
        for ik, k in enumerate(kpts):
            for T, block in self._H.items():
                Hs[ik] += block * np.exp(2j * np.pi * k[0] * T)
            for T, block in self.SR.items():
                Ss[ik] += block * np.exp(2j * np.pi * k[0] * T)
        return Hs, Ss


class _CollinearParsed:
    """Parsed-model stand-in exposing up/down channel objects."""

    def __init__(self, up, down):
        self.up = up
        self.down = down


@pytest.fixture()
def parsed_collinear():
    prim = Atoms(
        "Si", positions=[[0.0, 0, 0]], cell=[[1.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]
    )
    sc = prim.repeat((N_CELLS, 1, 1))
    orb_dict = {i: ["2s", "2p"] for i in range(len(sc))}
    up = _RawChannel(1.0, sc, orb_dict)
    down = _RawChannel(1.1, sc, orb_dict)
    return _CollinearParsed(up, down), prim


def _kpath():
    t = np.linspace(0.0, 0.5, 9)
    kpts = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    return kpts, np.arange(len(kpts))


def test_unfold_siesta_returns_weighted_axes(parsed_collinear):
    """TEST-001: end-to-end adapter on a toy HamiltonIO-compatible model."""
    from unfolding.lcao_unfolder import HamiltonIOModel
    from unfolding.mapping import RelabelMap

    parsed, prim = parsed_collinear
    kpts, xqpts = _kpath()
    ax = unfold_siesta(
        model=parsed,
        prim_atoms=prim,
        unfold_sc_mat=np.diag([N_CELLS, 1, 1]),
        kpts=kpts,
        xqpts=xqpts,
        knames=["G", "X"],
        Xqpts=[xqpts[0], xqpts[-1]],
    )
    # one colored LineCollection per supercell band (8 = 2 orbitals x 4 cells)
    assert len(ax.collections) == 8
    # axes labeled in eV, tick positions honored
    assert "eV" in ax.get_ylabel()
    np.testing.assert_allclose(ax.get_xticks(), [0, 8])
    assert ax.get_xticklabels()[0].get_text() == "G"

    # energies drawn are the adapted model's eigenvalues
    rm = RelabelMap.from_atoms(
        parsed.up.atoms, prim, np.diag([N_CELLS, 1, 1]),
        orb_counts_sc=[2] * N_CELLS, orb_counts_prim=[2],
    )
    from unfolding.lcao_unfolder import LCAOUnfolder

    res = LCAOUnfolder(HamiltonIOModel(parsed.up), rm).compute(kpts)
    seg_y = np.concatenate(
        [np.asarray(c.get_segments())[..., 1].ravel() for c in ax.collections]
    )
    assert np.isclose(seg_y.min(), res.eigenvalues.min(), atol=1e-9)
    assert np.isclose(seg_y.max(), res.eigenvalues.max(), atol=1e-9)


def test_unfold_siesta_selects_spin_channel(parsed_collinear):
    """spin='down' routes to the down channel (different energies)."""
    parsed, prim = parsed_collinear
    kpts, xqpts = _kpath()
    kw = dict(
        model=parsed, prim_atoms=prim,
        unfold_sc_mat=np.diag([N_CELLS, 1, 1]), kpts=kpts, xqpts=xqpts,
    )
    ax_up = unfold_siesta(spin="up", **kw)
    ax_down = unfold_siesta(spin="down", **kw)
    y_up = np.concatenate([np.asarray(c.get_segments())[..., 1].ravel() for c in ax_up.collections])
    y_dn = np.concatenate([np.asarray(c.get_segments())[..., 1].ravel() for c in ax_down.collections])
    assert not np.isclose(y_up.mean(), y_dn.mean())


def test_unfold_siesta_error_paths(parsed_collinear, tmp_path):
    """TEST-002: missing file, missing backend, and missing-argument errors."""
    parsed, prim = parsed_collinear
    kpts, xqpts = _kpath()

    with pytest.raises(ValueError, match="either fdf or a parsed model"):
        unfold_siesta()
    with pytest.raises(FileNotFoundError):
        unfold_siesta(fdf=str(tmp_path / "nope.fdf"), prim_atoms=prim,
                      unfold_sc_mat=np.diag([N_CELLS, 1, 1]), kpts=kpts)
    with pytest.raises(ValueError, match="prim_atoms"):
        unfold_siesta(model=parsed, unfold_sc_mat=np.diag([N_CELLS, 1, 1]), kpts=kpts)
    with pytest.raises(ValueError, match="unfold_sc_mat"):
        unfold_siesta(model=parsed, prim_atoms=prim, kpts=kpts)
    with pytest.raises(ValueError, match="kpts"):
        unfold_siesta(model=parsed, prim_atoms=prim, unfold_sc_mat=np.diag([4, 1, 1]))

    # fdf present but HamiltonIO import blocked (simulated absence)
    fdf = tmp_path / "run.fdf"
    fdf.write_text("SystemName toy\n")
    import sys

    mp = pytest.MonkeyPatch()
    mp.setitem(sys.modules, "HamiltonIO", None)
    mp.setitem(sys.modules, "HamiltonIO.siesta", None)
    try:
        with pytest.raises(ImportError, match="pip install HamiltonIO sisl"):
            unfold_siesta(fdf=str(fdf), prim_atoms=prim,
                          unfold_sc_mat=np.diag([N_CELLS, 1, 1]), kpts=kpts)
    finally:
        mp.undo()


def test_unfold_siesta_importable_from_package():
    """Lazy export resolves; the module itself imports without HamiltonIO."""
    import unfolding

    assert "unfold_siesta" in dir(unfolding)
    assert callable(unfolding.unfold_siesta)
