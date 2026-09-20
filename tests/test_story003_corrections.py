"""Story 003 corrections: deprecation, wannier module repair, unit constants."""
import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unfolder_deprecation_warning():
    from unfolding.unfolder import Unfolder

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Unfolder(
            cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            basis=["s"],
            positions=[[0.0, 0.0, 0.0]],
            supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            eigenvectors=np.zeros((1, 1, 1)),
            qpoints=np.zeros((1, 3)),
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in record)


class _ToyTBModel:
    """Minimal real tight-binding model for WannierUnfolder (2 orbitals, 1D).

    Implements the duck type wannier_unfold expects: .atoms (ASE Atoms),
    ._orb (scaled orbital positions), .solve_all(k_list=, eig_vectors=)
    returning (evals [nband, nk], evecs [nband, nk, norb]) — the minimulti
    MyTB convention.
    """

    atoms = None
    _orb = None

    def __init__(self):
        from ase.atoms import Atoms

        from unfolding.toy_models import toy_lcao_model

        self._m = toy_lcao_model()
        self.atoms = Atoms("Si", cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], positions=[[0, 0, 0]])
        self._orb = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def solve_all(self, k_list, eig_vectors=True):
        evals = np.zeros((2, len(k_list)))
        evecs = np.zeros((2, len(k_list), 2), dtype=complex)
        from scipy.linalg import eigh

        for ik, k in enumerate(k_list):
            k1d = np.atleast_1d(np.asarray(k))[0]  # 1D model: first component
            w, v = eigh(self._m["H_of_k"](k1d), self._m["S_of_k"](k1d))
            evals[:, ik] = w
            v = v.T  # (nband, norb)
            # eigh(H,S) vectors are S-orthonormal; the legacy Unfolder weight
            # assumes Euclidean-normalized columns, so renormalize here. The
            # S-metric-correct weight is implemented by story 007 (Eq. 25).
            v = v / np.sqrt(np.einsum("bo,bo->b", v.conj(), v))[:, None]
            evecs[:, ik, :] = v
        return evals, evecs


def test_wannier_module_imports_and_unfolds():
    # minimulti / MyTB are NOT installed: the module must still import, and
    # the unfolder must work with any injected tight-binding model.
    assert importlib.util.find_spec("minimulti") is None
    from unfolding.wannier_unfold import WannierUnfolder

    kpts = np.array([[k / 8, 0, 0] for k in range(-4, 5)])
    u = WannierUnfolder(_ToyTBModel(), labels=["s", "p"], sc_matrix=np.eye(3, dtype=int))
    weights = u.unfold(kpts)
    assert weights.shape == (len(kpts), 2)
    # pristine single-cell "supercell": everything folds back with w = 1
    assert np.allclose(weights, 1.0, atol=1e-10)


def test_no_syntax_warnings_compiling_package_sources():
    src = REPO_ROOT / "unfolding"
    r = subprocess.run(
        [sys.executable, "-W", "error::SyntaxWarning", "-W", "error::DeprecationWarning", "-c",
         "import compileall, sys; sys.exit(0 if compileall.compile_dir(sys.argv[1], quiet=2, force=True) else 1)",
         str(src)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr


def test_unit_conversion_constants_are_derived():
    from unfolding.units import EV_TO_CM, THZ_TO_CM

    # CODATA-derived values (independent of ase): 1 THz and 1 eV in cm^-1
    assert THZ_TO_CM == pytest.approx(33.35640951, rel=1e-6)
    assert EV_TO_CM == pytest.approx(8065.544005, rel=1e-6)


def test_adapters_use_shared_units():
    from unfolding import units

    pytest.importorskip("phonopy")
    phonopy_mod = importlib.import_module("unfolding.phonopy_unfolder")
    assert phonopy_mod.THZ_TO_CM is units.THZ_TO_CM
    pytest.importorskip("abipy")
    ddb_mod = importlib.import_module("unfolding.DDB_unfolder")
    assert ddb_mod.EV_TO_CM is units.EV_TO_CM


def test_wannier_plot_path_produces_lines():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from unfolding.wannier_unfold import WannierUnfolder

    fig, ax = plt.subplots()
    u = WannierUnfolder(_ToyTBModel(), labels=["s", "p"], sc_matrix=np.eye(3, dtype=int))
    ret = u.plot_unfolded_band(npoints=24, ax=ax)
    collections = [c for c in ret.collections if isinstance(c, LineCollection)]
    assert len(collections) >= 2
    # embedded-axis styling must be applied (review F-007): knames as tick labels
    labels = [t.get_text() for t in ret.get_xticklabels()]
    assert any("Gamma" in lab or "\u0393" in lab for lab in labels if lab), labels
    # ypad must be in data units (review R2-001): bands must not be squashed
    lo, hi = ret.get_ylim()
    assert lo > -5.0 and hi < 5.0, (lo, hi)  # toy bands span ~(-1.2, 1.2) eV
    plt.close(fig)


def test_ddb_atomic_masses_come_from_ase():
    pytest.importorskip("abipy")
    from ase.data import atomic_masses

    from unfolding.DDB_unfolder import atomic_masses

    assert atomic_masses[29] == pytest.approx(63.546, abs=0.1)  # Cu
