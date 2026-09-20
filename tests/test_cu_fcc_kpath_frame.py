"""Cu_fcc example: the k-path must be expressed in the DDB structure's frame.

The out_DDB fixture stores the conventional cubic fcc cell (natom=4), so
abipy interprets ``kpath_bounds`` in the conventional-cubic reciprocal basis
(X=(0,1,0), W=(1/2,1,0), L=(1/2,1/2,1/2)).  ase's ``get_special_points`` for
the primitive fcc cell returns primitive-fractional coordinates instead
(X=(1/2,0,1/2)); the example must convert them, ``k_conv = k_prim @ sc_mat``.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "Cu_fcc" / "unfold.py"


def _load_example_module():
    # unfold.py imports unfolding.DDB_unfolder, which needs abipy at import.
    pytest.importorskip("abipy")
    spec = importlib.util.spec_from_file_location("cu_fcc_unfold_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __main__ guard: no abinit run at import
    return mod


def test_ase_fcc_special_points_are_primitive_fractional():
    # Premise of the example's conversion: ase returns primitive-cell
    # fractional coordinates, not conventional-cubic ones.
    from ase.build import bulk
    from ase.dft.kpoints import get_special_points

    points = get_special_points(bulk("Cu", "fcc").cell, eps=0.01)
    assert np.allclose(points["X"], [0.5, 0.0, 0.5])
    assert np.allclose(points["W"], [0.5, 0.25, 0.75])
    assert np.allclose(points["L"], [0.5, 0.5, 0.5])


def test_example_kpath_is_in_conventional_frame():
    mod = _load_example_module()
    sc_mat = np.linalg.inv((np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0))
    bounds, names = mod.build_fcc_kpath(sc_mat)

    assert names == [r"$\Gamma$", "X", "W", r"$\Gamma$", "L"]
    # G-X-W-G-L in conventional-cubic reciprocal coordinates (the frame of
    # the conventional-cell DDB that abipy reads).
    expected = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0, 0.0],
         [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    assert np.allclose(bounds, expected, atol=1e-12)
