"""CI wiring for the symbolic derivations (story 005).

Each test executes one ``derivations/*.py`` module end to end; every
module's ``run()`` asserts its identity internally, so a broken formula
fails these tests and CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_lcao_weight_dual_basis_identity():
    from derivations.lcao_weight import run

    res = run()
    assert res["pristine_weight_err"] < 1e-10
    assert res["symbolic_seal_err"] < 1e-10
    assert min(res["S_prim_eigmins"]) > 1e-8


def test_normalization_sum_rule():
    from derivations.normalization import run

    res = run()
    assert res["sector_sum_err"] < 1e-10


def test_phonon_projection_sum_rule():
    from derivations.phonon_projection import run

    res = run()
    assert res["sector_sum_err"] < 1e-10
