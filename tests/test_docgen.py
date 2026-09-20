"""Story-011: docgen figure scripts run green and produce files."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script, out_name):
    out = os.path.join(ROOT, "docgen", "_check_" + out_name)
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "docgen", script), out],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert os.path.exists(out) and os.path.getsize(out) > 5000, out
    os.remove(out)


def test_fig_phonopy_cu():
    pytest.importorskip("phonopy")
    _run("fig_phonopy_cu.py", "phonopy.png")


def test_fig_siesta_si():
    pytest_importorskip_hamiltonio()
    _run("fig_siesta_si.py", "si.png")


def test_fig_siesta_p_doped():
    pytest_importorskip_hamiltonio()
    _run("fig_siesta_p_doped.py", "si_p_doped.png")


def test_fig_siesta_wfsx():
    pytest_importorskip_hamiltonio()
    _run("fig_siesta_wfsx.py", "si_wfsx.png")


def test_fig_siesta_spinor():
    pytest_importorskip_hamiltonio()
    _run("fig_siesta_spinor.py", "si_spinor.png")


def pytest_importorskip_hamiltonio():
    try:
        import HamiltonIO  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("HamiltonIO not installed")
    return True
