"""Import-guard behavior: missing optional backend -> actionable ImportError.

Uses subprocesses with the backend module poisoned (sys.modules[name] = None),
which reproduces 'module not installed' import failures deterministically.
"""
import importlib.util
import subprocess
import sys

import pytest

POISON_PRELUDE = "import sys; sys.modules[{name!r}] = None\n"


def _run_poisoned(code: str, poisoned_module: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", POISON_PRELUDE.format(name=poisoned_module) + code],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_core_import_works_without_phonopy():
    r = _run_poisoned(
        "import unfolding; print('OK', unfolding.Unfolder.__name__, "
        "unfolding.phonon_unfolder.__name__)",
        "phonopy",
    )
    assert r.returncode == 0, r.stderr
    assert "OK Unfolder phonon_unfolder" in r.stdout


def test_core_import_works_without_abipy():
    r = _run_poisoned("import unfolding; print('OK')", "abipy")
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_phonopy_unfold_lazy_access_names_extra():
    r = _run_poisoned(
        "import unfolding\n"
        "try:\n"
        "    unfolding.phonopy_unfold\n"
        "except ImportError as e:\n"
        "    print('CAUGHT:', e)\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')",
        "phonopy",
    )
    assert "unfolding[phonopy]" in r.stdout


def test_wfk_reader_names_abinit_extra_when_netcdf_missing():
    r = _run_poisoned(
        "import unfolding\n"
        "try:\n"
        "    unfolding.read_wfk(unfolding.__file__)\n"
        "except ImportError as e:\n"
        "    print('CAUGHT:', e)\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')",
        "netCDF4",
    )
    assert r.returncode == 0, r.stderr
    assert "unfolding[abinit]" in r.stdout


def test_netcdf_transitive_missing_dependency_is_not_masked():
    r = _run_poisoned(
        "try:\n"
        "    from unfolding.abinit_unfold import _require_netcdf4\n"
        "    _require_netcdf4()\n"
        "except ModuleNotFoundError as e:\n"
        "    print('REAL:', e.name)\n"
        "else:\n"
        "    raise SystemExit('expected ModuleNotFoundError')",
        "cftime",
    )
    assert r.returncode == 0, r.stderr
    assert "REAL: cftime" in r.stdout


def test_phonopy_module_direct_import_names_extra():
    r = _run_poisoned("import unfolding.phonopy_unfolder", "phonopy")
    assert r.returncode != 0
    assert "unfolding[phonopy]" in r.stderr


def test_ddb_module_direct_import_names_extra():
    r = _run_poisoned("import unfolding.DDB_unfolder", "abipy")
    assert r.returncode != 0
    assert "unfolding[abipy]" in r.stderr


def test_ddb_symbol_from_import_names_extra():
    r = _run_poisoned(
        "try:\n"
        "    from unfolding import DDB_unfolder\n"
        "except ImportError as e:\n"
        "    print('CAUGHT:', e)\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')",
        "abipy",
    )
    assert "unfolding[abipy]" in r.stdout


def test_nc_unfolder_from_import_names_extra():
    r = _run_poisoned(
        "try:\n"
        "    from unfolding.DDB_unfolder import nc_unfolder\n"
        "except ImportError as e:\n"
        "    print('CAUGHT:', e)\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')",
        "abipy",
    )
    assert "unfolding[abipy]" in r.stdout


def test_unrelated_missing_dependency_is_not_masked():
    # With phonopy importable, a missing non-phonopy dependency further down
    # the import chain must surface as-is (not be rewritten by the guard).
    if importlib.util.find_spec("phonopy") is None:
        pytest.skip("phonopy not installed; needs the real backend importable")
    r = _run_poisoned(
        "try:\n"
        "    import unfolding.phonopy_unfolder\n"
        "except ModuleNotFoundError as e:\n"
        "    print('REAL:', e.name)\n"
        "else:\n"
        "    raise SystemExit('expected ModuleNotFoundError')",
        "matplotlib",
    )
    assert "REAL: matplotlib" in r.stdout


def test_lazy_export_callable_when_backend_installed():
    pytest.importorskip("phonopy")
    import unfolding

    assert callable(unfolding.phonopy_unfold)
