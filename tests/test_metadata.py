"""Packaging metadata tests: what pip actually sees after install."""
import re

import importlib.metadata as im

import pytest


def _dist():
    try:
        return im.distribution("unfolding")
    except im.PackageNotFoundError:  # pragma: no cover
        pytest.fail("unfolding is not installed; run `pip install -e .[dev]`")


def test_requires_python():
    assert _dist().metadata["Requires-Python"] == ">=3.9"


def test_license_is_bsd2():
    license_field = _dist().metadata["License"] or ""
    assert "BSD" in license_field
    assert "GPL" not in license_field


def test_core_dependencies():
    deps = _dist().metadata.get_all("Requires-Dist") or []
    core = [d for d in deps if "extra" not in d]
    names = {re.split(r"[<>=!\[]", d)[0] for d in core}
    assert {"numpy", "ase", "matplotlib"} <= names


def test_extras_declared():
    extras = set()
    for req in _dist().requires or []:
        m = re.search(r'extra == "([A-Za-z0-9_.-]+)"', req)
        if m:
            extras.add(m.group(1))
    assert {"phonopy", "abipy", "siesta", "dev"} <= extras


def test_siesta_extra_pins_hamiltonio():
    reqs = [r for r in (_dist().requires or []) if 'extra == "siesta"' in r]
    assert any(r.startswith("HamiltonIO") for r in reqs)
