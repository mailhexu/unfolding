"""Sparse published Si:P outputs must not masquerade as band paths."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_abinit_reproducer_skips_corner_only_si7p_map(tmp_path, monkeypatch, capsys):
    source = (Path(__file__).resolve().parents[1]
              / "examples" / "abinit-wfk-si" / "bundle" / "reproduce.py")
    spec = spec_from_file_location("abinit_wfk_reproduce", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    data = tmp_path / "data"
    data.mkdir()
    (data / "si7p_gxwglx_cornerso_DS2_WFK.nc").touch()
    output = tmp_path / "si7p.png"
    monkeypatch.setattr(module, "DATA", data)

    assert module.si7p_figure(output) is None
    assert not output.exists()
    assert "covers 7 of 305 path folds" in capsys.readouterr().out


def test_abinit_reproducer_checks_actual_si7p_path_coverage(tmp_path, monkeypatch, capsys):
    source = (Path(__file__).resolve().parents[1]
              / "examples" / "abinit-wfk-si" / "bundle" / "reproduce.py")
    spec = spec_from_file_location("abinit_wfk_reproduce_undercovered", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    data = tmp_path / "data"
    data.mkdir()
    (data / "si7p_gamma_x_patho_DS2_WFK.nc").touch()
    monkeypatch.setattr(module, "DATA", data)
    path_kpts = np.column_stack((np.arange(305) / 305, np.zeros((305, 2))))
    stored = np.tile(path_kpts[:7], (44, 1))[:305] @ module.MATRIX.T
    monkeypatch.setattr(
        module, "siesta_style_path",
        lambda: (path_kpts, np.arange(305), np.array([0, 1])),
    )
    monkeypatch.setattr(
        module, "match_path_subset",
        lambda _wfk: (SimpleNamespace(kpoints=stored), path_kpts[np.arange(305) % 7], np.arange(305)),
    )
    output = tmp_path / "undercovered.png"

    assert module.si7p_figure(output) is None
    assert not output.exists()
    assert "covers 7 of 305 requested path folds" in capsys.readouterr().out


def test_gpaw_pw_config_matches_generated_restart_grid(tmp_path):
    import os
    import shutil

    root = Path(__file__).resolve().parents[1]
    bundle = root / "examples" / "gpaw-si-pw" / "bundle"

    repro_spec = spec_from_file_location("gpaw_pw_reproduce", bundle / "reproduce.py")
    repro = module_from_spec(repro_spec)
    repro_spec.loader.exec_module(repro)
    fixture_spec = spec_from_file_location(
        "gpaw_fixture_generator", root / "examples" / "gpaw_si" / "generate_fixtures.py"
    )
    fixture = module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)

    from unfolding.config import load_config

    config_dir = tmp_path / "bundle"
    (config_dir / "data").mkdir(parents=True)
    shutil.copy2(bundle / "unfold.toml", config_dir / "unfold.toml")
    (config_dir / "data" / "si8_pw.gpw").touch()
    previous = Path.cwd()
    os.chdir(config_dir)
    try:
        config = load_config("unfold.toml")
    finally:
        os.chdir(previous)

    fixture_kpts, _, fixture_supercell_kpts = fixture.path_kpoints()
    repro_kpts, xcoords, xticks, names = repro.band_path()
    assert len(config.kpoints) == 305
    np.testing.assert_allclose(config.kpoints, fixture_kpts, atol=1e-14, rtol=0)
    np.testing.assert_allclose(np.asarray(config.kpoints) @ fixture.B.T, fixture_supercell_kpts, atol=1e-14, rtol=0)
    np.testing.assert_allclose(config.xcoords, xcoords, atol=1e-14, rtol=0)
    np.testing.assert_allclose(config.xticks, xticks, atol=1e-14, rtol=0)
    assert config.names == names
    assert config.output == "gpaw_si_pw_cli.png"
    assert config.output != "gpaw_si_pw_unfolded.png"
    np.testing.assert_allclose(repro_kpts, fixture_kpts, atol=1e-14, rtol=0)


def test_gpaw_pw_reproducer_fails_when_restart_is_missing(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = root / "examples" / "gpaw-si-pw" / "bundle" / "reproduce.py"
    spec = spec_from_file_location("gpaw_pw_missing_restart", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DATA", tmp_path / "no_data")
    monkeypatch.setattr(module, "HERE", tmp_path)

    with pytest.raises(FileNotFoundError, match="generate_restart.py"):
        module.pristine()
    assert not (tmp_path / "gpaw_si_pw_unfolded.png").exists()


def test_gpaw_restart_generator_uses_exact_fixture_kpoints(monkeypatch):
    import sys

    root = Path(__file__).resolve().parents[1]
    bundle = root / "examples" / "gpaw-si-pw" / "bundle"
    repro_spec = spec_from_file_location("gpaw_restart_repro", bundle / "reproduce.py")
    repro = module_from_spec(repro_spec)
    repro_spec.loader.exec_module(repro)
    monkeypatch.setitem(sys.modules, "reproduce", repro)
    generator_spec = spec_from_file_location(
        "gpaw_pw_restart_generator", bundle / "generate_restart.py"
    )
    generator = module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator)
    fixture_spec = spec_from_file_location(
        "gpaw_restart_fixture_generator", root / "examples" / "gpaw_si" / "generate_fixtures.py"
    )
    fixture = module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    _, _, expected = fixture.path_kpoints()

    np.testing.assert_allclose(generator.path_kpoints(), expected, atol=1e-14, rtol=0)
