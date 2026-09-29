"""TOML configuration core for the ``unfolding`` CLI (story 033).

One dataclass per route describes every route parameter; field metadata
declares the TOML section, the flag kind, and the help text. The registry
``ROUTES`` maps route names to a :class:`RouteSpec` (dataclass + runner),
and both the TOML schema and the CLI flags are derived from it, so flags,
TOML, and the Python adapters cannot drift apart.

TOML schema (mirrors the Python parameters)::

    route = "siesta"          # selects the route dataclass
    [input]                   # route input files (fdf, wfk, ddb, ...)
    [structure]               # primitive cell, supercell matrix
    [path]                    # special points/path string or explicit points
    [options]                 # spin, tolerances, windows, shifts
    [output]                  # figure / dataset paths

Validation errors raise :class:`ConfigError` with the offending key path
(e.g. ``structure.supercell_matrix``).
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Callable, Union

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

__all__ = [
    "ConfigError",
    "FieldSpec",
    "RouteSpec",
    "ROUTES",
    "RouteConfig",
    "SiestaConfig",
    "PhonopyConfig",
    "AbinitWfkConfig",
    "AbinitDdbConfig",
    "MagnonConfig",
    "WfsxConfig",
    "AbinitPawConfig",
    "OpenmxConfig",
    "GpawConfig",
    "AbacusConfig",
    "VaspPawConfig",
    "WannierConfig",
    "load_config",
    "validate_config",
    "config_from_flags",
]


class ConfigError(Exception):
    """Invalid or incomplete route configuration (message carries the key path)."""


@dataclass(frozen=True)
class FieldSpec:
    """Declarative metadata for one route parameter.

    ``section`` is the TOML table the key lives in; ``kind`` selects the
    validation/coercion rule and the argparse shape (see ``cli.py``);
    ``flag`` overrides the default ``--dashed-name``; ``choices`` restricts
    string values.
    """

    section: str
    kind: str  # path|out|str|int|float|bool|matrix|float-list|str-list|kpoints
    help: str
    flag: str | None = None
    choices: tuple | None = None


def F(section, kind, help, flag=None, choices=None):
    """``dataclasses.field`` metadata for a route parameter."""
    return {"spec": FieldSpec(section, kind, help, flag, choices)}


# ---------------------------------------------------------------------------
# Route dataclasses (one per route; field order = documentation order)
# ---------------------------------------------------------------------------

@dataclass
class SiestaConfig:
    # [input]
    fdf: str = dataclasses.field(metadata=F(
        "input", "path", "SIESTA fdf of the supercell run (Hamiltonian saved)"))
    # [structure]
    primitive: str = dataclasses.field(metadata=F(
        "structure", "path", "primitive cell (ASE-readable)"))
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "supercell matrix M with A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    # [path]
    special_points: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "str", "special-point letters on the path cell, e.g. GXWGLX"))
    kpoints: list | None = dataclasses.field(
        default=None, metadata=F(
            "path", "kpoints", "explicit path points, three fractional numbers each"))
    npts: int = dataclasses.field(
        default=200, metadata=F("path", "int", "points per path (special_points mode)"))
    path_cell: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "path", "path-frame cell (ASE-readable); default: the primitive cell",
            flag="path-cell"))
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for path vertices"))
    xcoords: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "path x coordinates"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the vertex ticks"))
    # [options]
    spin: str = dataclasses.field(
        default="up", metadata=F(
            "options", "str", "collinear channel", choices=("up", "down")))
    method: str = dataclasses.field(
        default="ring", metadata=F(
            "options", "str",
            "weight definition: ring (torus grid) or ideal (arbitrary k-paths)",
            choices=("ring", "ideal")))
    tol_r: float = dataclasses.field(
        default=0.04, metadata=F(
            "options", "float", "atom-matching tolerance (Angstrom)"))
    match_species: bool = dataclasses.field(
        default=True, metadata=F(
            "options", "bool",
            "match species in the relabel map; false for substitutional dopants"))
    efermi: float = dataclasses.field(
        default=0.0, metadata=F("options", "float", "Fermi level (eV)"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class WfsxConfig:
    # [input]
    wfsx: str = dataclasses.field(metadata=F(
        "input", "path", "SIESTA WFSX wavefunctions of the supercell path run"))
    hs_fdf: str = dataclasses.field(metadata=F(
        "input", "path", "SIESTA fdf of the supercell run (Hamiltonian saved)",
        flag="hs-fdf"))
    # [structure]
    primitive: str = dataclasses.field(metadata=F(
        "structure", "path", "primitive cell fdf (ASE/SIESTA-readable)"))
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "supercell matrix M with A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    # [path]
    special_points: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "str", "special-point letters on the primitive cell, e.g. GXWGLX"))
    kpoints: list | None = dataclasses.field(
        default=None, metadata=F(
            "path", "kpoints", "explicit path points, three fractional numbers each"))
    npts: int = dataclasses.field(
        default=200, metadata=F("path", "int", "points per path (special_points mode)"))
    path_cell: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "path", "path-frame cell (ASE-readable); default: the primitive cell",
            flag="path-cell"))
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for path vertices"))
    xcoords: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "path x coordinates"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the vertex ticks"))
    # [options]
    method: str = dataclasses.field(
        default="ideal", metadata=F(
            "options", "str",
            "weight definition: ring (torus grid) or ideal (arbitrary k-paths)",
            choices=("ring", "ideal")))
    tol_r: float = dataclasses.field(
        default=0.04, metadata=F(
            "options", "float", "atom-matching tolerance (Angstrom)"))
    efermi: float | None = dataclasses.field(
        default=None, metadata=F(
            "options", "float",
            "Fermi level (eV); default: the <wfsx stem>.EIG header of the writing run"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class PhonopyConfig:
    # [input]
    force_constants: str = dataclasses.field(metadata=F(
        "input", "path", "phonopy FORCE_CONSTANTS file"))
    sposcar: str = dataclasses.field(metadata=F(
        "input", "path", "phonopy SPOSCAR file"))
    # [structure]
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "unfolding matrix, A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    sc_mat: list = dataclasses.field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        metadata=F("structure", "matrix",
                   "phonopy reading matrix, SPOSCAR cell = sc_mat @ its primitive"))
    # [path]
    special_points: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "str", "special-point letters on the derived primitive cell"))
    kpoints: list | None = dataclasses.field(
        default=None, metadata=F(
            "path", "kpoints", "explicit path points, three fractional numbers each"))
    npts: int = dataclasses.field(
        default=200, metadata=F("path", "int", "points per path (special_points mode)"))
    path_cell: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "path",
            "path-frame cell (ASE-readable); default: inv(supercell_matrix) @ SPOSCAR cell",
            flag="path-cell"))
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for path vertices"))
    xcoords: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "path x coordinates"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the vertex ticks"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class AbinitWfkConfig:
    # [input]
    wfk: str = dataclasses.field(metadata=F(
        "input", "path", "ABINIT netCDF WFK file (iomode 3, istwfk 1)"))
    # [structure]
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "supercell matrix M with A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    primitive: str | None = dataclasses.field(
        default=None, metadata=F(
            "structure", "path",
            "primitive cell (ASE-readable); needed for special_points mode"))
    # [path]
    special_points: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "str", "special-point letters on the primitive cell"))
    kpoints: list | None = dataclasses.field(
        default=None, metadata=F(
            "path", "kpoints", "explicit path points, three primitive-frame numbers each"))
    npts: int = dataclasses.field(
        default=200, metadata=F("path", "int", "points per path (special_points mode)"))
    path_cell: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "path", "path-frame cell (ASE-readable); default: structure.primitive",
            flag="path-cell"))
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for path vertices"))
    xcoords: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "path x coordinates"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the vertex ticks"))
    # [options]
    spin: int = dataclasses.field(
        default=0, metadata=F("options", "int", "spin channel index"))
    resolve_degenerate: float | None = dataclasses.field(
        default=None, metadata=F(
            "options", "float", "degeneracy tolerance (eV) for eigen-assigned weights",
            flag="resolve-degenerate"))
    average_degenerate: float | None = dataclasses.field(
        default=None, metadata=F(
            "options", "float", "degeneracy tolerance (eV) for group-averaged weights",
            flag="average-degenerate"))
    fermi_shift: bool = dataclasses.field(
        default=True, metadata=F(
            "options", "bool", "shift energies by the WFK Fermi level",
            flag="fermi-shift"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class AbinitDdbConfig:
    # [input]
    ddb: str = dataclasses.field(metadata=F(
        "input", "path", "ABINIT DDB (derivatives database) file"))
    # [structure]
    sc_mat: list = dataclasses.field(metadata=F(
        "structure", "matrix",
        "supercell matrix in DDB-cell units, S = A_ddb @ sc_mat (row convention)",
        flag="sc-mat"))
    # [path] (vertices: anaddb interpolates between them; no npts)
    special_points: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "str", "special-point letters on the path cell "
            "(default: the DDB cell)"))
    kpoints: list | None = dataclasses.field(
        default=None, metadata=F(
            "path", "kpoints", "path vertices in the DDB-cell fractional frame"))
    path_cell: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "path", "path-frame cell (ASE-readable); default: the DDB cell",
            flag="path-cell"))
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for path vertices"))
    # [options]
    dipdip: int = dataclasses.field(
        default=1, metadata=F("options", "int", "dipole-dipole treatment (LO-TO)"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class MagnonConfig:
    # [input]
    results: str = dataclasses.field(metadata=F(
        "input", "path", "TB2J results directory containing TB2J.pickle"))
    # [structure]
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "unfold matrix, A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    # [path]
    special_points: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "str", "special-point letters on the derived primitive cell"))
    kpoints: list | None = dataclasses.field(
        default=None, metadata=F(
            "path", "kpoints", "explicit q-path points, three fractional numbers each"))
    npts: int = dataclasses.field(
        default=200, metadata=F("path", "int", "points per path (special_points mode)"))
    path_cell: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "path",
            "path-frame cell (ASE-readable); default: the derived primitive cell",
            flag="path-cell"))
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for path vertices"))
    xcoords: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "path x coordinates"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the vertex ticks"))
    # [options]
    spin_conf: list | None = dataclasses.field(
        default=None, metadata=F(
            "options", "float-list",
            "collinear moments as nspin x 3 floats (default: from the pickle)",
            flag="spin-conf"))
    degen_tolerance: float = dataclasses.field(
        default=1e-5, metadata=F(
            "options", "float", "degenerate-group energy tolerance (eV)",
            flag="degen-tolerance"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class AbinitPawConfig:
    # [input]
    supercell: str = dataclasses.field(metadata=F(
        "input", "path", "supercell PAW WFK (netCDF, usepaw=1)"))
    primitive: str = dataclasses.field(metadata=F(
        "input", "path", "primitive reference PAW WFK (its stored k-points "
        "carry the path)"))
    paw: str = dataclasses.field(metadata=F(
        "input", "path", "directory with per-species JTH XML files "
        "(<Symbol>.xml, e.g. Si.xml)"))
    # [structure]
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "supercell matrix M with A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    # [path] (weights are computed at the stored primitive WFK k-points;
    # the path itself is carried by the WFK, so only tick labels are settable)
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for the path ticks"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the tick labels"))
    # [options]
    spin: int = dataclasses.field(
        default=0, metadata=F("options", "int", "spin channel index"))
    resolve_degenerate: float | None = dataclasses.field(
        default=None, metadata=F(
            "options", "float", "degeneracy tolerance (eV) for gauge-invariant weights",
            flag="resolve-degenerate"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class OpenmxConfig:
    # [input]
    scfout: str = dataclasses.field(metadata=F(
        "input", "path", "OpenMX .scfout file of the supercell run"))
    # [structure]
    primitive: str = dataclasses.field(metadata=F(
        "structure", "path", "primitive cell (ASE-readable)"))
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "supercell matrix M with A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    # [path]
    special_points: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "str", "special-point letters on the primitive cell, e.g. GXWGLWX"))
    kpoints: list | None = dataclasses.field(
        default=None, metadata=F(
            "path", "kpoints", "explicit path points, three fractional numbers each"))
    npts: int = dataclasses.field(
        default=200, metadata=F("path", "int", "points per path (special_points mode)"))
    path_cell: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "path", "path-frame cell (ASE-readable); default: the primitive cell",
            flag="path-cell"))
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for path vertices"))
    xcoords: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "path x coordinates"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the vertex ticks"))
    # [options]
    spin: int | None = dataclasses.field(
        default=None, metadata=F(
            "options", "int", "collinear channel (0/1); omit for unpolarized runs"))
    match_species: bool = dataclasses.field(
        default=True, metadata=F(
            "options", "bool",
            "match species in the relabel map; false for substitutional dopants"))
    method: str = dataclasses.field(
        default="ideal", metadata=F(
            "options", "str",
            "weight definition: ring (torus grid) or ideal (arbitrary k-paths)",
            choices=("ring", "ideal")))
    tol_r: float = dataclasses.field(
        default=0.04, metadata=F(
            "options", "float", "atom-matching tolerance (Angstrom)"))
    efermi: float = dataclasses.field(
        default=0.0, metadata=F(
            "options", "float", "Fermi level (eV); 0 uses the scfout ChemP"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class GpawConfig:
    # [input]
    supercell: str = dataclasses.field(metadata=F(
        "input", "path", "supercell GPAW .gpw restart (mode='all')"))
    # [structure]
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "supercell matrix M with A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    primitive: str | None = dataclasses.field(
        default=None, metadata=F(
            "structure", "path", "primitive GPAW .gpw restart (LCAO mode)"))
    # [path]
    special_points: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "str", "special-point letters on the primitive cell, e.g. GXWGLWX"))
    kpoints: list | None = dataclasses.field(
        default=None, metadata=F(
            "path", "kpoints", "explicit path points, three primitive-frame numbers each"))
    npts: int = dataclasses.field(
        default=200, metadata=F("path", "int", "points per path (special_points mode)"))
    path_cell: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "path",
            "path-frame cell (ASE-readable); default: derived from the inputs",
            flag="path-cell"))
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for path vertices"))
    xcoords: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "path x coordinates"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the vertex ticks"))
    # [options]
    mode: str = dataclasses.field(
        default="lcao", metadata=F(
            "options", "str", "LCAO (real-space tables) or PW (planewave) unfolding",
            choices=("lcao", "pw")))
    spin: int = dataclasses.field(
        default=0, metadata=F("options", "int", "spin channel (LCAO reader)"))
    match_species: bool = dataclasses.field(
        default=True, metadata=F(
            "options", "bool",
            "match species in the relabel map; false for substitutional dopants"))
    method: str = dataclasses.field(
        default="ideal", metadata=F(
            "options", "str",
            "weight definition: ring (torus grid) or ideal (arbitrary k-paths)",
            choices=("ring", "ideal")))
    tol_r: float = dataclasses.field(
        default=0.04, metadata=F(
            "options", "float", "atom-matching tolerance (Angstrom)"))
    resolve_degenerate: float | None = dataclasses.field(
        default=None, metadata=F(
            "options", "float", "PW mode: degeneracy tolerance (eV)",
            flag="resolve-degenerate"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class AbacusConfig:
    # [input]
    supercell: str = dataclasses.field(metadata=F(
        "input", "path", "supercell ABACUS OUT.* output directory"))
    # [structure]
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "supercell matrix M with A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    primitive: str | None = dataclasses.field(
        default=None, metadata=F(
            "structure", "path", "primitive OUT.* output directory (LCAO mode)"))
    # [path]
    special_points: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "str", "special-point letters on the primitive cell, e.g. GXWGLWX"))
    kpoints: list | None = dataclasses.field(
        default=None, metadata=F(
            "path", "kpoints", "explicit path points, three primitive-frame numbers each"))
    npts: int = dataclasses.field(
        default=200, metadata=F("path", "int", "points per path (special_points mode)"))
    path_cell: str | None = dataclasses.field(
        default=None, metadata=F(
            "path", "path",
            "path-frame cell (ASE-readable); default: derived from the inputs",
            flag="path-cell"))
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for path vertices"))
    xcoords: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "path x coordinates"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the vertex ticks"))
    # [options]
    mode: str = dataclasses.field(
        default="lcao", metadata=F(
            "options", "str", "LCAO (real-space tables) or PW (WAVEFUNC) unfolding",
            choices=("lcao", "pw")))
    spin: str = dataclasses.field(
        default="up", metadata=F(
            "options", "str", "collinear channel (LCAO mode)",
            choices=("up", "down")))
    match_species: bool = dataclasses.field(
        default=True, metadata=F(
            "options", "bool",
            "match species in the relabel map; false for substitutional dopants"))
    method: str = dataclasses.field(
        default="ideal", metadata=F(
            "options", "str",
            "weight definition: ring (torus grid) or ideal (arbitrary k-paths)",
            choices=("ring", "ideal")))
    tol_r: float = dataclasses.field(
        default=0.04, metadata=F(
            "options", "float", "atom-matching tolerance (Angstrom)"))
    resolve_degenerate: float | None = dataclasses.field(
        default=None, metadata=F(
            "options", "float", "PW mode: degeneracy tolerance (eV)",
            flag="resolve-degenerate"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class VaspPawConfig:
    # [input]
    supercell: str = dataclasses.field(metadata=F(
        "input", "path", "supercell WAVECAR"))
    supercell_poscar: str = dataclasses.field(metadata=F(
        "input", "path", "POSCAR of the supercell WAVECAR",
        flag="supercell-poscar"))
    primitive: str = dataclasses.field(metadata=F(
        "input", "path", "primitive reference WAVECAR (its stored k-points "
        "carry the path)"))
    primitive_poscar: str = dataclasses.field(metadata=F(
        "input", "path", "POSCAR of the primitive WAVECAR",
        flag="primitive-poscar"))
    potcar: str = dataclasses.field(metadata=F(
        "input", "path", "matching licensed POTCAR (never redistributed)"))
    # [structure]
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "supercell matrix M with A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    # [path] (weights are computed at the stored primitive WAVECAR k-points;
    # the path itself is carried by the WAVECAR, so only tick labels are settable)
    names: list | None = dataclasses.field(
        default=None, metadata=F("path", "str-list", "labels for the path ticks"))
    xticks: list | None = dataclasses.field(
        default=None, metadata=F("path", "float-list", "x positions of the tick labels"))
    # [options]
    spin: int = dataclasses.field(
        default=0, metadata=F("options", "int", "spin channel index"))
    resolve_degenerate: float | None = dataclasses.field(
        default=None, metadata=F(
            "options", "float", "degeneracy tolerance (eV) for gauge-invariant weights",
            flag="resolve-degenerate"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))


@dataclass
class WannierConfig:
    # [input]
    path: str = dataclasses.field(metadata=F(
        "input", "path", "Wannier90 directory (<prefix>.win, <prefix>_hr.dat)"))
    # [structure]
    supercell_matrix: list = dataclasses.field(metadata=F(
        "structure", "matrix", "supercell matrix M with A_sc = M @ A_prim (row convention)",
        flag="unfold-mat"))
    labels: list = dataclasses.field(metadata=F(
        "structure", "str-list",
        "basis labels, one per Wannier orbital (used for the weight sectors)"))
    # [path]
    kpoints: list = dataclasses.field(metadata=F(
        "path", "kpoints", "path vertices, three primitive-frame numbers each"))
    names: list = dataclasses.field(metadata=F(
        "path", "str-list", "labels for path vertices"))
    # [output]
    output: str | None = dataclasses.field(
        default=None, metadata=F("output", "out", "output figure path"))
    data: str | None = dataclasses.field(
        default=None, metadata=F(
            "output", "out", "output unfolded-band dataset path (JSON)"))
    # [options]
    prefix: str = dataclasses.field(
        default="wannier90", metadata=F("input", "str", "Wannier90 seed name"))
    npoints: int = dataclasses.field(
        default=200, metadata=F("options", "int", "points per path"))
    # [structure]
    cell: list | None = dataclasses.field(
        default=None, metadata=F(
            "structure", "matrix",
            "unit cell of the Wannierisation (3x3); default: read from <prefix>.win"))


RouteConfig = Union[
    SiestaConfig, PhonopyConfig, WfsxConfig, AbinitWfkConfig, AbinitDdbConfig,
    MagnonConfig, AbinitPawConfig, OpenmxConfig, GpawConfig, AbacusConfig,
    VaspPawConfig, WannierConfig,
]


@dataclass(frozen=True)
class RouteSpec:
    """Registry entry: route name -> config dataclass + runner.

    ``path_cell_field`` names the config field holding the default
    special-points cell file (``None``: the runner derives the cell, e.g.
    from the DDB or the supercell matrix).
    """

    name: str
    config: type
    runner: Callable
    path_cell_field: str | None = None

    def fields(self):
        """(field name, FieldSpec) pairs in declaration order."""
        return [(f.name, f.metadata["spec"]) for f in dataclasses.fields(self.config)]


# Imported after the dataclass definitions above; routes.py imports
# ConfigError from this (partially initialized) module, which is defined
# before this point.
from .routes import (  # noqa: E402
    run_abacus,
    run_abinit_ddb,
    run_abinit_paw,
    run_abinit_wfk,
    run_gpaw,
    run_magnon,
    run_openmx,
    run_phonopy,
    run_siesta,
    run_siesta_wfsx,
    run_vasp_paw,
    run_wannier,
)

ROUTES: dict[str, RouteSpec] = {
    "siesta": RouteSpec(
        name="siesta", config=SiestaConfig, runner=run_siesta,
        path_cell_field="primitive"),
    "siesta-wfsx": RouteSpec(
        name="siesta-wfsx", config=WfsxConfig, runner=run_siesta_wfsx),
    "phonopy": RouteSpec(
        name="phonopy", config=PhonopyConfig, runner=run_phonopy),
    "abinit-wfk": RouteSpec(
        name="abinit-wfk", config=AbinitWfkConfig, runner=run_abinit_wfk,
        path_cell_field="primitive"),
    "abinit-ddb": RouteSpec(
        name="abinit-ddb", config=AbinitDdbConfig, runner=run_abinit_ddb),
    "magnon": RouteSpec(
        name="magnon", config=MagnonConfig, runner=run_magnon),
    "abinit-paw": RouteSpec(
        name="abinit-paw", config=AbinitPawConfig, runner=run_abinit_paw),
    "openmx": RouteSpec(
        name="openmx", config=OpenmxConfig, runner=run_openmx,
        path_cell_field="primitive"),
    "gpaw": RouteSpec(
        name="gpaw", config=GpawConfig, runner=run_gpaw),
    "abacus": RouteSpec(
        name="abacus", config=AbacusConfig, runner=run_abacus),
    "vasp-paw": RouteSpec(
        name="vasp-paw", config=VaspPawConfig, runner=run_vasp_paw),
    "wannier": RouteSpec(
        name="wannier", config=WannierConfig, runner=run_wannier),
}

_SECTIONS = ("input", "structure", "path", "options", "output")


# ---------------------------------------------------------------------------
# Validation and coercion
# ---------------------------------------------------------------------------

def _is_required(f) -> bool:
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def _fail(key_path, message):
    raise ConfigError(f"{key_path}: {message}")


def _coerce_value(kind, key_path, value):
    """Type-check/normalize one TOML value; returns the Python value."""
    if kind in ("path", "out", "str"):
        if not isinstance(value, str):
            _fail(key_path, f"expected a string, got {type(value).__name__}")
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(key_path, f"expected an integer, got {value!r}")
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(key_path, f"expected a number, got {value!r}")
        return float(value)
    if kind == "bool":
        if not isinstance(value, bool):
            _fail(key_path, f"expected a boolean, got {value!r}")
        return value
    if kind == "matrix":
        return _coerce_matrix(key_path, value)
    if kind == "kpoints":
        return _coerce_kpoints(key_path, value)
    if kind == "float-list":
        if not isinstance(value, list) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            _fail(key_path, f"expected a list of numbers, got {value!r}")
        return [float(v) for v in value]
    if kind == "str-list":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            _fail(key_path, f"expected a list of strings, got {value!r}")
        return list(value)
    raise ConfigError(f"{key_path}: unknown field kind {kind!r}")  # pragma: no cover


def _coerce_matrix(key_path, value):
    ok = (
        isinstance(value, list) and len(value) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in value)
    )
    if not ok:
        _fail(key_path, "expected a 3x3 matrix of numbers")
    out = []
    for row in value:
        ints = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                _fail(key_path, f"expected a number, got {v!r}")
            if float(v) != int(v):
                _fail(key_path, f"matrix entries must be integers, got {v!r}")
            ints.append(int(v))
        out.append(ints)
    return out


def _coerce_kpoints(key_path, value):
    if not isinstance(value, list) or len(value) < 2:
        _fail(key_path, "expected at least two points")
    out = []
    for k in value:
        if not isinstance(k, list) or len(k) != 3 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in k):
            _fail(key_path, f"each point needs three numbers, got {k!r}")
        out.append([float(v) for v in k])
    return out


def validate_config(cfg) -> None:
    """Validate a route config object; raises ConfigError with key paths."""
    spec = next(
        (s for s in ROUTES.values() if isinstance(cfg, s.config)), None)
    if spec is None:
        raise ConfigError(
            f"not a route config: {type(cfg).__name__}; expected one of "
            f"{[s.config.__name__ for s in ROUTES.values()]}")

    for f in dataclasses.fields(spec.config):
        fs = f.metadata["spec"]
        value = getattr(cfg, f.name)
        if value is None:
            if _is_required(f):
                _fail(f"{fs.section}.{f.name}", "is required")
            continue
        _check_value(fs, f.name, value)

    _validate_path(spec, cfg)


def _check_value(fs, name, value):
    key = f"{fs.section}.{name}"
    _coerce_value(fs.kind, key, value)
    if fs.kind == "path" and not os.path.exists(value):
        _fail(key, f"file not found: {value}")
    if fs.choices and value not in fs.choices:
        _fail(key, f"must be one of {list(fs.choices)}, got {value!r}")


def _validate_path(spec, cfg):
    field_names = {name for name, _ in spec.fields()}
    if "special_points" not in field_names and "kpoints" not in field_names:
        # Stored-k routes (abinit-paw, vasp-paw): the path is carried by the
        # reference WFK/WAVECAR, so [path] holds optional tick labels only.
        names = getattr(cfg, "names", None)
        xticks = getattr(cfg, "xticks", None)
        if (names is None) != (xticks is None):
            raise ConfigError(
                "path: path.names and path.xticks must be given together")
        if names is not None and xticks is not None \
                and len(names) != len(xticks):
            _fail("path.names", "needs one label per path.xticks entry")
        return
    special = getattr(cfg, "special_points", None)
    kpts = getattr(cfg, "kpoints", None)
    if special and kpts:
        raise ConfigError(
            "path: give either path.special_points or path.kpoints, not both")
    if not special and not kpts:
        raise ConfigError(
            "path: one of path.special_points or path.kpoints is required")
    if special:
        if not isinstance(special, str) or not special:
            _fail("path.special_points", "expected a non-empty letters string")
        if spec.path_cell_field and getattr(cfg, spec.path_cell_field) is None:
            raise ConfigError(
                f"path: path.special_points requires "
                f"{spec.path_cell_field} (the path cell)")
    else:
        names = getattr(cfg, "names", None)
        xticks = getattr(cfg, "xticks", None)
        has_xticks_field = any(n == "xticks" for n, _ in spec.fields())
        if has_xticks_field and (names is None) != (xticks is None):
            raise ConfigError(
                "path: path.names and path.xticks must be given together")
        if names is not None and xticks is not None \
                and len(names) != len(xticks):
            _fail("path.names", "needs one label per path.xticks entry")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path) -> RouteConfig:
    """Load and validate a route TOML config; returns the route config object."""
    file = os.fspath(path)
    try:
        with open(file, "rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {file}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{file}: invalid TOML: {exc}") from None
    except IsADirectoryError:
        raise ConfigError(f"config file not found: {file}") from None

    route = raw.get("route")
    if route is None:
        raise ConfigError("missing required key 'route'")
    if not isinstance(route, str) or route not in ROUTES:
        raise ConfigError(
            f"route: unknown route {route!r}; available: {sorted(ROUTES)}")
    spec = ROUTES[route]

    # bucket the remaining top-level keys into TOML sections
    sections: dict[str, dict] = {}
    for key, value in raw.items():
        if key == "route":
            continue
        if not isinstance(value, dict):
            raise ConfigError(
                f"unknown key {key!r} (expected 'route' or a section: "
                f"{', '.join(_SECTIONS)})")
        if key not in _SECTIONS:
            raise ConfigError(
                f"unknown section [{key}] (expected: {', '.join(_SECTIONS)})")
        sections[key] = value

    kwargs = {}
    known = {f.name for f in dataclasses.fields(spec.config)}
    for section, table in sections.items():
        unknown = set(table) - known
        if unknown:
            raise ConfigError(
                f"unknown key in [{section}]: {', '.join(sorted(unknown))}")
    for f in dataclasses.fields(spec.config):
        fs = f.metadata["spec"]
        table = sections.get(fs.section, {})
        if f.name in table:
            kwargs[f.name] = _coerce_value(
                fs.kind, f"{fs.section}.{f.name}", table[f.name])
        elif _is_required(f):
            _fail(f"{fs.section}.{f.name}", "is required")

    cfg = spec.config(**kwargs)
    validate_config(cfg)
    return cfg


def config_from_flags(route: str, args) -> RouteConfig:
    """Build a validated route config from parsed argparse flags."""
    spec = ROUTES[route]
    kwargs = {}
    for f in dataclasses.fields(spec.config):
        value = getattr(args, f.name)
        if f.metadata["spec"].kind == "matrix" and value is not None:
            if len(value) != 9:
                _fail(f"{f.metadata['spec'].section}.{f.name}",
                      "needs 9 values (3x3 matrix)")
            value = [[int(value[r * 3 + c]) for c in range(3)] for r in range(3)]
        if f.metadata["spec"].kind == "kpoints" and value is not None:
            if len(value) % 3:
                _fail("path.kpoints", "needs a multiple of 3 numbers")
            value = [[float(value[i]), float(value[i + 1]), float(value[i + 2])]
                     for i in range(0, len(value), 3)]
        kwargs[f.name] = value
    cfg = spec.config(**kwargs)
    validate_config(cfg)
    return cfg
