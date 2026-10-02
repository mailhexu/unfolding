"""Route runners: build adapter calls from validated config objects.

Thin wrappers only -- the adapters keep their explicit-parameter
signatures and all physics. Each runner is reachable as
``unfolding.run(cfg)`` and via the ``unfolding`` CLI; heavy/optional
backends are imported inside the runners so that ``import unfolding``
stays cheap and extra-free.
"""
from __future__ import annotations

import os

import numpy as np

from .config import ConfigError

__all__ = [
    "run",
    "run_siesta",
    "run_phonopy",
    "run_abinit_wfk",
    "run_abinit_ddb",
    "run_magnon",
    "run_abinit_paw",
    "run_openmx",
    "run_gpaw",
    "run_abacus",
    "run_vasp_paw",
    "run_wannier",
]


def run(cfg):
    """Execute a route config object (see :func:`unfolding.config.load_config`)."""
    # deferred: config.py imports this module while ROUTES is being defined
    from .config import ROUTES

    for spec in ROUTES.values():
        if isinstance(cfg, spec.config):
            return spec.runner(cfg)
    raise ConfigError(
        f"not a route config: {type(cfg).__name__}; expected one of "
        f"{[s.config.__name__ for s in ROUTES.values()]}")


def _save(ax, output):
    if output is not None:
        ax.figure.savefig(output)
    return ax

def _route_name(cfg):
    from .config import ROUTES

    for name, spec in ROUTES.items():
        if isinstance(cfg, spec.config):
            return name
    return type(cfg).__name__


def _provenance(cfg):
    """Small config subset recorded in the dataset (never path arrays)."""
    keys = ("supercell_matrix", "unfold_sc_mat", "sc_mat", "spin",
            "method", "mode", "resolve_degenerate", "spin_conf",
            "special_points", "match_species")
    out = {}
    for key in keys:
        value = getattr(cfg, key, None)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (list, tuple)):
            value = [v.tolist() if isinstance(v, np.ndarray) else v
                     for v in value]
        out[key] = value
    return out


def _save_data(cfg, result, *, energy_unit="eV", fermi_energy=None,
               energy_reference=None):
    """Write the JSON dataset when ``[output] data`` is configured."""
    if getattr(cfg, "data", None) is None:
        return
    from .dataset import save_dataset

    save_dataset(
        result, cfg.data, route=_route_name(cfg),
        provenance=_provenance(cfg), fermi_energy=fermi_energy,
        energy_reference=energy_reference, energy_unit=energy_unit)



def _path_cell_array(cfg, default=None):
    """(3, 3) array of the special-points frame cell."""
    from ase.io import read as ase_read

    cell = getattr(cfg, "path_cell", None) or default
    if cell is None:
        raise ConfigError(
            "path: special_points mode needs a path cell "
            "(path.path_cell or the route's reference structure)")
    if isinstance(cell, (str, os.PathLike)):
        return np.asarray(ase_read(cell).cell.array, dtype=float)
    return np.asarray(cell, dtype=float)


def _resolve_path(cfg, default_cell=None, dense=True):
    """(kpts, knames, xqpts, Xqpts) from the [path] fields of a config.

    ``dense=True`` interpolates between special points (ase bandpath);
    ``dense=False`` returns the bare special points as path vertices for
    adapters that interpolate themselves (abinit-ddb through anaddb).
    """
    from ase.dft.kpoints import bandpath, get_special_points

    if getattr(cfg, 'special_points', None):
        if isinstance(cfg.special_points, str) and ',' in cfg.special_points:
            raise ConfigError(
                'path.special_points: disconnected paths (commas) are not '
                'supported; use path.kpoints and path.names instead')
        cell = _path_cell_array(cfg, default_cell)
        points = get_special_points(cell, eps=0.01)
        unknown = [c for c in cfg.special_points if c not in points]
        if unknown:
            raise ConfigError(
                f'path.special_points: {unknown} not available on the path '
                f'cell; available: {sorted(points)}')
        letters = list(cfg.special_points)
        if dense:
            path = bandpath([points[c] for c in letters], cell, cfg.npts)
            x, X, _labels = path.get_linear_kpoint_axis()
            return path.kpts, letters, x, X
        return np.asarray([points[c] for c in letters], dtype=float), letters, None, None

    kpts = np.asarray(
        [np.asarray(k, dtype=float) for k in cfg.kpoints], dtype=float
    ).reshape(-1, 3)
    xcoords = getattr(cfg, 'xcoords', None)
    xqpts = np.arange(len(kpts), dtype=float) if xcoords is None \
        else np.asarray(xcoords, dtype=float)
    return kpts, cfg.names, xqpts, getattr(cfg, 'xticks', None)


# ---------------------------------------------------------------------------
# siesta
# ---------------------------------------------------------------------------

def run_siesta(cfg):
    from .siesta_unfold import unfold_siesta

    kpts, knames, xqpts, Xqpts = _resolve_path(cfg, default_cell=cfg.primitive)
    ax, res = unfold_siesta(
        fdf=cfg.fdf,
        prim_atoms=cfg.primitive,
        unfold_sc_mat=np.asarray(cfg.supercell_matrix, dtype=int),
        spin=cfg.spin,
        kpts=kpts,
        knames=knames,
        xqpts=xqpts,
        Xqpts=Xqpts,
        tol_r=cfg.tol_r,
        match_species=cfg.match_species,
        efermi=cfg.efermi,
        method=cfg.method,
        return_result=True,
    )
    _save_data(cfg, res, fermi_energy=cfg.efermi,
               energy_reference="absolute")
    return _save(ax, cfg.output)


# ---------------------------------------------------------------------------
# siesta-wfsx
# ---------------------------------------------------------------------------

def _wfsx_efermi(wfsx, explicit):
    """Fermi level of the WFSX's writing run (its .EIG header)."""
    if explicit is not None:
        return float(explicit)
    stem = os.fspath(wfsx)
    for suffix in (".selected.WFSX", ".WFSX"):
        if stem.endswith(suffix):
            cand = stem[: -len(suffix)] + ".EIG"
            if os.path.isfile(cand):
                with open(cand) as fh:
                    return float(fh.readline().split()[0])
    raise ConfigError(
        "options.efermi: no <wfsx stem>.EIG found next to the WFSX; "
        "set options.efermi explicitly")


def _load_siesta_model(fdf):
    """One spin channel of a SIESTA calculation parsed through HamiltonIO."""
    from .siesta_unfold import _load_channel_model

    return _load_channel_model(fdf, "up")


def run_siesta_wfsx(cfg):
    from HamiltonIO.siesta.wfsx import SiestaWFSXParser

    from .wfsx_unfolder import WFSXUnfolder

    sc = _load_siesta_model(cfg.hs_fdf)
    prim = _load_siesta_model(cfg.primitive)
    M = np.asarray(cfg.supercell_matrix, dtype=int)
    kpts, knames, xqpts, Xqpts = _resolve_path(
        cfg, default_cell=np.asarray(prim.atoms.cell, dtype=float))
    adapted, rm = _lcao_relabel(sc, prim.atoms, M, cfg.tol_r,
                                match_species=True)
    wfsx = SiestaWFSXParser(
        cfg.wfsx, cell=np.asarray(sc.atoms.cell, dtype=float)).read()
    unf = WFSXUnfolder(wfsx, adapted, rm, M)
    res = unf.compute(kpts, method=cfg.method)
    efermi = _wfsx_efermi(cfg.wfsx, cfg.efermi)
    from types import SimpleNamespace

    _save_data(cfg,
               SimpleNamespace(kpoints=res.kpoints,
                               eigenvalues=res.eigenvalues,
                               weights=res.weights),
               fermi_energy=efermi, energy_reference="absolute")
    return _band_figure(cfg, res.eigenvalues, res.weights,
                        xqpts, knames, Xqpts, efermi)


# ---------------------------------------------------------------------------
# phonopy
# ---------------------------------------------------------------------------

def run_phonopy(cfg):
    from ase.io import read as ase_read

    from .phonopy_unfolder import phonopy_unfold

    M = np.asarray(cfg.supercell_matrix, dtype=int)
    sc_mat = np.asarray(cfg.sc_mat, dtype=int)
    default_cell = None
    if not cfg.path_cell:
        sposcar_cell = np.asarray(ase_read(cfg.sposcar).cell.array, dtype=float)
        default_cell = np.linalg.inv(M.astype(float)) @ sposcar_cell
    kpts, knames, xqpts, Xqpts = _resolve_path(cfg, default_cell=default_cell)
    ax, res = phonopy_unfold(
        sc_mat=sc_mat,
        unfold_sc_mat=M,
        force_constants=cfg.force_constants,
        sposcar=cfg.sposcar,
        qpts=kpts,
        qnames=knames,
        xqpts=xqpts,
        Xqpts=Xqpts,
        return_result=True,
    )
    _save_data(cfg, res, energy_unit="cm^-1",
               energy_reference="absolute")
    return _save(ax, cfg.output)


# ---------------------------------------------------------------------------
# abinit-wfk
# ---------------------------------------------------------------------------
def run_abinit_wfk(cfg):
    from .abinit_unfold import unfold_abinit

    kpts, knames, xqpts, Xqpts = _resolve_path(cfg, default_cell=cfg.primitive)
    ax, res = unfold_abinit(
        wfk=cfg.wfk,
        unfold_sc_mat=np.asarray(cfg.supercell_matrix, dtype=int),
        kpts=kpts,
        knames=knames,
        xqpts=xqpts,
        Xqpts=Xqpts,
        spin=cfg.spin,
        resolve_degenerate=cfg.resolve_degenerate,
        average_degenerate=cfg.average_degenerate,
        fermi_shift=cfg.fermi_shift,
        return_result=True,
    )
    _save_data(cfg, res, energy_reference=(
        "fermi" if cfg.fermi_shift else "absolute"))
    return _save(ax, cfg.output)


# ---------------------------------------------------------------------------
# abinit-ddb
# ---------------------------------------------------------------------------

def _ddb_primitive_kpoints(kpoints, sc_mat):
    """Convert DDB-cell reciprocal fractions to primitive fractions.

    If ``A_ddb = M @ A_prim``, then ``q_prim = q_ddb @ M^-T`` so both
    coordinates describe the same Cartesian reciprocal vector.
    """
    return np.asarray(kpoints, dtype=float) @ np.linalg.inv(
        np.asarray(sc_mat, dtype=float)).T


def run_abinit_ddb(cfg):
    from .DDB_unfolder import DDB_unfolder

    default_cell = None
    if not cfg.path_cell and cfg.special_points:
        # the path frame defaults to the cell stored in the DDB
        import abipy.abilab as abilab

        with abilab.abiopen(cfg.ddb) as ddb:
            default_cell = np.asarray(
                ddb.structure.to_ase_atoms().cell.array, dtype=float)
    kpts, knames, _x, _X = _resolve_path(
        cfg, default_cell=default_cell, dense=False)
    ax, res = DDB_unfolder(
        cfg.ddb,
        kpath_bounds=kpts,
        sc_mat=np.asarray(cfg.sc_mat, dtype=float),
        knames=knames,
        dipdip=cfg.dipdip,
        return_result=True,
    )
    from types import SimpleNamespace

    primitive_kpoints = _ddb_primitive_kpoints(res.kpoints, cfg.sc_mat)
    _save_data(cfg, SimpleNamespace(kpoints=primitive_kpoints,
                                    eigenvalues=res.eigenvalues,
                                    weights=res.weights),
               energy_unit="cm^-1", energy_reference="absolute")
    return _save(ax, cfg.output)


# ---------------------------------------------------------------------------
# magnon
# ---------------------------------------------------------------------------

def _load_magnon(results, spin_conf):
    """Configured ``Magnon`` (collinear reference + optional moment override)."""
    from TB2J.magnon.magnon3 import Magnon

    magnon = Magnon.from_TB2J_results(path=str(results))
    magmoms = np.asarray(magnon.magmom, dtype=float)
    if spin_conf is not None:
        if len(spin_conf) != 3 * len(magmoms):
            raise ConfigError(
                f"options.spin_conf: needs 3 x nspin = {3 * len(magmoms)} "
                f"values, got {len(spin_conf)}")
        magmoms = np.asarray(spin_conf, dtype=float).reshape(-1, 3)
    # set_reference also initializes Snorm (required by Hq)
    magnon.set_reference(
        Q=(0, 0, 0),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([1.0, 0.0, 0.0]),
        magmoms=magmoms,
    )
    return magnon


def run_magnon(cfg):
    from .tb2j_unfold import unfold_tb2j

    M = np.asarray(cfg.supercell_matrix, dtype=int)
    magnon = _load_magnon(cfg.results, cfg.spin_conf)
    default_cell = None
    if not cfg.path_cell:
        # primitive cell derived through the unfold matrix (A_prim = M^-1 A_sc)
        default_cell = np.linalg.solve(
            M.astype(float), np.asarray(magnon.cell, dtype=float))
    kpts, knames, xqpts, Xqpts = _resolve_path(cfg, default_cell=default_cell)
    # pass the configured Magnon object so the reference settings (which may
    # carry spin_conf) are used as-is
    ax, res = unfold_tb2j(
        magnon,
        M,
        kpts,
        knames=knames,
        xqpts=xqpts,
        Xqpts=Xqpts,
        degen_tolerance=cfg.degen_tolerance,
        return_result=True,
    )
    _save_data(cfg, res, energy_unit="meV",
               energy_reference="absolute")
    return _save(ax, cfg.output)


# ---------------------------------------------------------------------------
# abinit-paw
# ---------------------------------------------------------------------------

def _xml_datasets(root):
    """Species -> JTH XML path map from a ``<Symbol>.xml`` directory."""
    datasets = {}
    for fname in sorted(os.listdir(root)):
        if fname.endswith(".xml"):
            datasets[fname[:-4]] = os.path.join(root, fname)
    return datasets


def _cumulative_cart_x(kpoints, cell):
    """x axis: cumulative Cartesian distance along the stored k-points."""
    bcart = 2 * np.pi * np.linalg.inv(np.asarray(cell, dtype=float)).T
    dk = np.diff(np.asarray(kpoints, dtype=float), axis=0) @ bcart
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(dk, axis=1))])


def _paw_figure(cfg, weights, energies, kpoints, cell):
    """Weight-coded band figure at the stored reference k-points."""
    from .plotphon import plot_band_weight

    x = _cumulative_cart_x(kpoints, cell)
    w = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    nb = w.shape[1]
    ax = plot_band_weight(
        [x] * nb,
        [energies[:, ib] for ib in range(nb)],
        [w[:, ib] for ib in range(nb)],
        xticks=[cfg.names, cfg.xticks]
        if cfg.names is not None and cfg.xticks is not None else None,
        ylabel="Energy (eV)",
        ypad=1.5,
    )
    return _save(ax, cfg.output)


def run_abinit_paw(cfg):
    from HamiltonIO.abinit import HARTREE_TO_EV, read_paw_wfk

    from .abinit_paw import unfold_abinit_paw

    datasets = _xml_datasets(cfg.paw)
    supercell = read_paw_wfk(cfg.supercell, datasets)
    primitive = read_paw_wfk(cfg.primitive, datasets)
    result = unfold_abinit_paw(
        supercell, primitive, datasets,
        np.asarray(cfg.supercell_matrix, dtype=int),
        spin=cfg.spin,
        resolve_degenerate=None if cfg.resolve_degenerate is None
        else cfg.resolve_degenerate / HARTREE_TO_EV,
    )
    from types import SimpleNamespace

    efermi = supercell.wavefunctions.fermi_energy * HARTREE_TO_EV
    _save_data(
        cfg,
        SimpleNamespace(kpoints=result.kpoints,
                        eigenvalues=result.eigenvalues * HARTREE_TO_EV,
                        weights=result.weights),
        fermi_energy=efermi, energy_reference="absolute")
    energies = result.eigenvalues * HARTREE_TO_EV - efermi
    return _paw_figure(cfg, result.weights, energies,
                       result.kpoints, primitive.wavefunctions.rprimd)


# ---------------------------------------------------------------------------
# openmx
# ---------------------------------------------------------------------------

def run_openmx(cfg):
    from ase.io import read as ase_read

    from .openmx_unfold import unfold_openmx

    prim = ase_read(cfg.primitive)
    kpts, knames, xqpts, Xqpts = _resolve_path(
        cfg, default_cell=np.asarray(prim.cell.array, dtype=float))
    ax, res = unfold_openmx(
        scfout=cfg.scfout,
        prim_atoms=prim,
        unfold_sc_mat=np.asarray(cfg.supercell_matrix, dtype=int),
        spin=cfg.spin,
        kpts=kpts,
        knames=knames,
        xqpts=xqpts,
        Xqpts=Xqpts,
        tol_r=cfg.tol_r,
        match_species=cfg.match_species,
        method=cfg.method,
        efermi=cfg.efermi,
        return_result=True,
    )
    _save_data(cfg, res, energy_reference="fermi")
    return _save(ax, cfg.output)


# ---------------------------------------------------------------------------
# LCAO composition shared by the gpaw / abacus routes
# ---------------------------------------------------------------------------

def _lcao_orb_counts(model, n_atoms):
    """Per-atom orbital counts of a HamiltonIO model.

    Uses the model's per-atom orbital list when it exposes one
    (``orbs`` with ``iatom``), else a uniform split of the total basis
    size; non-uniform bases without per-atom information need the
    Python API (``LCAOUnfolder`` + ``RelabelMap`` directly).
    """
    orbs = getattr(model, "orbs", None)
    if orbs:
        iatom = [int(getattr(o, "iatom", -1)) for o in orbs]
        if min(iatom) >= 0 and max(iatom) < n_atoms:
            counts = np.bincount(np.asarray(iatom, dtype=int),
                                 minlength=n_atoms)
            if (counts > 0).all():
                return [int(c) for c in counts]
    sr = getattr(model, "SR", None)
    if sr is None:
        raise ConfigError(
            f"structure: model {type(model).__name__} exposes no overlap "
            "table to derive orbital counts from")
    block = next(iter(sr.values())) if isinstance(sr, dict) else np.asarray(sr)[0]
    nao = int(np.asarray(block).shape[0])
    if nao % n_atoms:
        raise ConfigError(
            f"structure: {nao} orbitals cannot be split uniformly over "
            f"{n_atoms} atoms; pass per-atom counts via the Python API")
    return [nao // n_atoms] * n_atoms


def _prim_orb_counts(sc_atoms, counts_sc, prim_atoms):
    """Primitive counts from species-level supercell counts."""
    by_symbol = {}
    for symbol, count in zip(sc_atoms.get_chemical_symbols(), counts_sc):
        if by_symbol.setdefault(symbol, count) != count:
            raise ConfigError(
                f"structure: species {symbol} carries mixed orbital counts; "
                "pass counts via the Python API")
    out = []
    for symbol in prim_atoms.get_chemical_symbols():
        if symbol not in by_symbol:
            raise ConfigError(
                f"structure: no orbital counts for species {symbol!r} in "
                "the parsed supercell model")
        out.append(by_symbol[symbol])
    return out


def _lcao_relabel(sc_model, prim_atoms, M, tol_r, match_species):
    """(HamiltonIOModel, RelabelMap) for a parsed supercell model.

    Orbital counts come from the model's ``orb_dict`` when it exposes
    one (SIESTA/OpenMX convention, species-mapped onto the primitive
    cell), else from the model's per-atom orbital list or a uniform
    split of the basis size (GPAW/ABACUS).
    """
    from .lcao_unfolder import HamiltonIOModel
    from .mapping import RelabelMap

    from .siesta_unfold import _derive_prim_counts

    if isinstance(sc_model, HamiltonIOModel):
        adapted = sc_model
    elif callable(getattr(sc_model, "hs_and_eigen", None)) \
            and isinstance(getattr(sc_model, "SR", None), dict):
        # already backend-neutral (lowercase hs_and_eigen + dict SR), e.g.
        # HamiltonIO's GpawLcaoModel; consumed un-wrapped
        adapted = sc_model
    else:
        adapted = HamiltonIOModel(sc_model)
    orb_dict = getattr(sc_model, "orb_dict", None)
    if orb_dict is not None:
        counts_sc = orb_dict
        counts_prim = _derive_prim_counts(adapted.atoms, orb_dict, prim_atoms)
        if counts_prim is None:
            raise ConfigError(
                "structure: a species carries mixed orbital counts in the "
                "parsed model; pass counts via the Python API")
    else:
        counts_sc = _lcao_orb_counts(sc_model, len(adapted.atoms))
        counts_prim = _prim_orb_counts(adapted.atoms, counts_sc, prim_atoms)
    rm = RelabelMap.from_atoms(
        adapted.atoms, prim_atoms, M, tol_r=tol_r,
        orb_counts_sc=counts_sc, orb_counts_prim=counts_prim,
        match_species=match_species,
    )
    return adapted, rm


def _lcao_unfolder(sc_model, prim_atoms, M, tol_r, match_species):
    """LCAOUnfolder(sc) with the relabel map derived from the models.

    The model is wrapped into the backend-neutral ``HamiltonIOModel``
    view only when it does not already expose it (``hs_and_eigen`` plus
    dict ``SR``): HamiltonIO's ``GpawLcaoModel`` is consumed un-wrapped
    (its surface is already backend-neutral, lowercase ``hs_and_eigen``),
    while e.g. ABACUS models speak the batched ``HS_and_eigen`` dialect.
    """
    from .lcao_unfolder import LCAOUnfolder

    adapted, rm = _lcao_relabel(sc_model, prim_atoms, M, tol_r, match_species)
    return LCAOUnfolder(adapted, rm)


def _model_efermi(model):
    """Fermi level (eV) of a HamiltonIO model (0.0 when absent)."""
    for m in (model, getattr(model, "_model", None)):
        efermi = getattr(m, "efermi", None)
        if efermi is not None:
            return float(efermi)
    return 0.0


def _band_figure(cfg, eigenvalues, weights, xqpts, knames, Xqpts, efermi):
    """Weight-coded band figure for an (nk, nband) result on a path."""
    from .plotphon import plot_band_weight

    energies = np.asarray(eigenvalues, dtype=float) - efermi
    w = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    nb = w.shape[1]
    ax = plot_band_weight(
        [xqpts] * nb,
        [energies[:, ib] for ib in range(nb)],
        [w[:, ib] for ib in range(nb)],
        xticks=[knames, Xqpts]
        if knames is not None and Xqpts is not None else None,
        ylabel="Energy (eV)",
        ypad=1.5,
    )
    return _save(ax, cfg.output)


def _pw_eigendata(data):
    """PWEigenData from a HamiltonIO (GpawPWData | AbacusPWData) parse."""
    from .pw_unfolder import PWEigenData

    return PWEigenData(
        data.kpoints,
        data.gvecs,
        [c[None, :, None, :] for c in data.coefficients],
        data.eigenvalues[:, None, :],
    )


# ---------------------------------------------------------------------------
# gpaw
# ---------------------------------------------------------------------------

def run_gpaw(cfg):
    M = np.asarray(cfg.supercell_matrix, dtype=int)
    if cfg.mode == "lcao":
        if cfg.primitive is None:
            raise ConfigError(
                "structure.primitive: is required in lcao mode")
        from HamiltonIO.gpaw import GpawLcaoModel

        prim = GpawLcaoModel.from_file(cfg.primitive, spin=cfg.spin)
        sc = GpawLcaoModel.from_file(cfg.supercell, spin=cfg.spin)
        kpts, knames, xqpts, Xqpts = _resolve_path(
            cfg, default_cell=np.asarray(prim.atoms.cell, dtype=float))
        unf = _lcao_unfolder(sc, prim.atoms, M, cfg.tol_r, cfg.match_species)
        res = unf.compute(kpts, method=cfg.method)
        _save_data(cfg, res, fermi_energy=_model_efermi(sc),
                   energy_reference="absolute")
        return _band_figure(cfg, res.eigenvalues, res.weights,
                            xqpts, knames, Xqpts, _model_efermi(sc))
    from HamiltonIO.gpaw import GpawPWParser

    from .pw_unfolder import PWUnfolder

    data = GpawPWParser(cfg.supercell).read()
    unf = PWUnfolder(_pw_eigendata(data), M)
    default_cell = np.linalg.inv(M.astype(float)) @ np.asarray(data.cell, float)
    kpts, knames, xqpts, Xqpts = _resolve_path(cfg, default_cell=default_cell)
    res = unf.compute(kpts, resolve_degenerate=cfg.resolve_degenerate)
    _save_data(cfg, res, fermi_energy=data.efermi,
               energy_reference="absolute")
    return _band_figure(cfg, res.eigenvalues, res.weights,
                        xqpts, knames, Xqpts, data.efermi)


# ---------------------------------------------------------------------------
# abacus
# ---------------------------------------------------------------------------

def _abacus_lcao_model(outdir, spin):
    """One spin channel of an ABACUS LCAO model parsed from an OUT dir."""
    from HamiltonIO.abacus.abacus_wrapper import AbacusParser

    model = AbacusParser(outpath=outdir).get_models()
    if isinstance(model, tuple):
        model = model[{"up": 0, "down": 1}[spin]]
    return model


def _abacus_pw_default_cell(outdir, M):
    """Primitive path cell from the STRU next to an OUT dir, or None."""
    parent = os.path.dirname(os.fspath(outdir).rstrip(os.sep)) or "."
    for name in ("STRU", "Stru"):
        stru = os.path.join(parent, name)
        if os.path.isfile(stru):
            from HamiltonIO.abacus.stru_api import read_abacus

            cell = np.asarray(read_abacus(stru, verbose=False).cell.array,
                              dtype=float)
            return np.linalg.inv(M.astype(float)) @ cell
    return None


def run_abacus(cfg):
    M = np.asarray(cfg.supercell_matrix, dtype=int)
    if cfg.mode == "lcao":
        if cfg.primitive is None:
            raise ConfigError(
                "structure.primitive: is required in lcao mode")
        sc = _abacus_lcao_model(cfg.supercell, cfg.spin)
        prim = _abacus_lcao_model(cfg.primitive, cfg.spin)
        kpts, knames, xqpts, Xqpts = _resolve_path(
            cfg, default_cell=np.linalg.inv(M.astype(float))
            @ np.asarray(sc.atoms.cell, dtype=float))
        unf = _lcao_unfolder(sc, prim.atoms, M, cfg.tol_r, cfg.match_species)
        res = unf.compute(kpts, method=cfg.method)
        _save_data(cfg, res, fermi_energy=_model_efermi(sc),
                   energy_reference="absolute")
        return _band_figure(cfg, res.eigenvalues, res.weights,
                            xqpts, knames, Xqpts, _model_efermi(sc))

    from HamiltonIO.abacus.pw_wfc import AbacusPWParser

    from .pw_unfolder import PWUnfolder

    data = AbacusPWParser(outpath=cfg.supercell).read()
    unf = PWUnfolder(_pw_eigendata(data), M)
    kpts, knames, xqpts, Xqpts = _resolve_path(
        cfg, default_cell=_abacus_pw_default_cell(cfg.supercell, M))
    res = unf.compute(kpts, resolve_degenerate=cfg.resolve_degenerate)
    _save_data(cfg, res, fermi_energy=data.efermi,
               energy_reference="absolute")
    return _band_figure(cfg, res.eigenvalues, res.weights,
                        xqpts, knames, Xqpts, data.efermi)


# ---------------------------------------------------------------------------
# vasp-paw
# ---------------------------------------------------------------------------

def run_vasp_paw(cfg):
    from .vasp_paw import read_wavecar_ordered, unfold_vasp_paw

    supercell = read_wavecar_ordered(cfg.supercell, cfg.supercell_poscar)
    primitive = read_wavecar_ordered(cfg.primitive, cfg.primitive_poscar)
    result = unfold_vasp_paw(
        supercell, primitive, cfg.potcar,
        np.asarray(cfg.supercell_matrix, dtype=int),
        spin=cfg.spin,
        resolve_degenerate=cfg.resolve_degenerate,
    )
    _save_data(cfg, result, fermi_energy=supercell.fermi_energy,
               energy_reference="absolute")
    return _paw_figure(cfg, result.weights,
                       result.eigenvalues - supercell.fermi_energy,
                       result.kpoints, primitive.lattice)


# ---------------------------------------------------------------------------
# wannier
# ---------------------------------------------------------------------------

def run_wannier(cfg):
    from .wannier_unfold import read_wannier90_win, run as wannier_run
    default_path_cell = None
    primitive_cell = None
    if cfg.special_points:
        sc_cell = cfg.cell
        if sc_cell is None:
            win_path = os.path.join(cfg.path, f"{cfg.prefix}.win")
            if not os.path.isfile(win_path):
                raise ConfigError(
                    "Wannier special_points needs structure.cell or "
                    f"{win_path} with unit_cell_cart")
            try:
                sc_cell, _sites = read_wannier90_win(win_path)
            except (ValueError, IndexError) as exc:
                raise ConfigError(
                    f"{win_path}: {exc}; set structure.cell or add "
                    "unit_cell_cart to the win") from exc
        try:
            primitive_cell = np.linalg.solve(
                np.asarray(cfg.supercell_matrix, dtype=float),
                np.asarray(sc_cell, dtype=float))
        except np.linalg.LinAlgError as exc:
            raise ConfigError(
                "structure.supercell_matrix must be invertible for "
                "Wannier special_points") from exc
        if cfg.path_cell is None:
            default_path_cell = primitive_cell

    kpts, knames, _x, _X = _resolve_path(
        cfg, default_cell=default_path_cell, dense=False)
    if cfg.special_points and cfg.path_cell is not None:
        path_cell = _path_cell_array(cfg, default_path_cell)
        kpts = kpts @ np.linalg.solve(path_cell.T, primitive_cell.T)
    ax, res = wannier_run(
        path=cfg.path,
        prefix=cfg.prefix,
        labels=list(cfg.labels),
        scmat=np.asarray(cfg.supercell_matrix, dtype=int),
        output_figure=cfg.output,
        kvectors=kpts,
        knames=list(knames) if knames is not None else None,
        npoints=cfg.npoints,
        cell=cfg.cell,
        return_result=True,
        resolve_degenerate=cfg.resolve_degenerate,
    )
    _save_data(cfg, res, energy_reference="absolute")
    return ax
