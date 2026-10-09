# Adding a new unfolding to `unfolding`

Contributor recipe (story 048, shared-core epic 010). It documents the
*existing* seams — there is no framework to subclass. A worked example
lives in `tests/test_toy_tb_fixture.py` (nearest-neighbour TB
supercell through the production core).

## Decide what you are adding

**A new source** (a code/file format that produces bands on a
supercell): you need a *parser* plus a thin *adapter*. Reuse one of the
existing engines and register a route. In practice a parser alone is
never runnable: every engine consumes a normalized in-memory contract
(layout, units, gauge), and producing that contract *is* the adapter —
typically tens of lines of glue plus the unit/gauge conversion, never
new weight math. Compare `unfolding/abinit_unfold.py` (WFK parsing
lives in HamiltonIO; the adapter builds `PWEigenData`, converts
Hartree→eV, shifts the Fermi level) with `unfolding/wfsx_unfolder.py`
(SIESTA's stored gauge needs a real per-orbital conversion) and
`unfolding/tb2j_unfold.py` (TB2J's Cholesky-frame modes need the
canonical-BdG conversion). None of them touches a kernel.

**A new projection/physics** (a genuinely different weight definition,
e.g. a new quasiparticle type): you need one *kernel*. Consume the
shared modules below and keep the kernel engine-local — the PAW
adapters (`abinit_paw.py`, `vasp_paw.py`) are the precedent: their
projector matrices live in the adapter, while degenerate-group
assignment goes through the shared `degeneracy` helpers.

## Pick the engine your source feeds

| Your source provides | Engine to reuse | Engine input contract | Result | Shipped consumers |
|---|---|---|---|---|
| Plane-wave eigendata (G-vectors + coefficients) | `pw_unfolder.PWUnfolder` | `PWEigenData`: `kpoints (nk,3)` SC fractional; per k-point `gvecs[ik] (npw,3)` int, `coefficients[ik] (nspin,nband,nspinor,npw)` complex, `eigenvalues[ik] (nspin,nband)` in the source unit | `PWWeights` (adds `fold_kpoints`, `sc_kpoints`) | `abinit_unfold.py` (HamiltonIO `read_wfk`); `routes._pw_eigendata` wraps HamiltonIO `GpawPWData`/`AbacusPWData` |
| An LCAO supercell Hamiltonian model | `lcao_unfolder.LCAOUnfolder` | model exposing `atoms`, `HR`/`SR` dicts `{R_tuple: (n_orb_sc, n_orb_sc)}` over supercell translations and `hs_and_eigen(k)`; plus `mapping.RelabelMap.from_atoms(sc_atoms, prim_atoms, M, tol_r=..., orb_counts_sc=..., orb_counts_prim=..., match_species=...)` | `LCAOWeights` | `siesta_unfold.py`, `openmx_unfold.py`, the gpaw/abacus LCAO routes (`routes._lcao_relabel`/`_lcao_unfolder`); `WFSXUnfolder` reuses its overlap internals without diagonalizing |
| Phonons on a stored supercell q-grid | `phonon_unfolder.phonon_unfolder` | constructor takes `atoms`, `supercell_matrix`, `eigenvectors`, `qpoints` (SC frame); the shipped phonopy/DDB adapters use `phase=False`, then `get_weights_robust(frequencies, gauge="fold")` for phonopy or `gauge="bloch"` for DDB | `SimpleNamespace(kpoints, eigenvalues, weights)` from the adapter | `phonopy_unfolder.py`, `DDB_unfolder.py` |
| Magnon eigendata (BdG) | `magnon_unfolder.MagnonUnfolder` | `MagnonEigenData`: `kpoints (nk,3)` SC fractional; `energies (nk,nmode)`; `wavefunctions (nk,nmode,2nmag)` canonical Euclidean amplitudes (u block then v); `positions (nmag,3)` SC fractional | `MagnonWeights` (adds `sc_kpoints`) | `tb2j_unfold.magnon_eigendata_from_tb2j` (Cholesky frame → `Psi = K @ X`, rows renormalized; Cartesian positions → fractional via `inv(cell)`) |
| Stored-k reference bands with PAW projectors | (no engine — adapter-local physics) | parse supercell + primitive reference + projectors yourself | duck-typed `kpoints/eigenvalues/weights` | `abinit_paw.unfold_abinit_paw`, `vasp_paw.unfold_vasp_paw` (both use `degeneracy.assign_eigvalsh_groups(..., descending=False)` for the as-returned spectrum) |

Stored-k routes are a special case worth noting: when the path is
carried by the *primitive reference file* (abinit-paw, vasp-paw), there
are no `path.special_points`/`path.kpoints` fields at all — only tick
`names`/`xticks`. `config.validate_config` detects this shape from the
dataclass fields.

## The seams you build on

| Seam | Module | Contract |
|---|---|---|
| Folding / cosets | `unfolding.cosets` | `fold(k, M)`, `unfold(K, M)`, `fold_kpoints(M)` (row convention `K = k @ M.T`; zero-first coset order), `sector(k, M, n_cells)`, `match_stored_k(stored, k, M, tol_k)` (L2 periodic-delta; WFSX/PAW keep their own metrics) |
| Degeneracy | `unfolding.degeneracy` | `energy_groups` (caller picks sorted-`order` vs in-place idiom), `assign_eigvalsh_groups` (Hermitian-block spectrum; `descending=False` for the PAW as-returned convention), `assign_total_groups` (magnon integer/fractional totals) |
| Results | existing `*Weights` dataclasses | Produce `kpoints (nk,3)`, energies/eigenvalues and `weights (nk,nb)`. **Do not** invent a common result base: LCAO/PW collapse degenerate groups and sum weights; magnon keeps all columns and averages. Keep each engine's semantics. A duck-typed `SimpleNamespace` with the three fields is acceptable at the route boundary (the DDB and phonopy adapters do exactly that). |
| Consumer export | `Dataset.from_result(result, route=..., ...)` | Accepts `eigenvalues` *or* `energies`; optional `fold_kpoints`/`sc_kpoints`; owns frozen copies; carries unit/Fermi/spin/provenance metadata. This is the single result→consumer conversion. |
| CLI/TOML | `config.ROUTES` (`RouteSpec`) | Config dataclass + `routes.run_<name>(cfg)` runner + one registry line. Validation, flags and `--data` output come from the registry. |

## Conventions to get right (each has bitten us)

- **Row convention**: `K = k @ M.T` everywhere (`A_sc = M @ A_prim`).
  The reciprocal fold vectors solve `M @ t = n`.
- **Eigenvector layout**: engines consume `evecs[ik]` with rows = DOF
  and modes as columns (`evecs[ik, :, imode]` is one mode). Produce that
  layout however your source stores it — e.g. the DDB adapter fills
  `evecs[iq, :, ibranch]` mode by mode from displacement patterns; only
  transpose when the source is genuinely mode-major.
- **Units**: `Dataset.from_result` defaults to **eV** and has no unit
  argument; `save_dataset` accepts an explicit `energy_unit`. Convert
  a source's native energies to eV before `from_result`, or use
  `save_dataset(..., energy_unit=...)` with the actual unit. The shipped
  routes pass `energy_unit="cm^-1"` (phonopy/DDB) and `"meV"` (magnon);
  electronic routes stay eV.
- **Fermi/reference metadata**: pick one of the two shipped patterns
  and record it. Either keep absolute energies and pass
  `fermi_energy=<eV>` with `energy_reference="absolute"` (siesta, gpaw,
  abacus, PAW routes), or pre-shift the energies and pass only
  `energy_reference="fermi"` (openmx, abinit-wfk with
  `fermi_shift=True`). `plot_dataset` shifts only absolute datasets
  that carry a `fermi_energy`, so the consumer never double-shifts.
- **Normalization**: dual-basis (LCAO) weights need the overlap metric
  (`w = c† S⁻¹ c`); BdG magnons are g-orthogonal — per-mode spectra in
  degenerate groups are not presentation-stable, hence the group-total
  strategy. Plane-wave/character weights are Euclidean.
- **Phases/gauge**: `exp(±2πi k·r)` signs differ per module *by design*
  (legacy `Unfolder` keeps its historical `+` sign). Note two distinct
  robust-projector contracts: the **public**
  `phonon_unfolder.get_weights_robust(frequencies, gauge=...)`
  targets the *stored* q itself (phonopy/DDB contract; its phase is
  `qpt − qcarry`); a **folded-path** engine that must distinguish
  primitive coset members instead uses `k* = fold(k) − K_stored`
  **without** mod 1 — that folded-target form is exercised by the TB
  fixture (`tests/test_toy_tb_fixture.py::_robust_weights`), which
  composes the public character-sum machinery, and is the formula a new
  engine on a folded k-path needs.
- **Spin**: PW engines select one collinear `spin` channel and sum the
  `nspinor` axis; spinor LCAO keeps interlaced blocks (the σ,σ′
  reduction is sealed in `derivations/lcao_weight.py::spinor_reduction`
  and `tests/test_spinor_unfolder.py`; non-collinear SIESTA runs go
  through the ordinary `siesta` route on the doubled orbital basis).
- **Optional backends**: wrap imports in try/except naming the extra
  (`pip install unfolding[<extra>]`); the package core stays
  backend-free. `tests/test_import_guards.py` pins this.

## Registration: one config, one runner, one registry line

Three edits make a source reachable from TOML, the CLI, and
`unfolding.run(cfg)`. The TOML schema, the argparse flags, and the
`--data` JSON output are all derived from the same dataclass, so they
cannot drift.

**1. Config dataclass in `unfolding/config.py`.** One field per
parameter, each with `F(section, kind, help, flag=None, choices=None)`
metadata; field order is the documentation/help order. Sections are the
fixed tuple `input, structure, path, options, output`. Field kinds
(selected by `kind`, enforced by `config._coerce_value`/`validate_config`):

| kind | TOML/flag shape | validation |
|---|---|---|
| `path` | string | must exist on disk at validation time |
| `out` | string | no existence check (output path) |
| `str` / `int` / `float` / `bool` | scalar | type-checked; `bool` via `--flag/--no-flag` |
| `matrix` | 3×3 integers | coerced to `list[list[int]]` |
| `kpoints` | TOML: list of three-number lists; CLI: flat triples | ≥ 2 points, normalized to a list of `(x, y, z)` triples |
| `float-list` / `str-list` | list | homogeneous element check |

`flag=` overrides the default `--dashed-name`; `choices=` restricts
string values (both CLI and TOML). Copy the shared `[path]`/`[output]`
field block from a sibling dataclass (`special_points`/`kpoints`/`npts`/
`path_cell`/`names`/`xcoords`/`xticks`; `output`/`data`). Add the class
to `config.__all__` and to the `RouteConfig` union.

**2. Runner in `unfolding/routes.py`.** Runners are thin wrappers —
the adapters keep their explicit-parameter signatures and all physics:

For example, the existing WFK runner is:

```python
def run_abinit_wfk(cfg):
    from .abinit_unfold import unfold_abinit

    kpts, knames, xqpts, Xqpts = _resolve_path(cfg, default_cell=cfg.primitive)
    ax, res = unfold_abinit(
        wfk=cfg.wfk,
        unfold_sc_mat=np.asarray(cfg.supercell_matrix, dtype=int),
        kpts=kpts, knames=knames, xqpts=xqpts, Xqpts=Xqpts,
        spin=cfg.spin, resolve_degenerate=cfg.resolve_degenerate,
        average_degenerate=cfg.average_degenerate,
        fermi_shift=cfg.fermi_shift, return_result=True,
    )
    _save_data(cfg, res, energy_reference=(
        "fermi" if cfg.fermi_shift else "absolute"))
    return _save(ax, cfg.output)
```

Reuse, in this order: `_resolve_path` (bandpath interpolation for
`special_points`, or the explicit `kpoints` list; `dense=False` only for
adapters that interpolate themselves, like the DDB), the adapter call
with `return_result=True`, `_save_data` (writes the `[output] data`
JSON with route/provenance metadata; `routes._provenance` records a
fixed key subset, never path arrays), `_band_figure` or the adapter's
own axes, `_save`. Extend `routes.__all__`.

**3. Wire the registry — back in `unfolding/config.py`.** `routes.py`
imports `ConfigError` from `config`, so the runner import lives in the
*deferred* `from .routes import (...)` block after the dataclasses:

Add the new runner to that deferred import block and its `RouteSpec`
to `ROUTES`. The existing WFK entry shows the required wiring:

```python
"abinit-wfk": RouteSpec(
    name="abinit-wfk", config=AbinitWfkConfig, runner=run_abinit_wfk,
    path_cell_field="primitive"),
```

`path_cell_field` names the config field holding the default
special-points cell (`None` when the runner derives the cell itself —
ddb, phonopy, magnon and wannier do). `validate_config` then enforces
`special_points` XOR `kpoints`, required fields, path existence, and
`choices` for you.

**4. CLI, TOML, public exports.** No parser or console-script change is
needed: `unfolding example --help` lists fields once the dataclass and
registry entry exist, and `route = "example"` TOMLs load through
`load_config`. `unfolding` (registry-driven) and the legacy
`unfolding-magnon` remain the entry points. For the Python API: export
engine and `*Weights` classes eagerly in `unfolding/__init__.py`
(they are backend-free), and adapter functions through the
`_LAZY_EXPORTS` map (`name: (module, extra)`; `extra=None` when the
module imports cheaply or degrades itself, as `siesta_unfold` does).
Never re-export a function whose name collides with its module (see the
`DDB_unfolder` comment in `__init__.py`). If the parser needs an
optional backend, add an extra in `pyproject.toml`
(`siesta = ["HamiltonIO[siesta]>=0.3.8"]` is the pattern) and raise
`ImportError` naming `pip install unfolding[<extra>]` in the adapter.

## Trace a shipped route end to end (`abinit-wfk`)

Read these once; every route follows the same path.

1. **TOML → config**: `load_config("unfold.toml")` reads `route =
   "abinit-wfk"`, buckets the remaining tables into the fixed sections,
   rejects unknown sections/keys, coerces each value by field kind, and
   runs `validate_config` (required `[input] wfk` exists on disk,
   `path.special_points` XOR `path.kpoints`; `structure.primitive` is
   required for named paths because `path_cell_field="primitive"`).
   The flag path is identical: `unfolding abinit-wfk --wfk ... --unfold-mat
   -1 1 1 1 -1 1 1 1 -1 ...` → `config_from_flags` re-nests matrix/kpoint
   flats and validates the same way. Either object runs via
   `unfolding.run(cfg)`, which dispatches on `isinstance` over `ROUTES`.
2. **Runner**: `run_abinit_wfk` resolves the k-path with
   `_resolve_path(cfg, default_cell=cfg.primitive)` — a dense
   interpolation when `special_points` is set, otherwise the explicit
   `kpoints` with `xcoords`/`xticks` passthrough.
3. **Adapter**: `unfold_abinit(wfk=..., return_result=True)` calls
   HamiltonIO's `read_wfk`, guards `data.usepaw`, builds
   `PWEigenData(data.kpoints, data.gvecs, data.coefficients,
   data.eigenvalues)` — the four-field contract from the table above,
   energies still in Hartree — and computes
   `PWUnfolder(eigendata, M).compute(kpts, spin=..., resolve_degenerate=
   tol / HARTREE_TO_EV)` (the config field help records eV, the engine
   speaks the source unit; the adapter owns the conversion).
4. **Boundary conversion**: the adapter returns a fresh `PWWeights`
   with `eigenvalues * HARTREE_TO_EV - shift` (`shift` = the WFK Fermi
   level when `fermi_shift=True`) and plots through `PWWeights.plot`.
5. **Dataset + figure**: back in the runner,
   `_save_data(cfg, res, energy_reference="fermi" if cfg.fermi_shift
   else "absolute")` writes the `[output] data` JSON (schema
   `unfolding.dataset/1`) with route, provenance, reference metadata;
   `_save` writes the `[output] output` figure. Because the energies
   were pre-shifted, no `fermi_energy` is attached — the "fermi"
   reference records that fact.
6. **Consumer**: `load_dataset` returns a frozen `Dataset`;
   `plot_dataset(ds)` renders weight-coded bands, shifting only
   absolute datasets that carry `fermi_energy`. `save_dataset`/`load_dataset`
   round-trip NaN-masked weights as JSON `null`.

## Local example data and verification

Runnable examples follow the bundle layout under `examples/<name>/`:
`bundle/unfold.toml` (the exact CLI/TOML invocation),
`bundle/reproduce.py` (the Python-API equivalent that regenerates the
published figure), `bundle/README.txt`, and `bundle.manifest` — lines
`repo/relative/source -> bundle/destination` (comments with `#`),
pulling committed fixtures from `tests/data/` or committed examples.

Large or private runtime inputs are **not** committed. Register them in
`docgen/local-data.toml` — one `["<bundle-name>"]` table mapping
bundle destinations to paths relative to the external data store
(default `<checkout-sibling>/unfolding-data/`, override with
`UNFOLDING_DATA_DIR` or `--data-root`) — and assemble a runnable tree
without copying bytes into Git:

```bash
python docgen/stage_examples.py abinit-wfk-si --output ../.tmp/example-run
python docgen/stage_examples.py --check      # verify inputs without staging
```

Staging copies the bundle's scripts/configs and symlinks the manifest
and catalog inputs; it fails clearly on missing or empty sources and
refuses targets inside the checkout or the data store. Staged inputs
are symlinks into the store or committed fixtures: run the *CLI/TOML*
on the staged tree and write figures/datasets into scratch, never the
data store. Do not run `reproduce.py` generators against a staged tree.
Keep runtime output in a workspace `.tmp/` directory. Checksums,
provenance and the full store procedure are in `devdoc/example-data.md`;
`docgen/pack_bundles.py` (10 MB cap) is only for the downloadable docs
archives.

Verify the integration at three levels (the shape to copy is
`tests/test_routes_backends.py::test_siesta_wfsx_parity`, which asserts
TOML ≡ flags ≡ the raw adapter composition):

1. **Engine/adapter**: pristine supercells give binary 0/1 weights on
   the commensurate grid; plant a near-degenerate pair at a generic k
   and check the group strategy you chose (`average_degenerate`,
   `resolve_degenerate`, or `assign_*_groups`).
2. **Dataset**: `Dataset.from_result` → `save_dataset` →
   `load_dataset` round-trip preserves shapes, units metadata, Fermi
   convention, and optional fold/sc k-points; `plot_dataset` renders.
3. **Route**: `unfolding <route> --help` lists every field; one TOML
   through `unfolding --config` and one flags invocation produce
   identical weights; `import unfolding` still works without the
   optional extra installed.

## Worked example: TB supercell through the production core

`tests/test_toy_tb_fixture.py` builds a one-orbital nn-TB 4×4×4
supercell with true-integer shell keys `T_sc = (D − d)/L` (D =
minimum-image displacement), diagonalizes at the commensurate grid,
and runs the character-sum engine + robust projector. It seals:
binary pristine weights, spectra = folded `eps(k+G)` over the coset,
group-total weight 1 at `eps(k)`, fractional weights under an onsite
defect, and a `Dataset.from_result` round-trip.

## Checklist for a new source

1. Parser → engine inputs (layout, units, frames per the engine table
   above); thin adapter module — no kernel edits.
2. Reuse an engine; add `SimpleNamespace`/dataclass result with the
   three contract fields (plus `fold_kpoints`/`sc_kpoints` when the
   engine provides them).
3. `Dataset.from_result` export test (round-trip), with the unit and
   Fermi/reference decision recorded.
4. `XConfig` dataclass + `run_x` runner + deferred import + `ROUTES`
   line (see *Registration*); CLI smoke (`unfolding <route> --help`).
5. Example bundle: `bundle.manifest` pulls for committed fixtures,
   `docgen/local-data.toml` entries for external runtime inputs,
   `bundle/unfold.toml` + `reproduce.py`;
   `python docgen/stage_examples.py <name> --check` passes.
6. Focused tests on commensurate + generic k, degeneracy planted.

## Checklist for a new projection

1. Derive the weight formula symbolically (SymPy) and seal numerically.
2. One engine module: kernel + result class; consume `cosets` and
   `degeneracy`; choose your aggregation strategy explicitly.
3. Old-vs-new or oracle equivalence evidence before review.
