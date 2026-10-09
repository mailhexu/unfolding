---
title: "Adding a new interface"
weight: 10
---

How to connect a new data source or a new projection to `unfolding`.
This page mirrors the contributor recipe in the repository
(`devdoc/adding-an-unfolding.md`), which is the canonical, most detailed
version with code snippets and a worked example
(`tests/test_toy_tb_fixture.py`).

## Decide what you are adding

**A new source** (a code or file format that produces bands on a
supercell): write a *parser* plus a thin *adapter*, reuse one of the
existing engines, and register a route. A parser alone is never
runnable: every engine consumes a normalized in-memory contract
(layout, units, gauge), and producing that contract is the adapter —
typically tens of lines of glue plus unit/gauge conversion, never new
weight math.

**A new projection** (a genuinely different weight definition, e.g. a
new quasiparticle type): write one *kernel* module. Consume the shared
`cosets` and `degeneracy` helpers and keep the kernel engine-local.

## Pick the engine your source feeds

| Your source provides | Engine to reuse | Input contract |
|---|---|---|
| Plane-wave eigendata (G-vectors + coefficients) | `pw_unfolder.PWUnfolder` | `PWEigenData`: `kpoints (nk,3)` supercell fractional; per k `gvecs[ik] (npw,3)` int, `coefficients[ik] (nspin,nband,nspinor,npw)` complex, `eigenvalues[ik] (nspin,nband)` in the source unit |
| An LCAO supercell Hamiltonian | `lcao_unfolder.LCAOUnfolder` | model exposing `atoms`, `HR`/`SR` dicts `{R: (norb_sc, norb_sc)}` and `hs_and_eigen(k)`; relabel via `mapping.RelabelMap.from_atoms(...)` |
| Phonons on a stored q-grid | `phonon_unfolder.phonon_unfolder` | `atoms`, `supercell_matrix`, `eigenvectors` (`evecs[iq, :, ibranch]` per mode), `qpoints` (supercell frame); robust weights via `get_weights_robust(frequencies, gauge="fold"|"bloch")` |
| Magnon eigendata (BdG) | `magnon_unfolder.MagnonUnfolder` | `MagnonEigenData`: `kpoints (nk,3)`, `energies (nk,nmode)`, `wavefunctions (nk,nmode,2nmag)` canonical Euclidean amplitudes, `positions (nmag,3)` supercell fractional |
| Stored-k reference bands with projectors | adapter-local physics (the PAW precedent) | parse supercell + primitive reference + projectors yourself; use `degeneracy.assign_eigvalsh_groups(..., descending=False)` |

## Conventions that bite

- **Row convention**: `K = k @ M.T` everywhere (`A_sc = M @ A_prim`).
- **Eigenvector layout**: engines consume `evecs[ik]` with rows = DOF,
  modes as columns.
- **Units**: the `Dataset` boundary is eV. Convert before
  `Dataset.from_result`, or pass `energy_unit` to `save_dataset`
  (shipped routes use `"cm^-1"` for phonons, `"meV"` for magnons).
- **Fermi metadata**: either absolute energies plus
  `fermi_energy=<eV>` with `energy_reference="absolute"`, or
  pre-shifted energies with `energy_reference="fermi"` — never both.
- **Normalization**: LCAO dual-basis weights need the overlap metric;
  BdG magnons are g-orthogonal (group-total presentation);
  plane-wave/character weights are Euclidean.
- **Gauge phases**: `exp(±2πi k·r)` signs differ per module by design;
  a folded-path engine distinguishes coset members with
  `k* = fold(k) − K_stored` without mod 1.
- **Optional backends**: lazy imports raising
  `ImportError` naming `pip install unfolding[<extra>]`; the core
  stays backend-free.

## Register a route (TOML + CLI + JSON for free)

Three edits, all derived from one dataclass so they cannot drift:

1. A config dataclass in `unfolding/config.py`: one field per
   parameter with `F(section, kind, help)` metadata; sections are
   `input, structure, path, options, output`; field kinds `path`,
   `out`, `str`, `int`, `float`, `bool`, `matrix` (3×3 ints),
   `kpoints`, `float-list`, `str-list`. Add it to `__all__` and the
   `RouteConfig` union.
2. A runner `run_<name>(cfg)` in `unfolding/routes.py`: resolve the
   path with `_resolve_path`, call the adapter with
   `return_result=True`, save the dataset with `_save_data`, the
   figure with `_save`.
3. One `RouteSpec` line in `config.ROUTES` (the runner import goes in
   the deferred import block at the bottom of `config.py`).

`unfolding <route> --help`, TOML loading, and the `--data` JSON output
then exist with no further code. Export the engine classes eagerly in
`unfolding/__init__.py` (backend-free) and adapter functions through
`_LAZY_EXPORTS`.

## Verify

1. **Engine/adapter**: pristine supercells give binary 0/1 weights on
   the commensurate grid; plant a near-degenerate pair at a generic k
   and check your group strategy.
2. **Dataset**: `Dataset.from_result` → `save_dataset` →
   `load_dataset` round-trip preserves shapes, units metadata, Fermi
   convention, and fold/sc k-points; `plot_dataset` renders.
3. **Route**: one TOML through `unfolding --config` and one flags
   invocation produce identical weights (the shipped pattern is
   `tests/test_routes_backends.py::test_siesta_wfsx_parity`); `import
   unfolding` still works without the optional extra.

For a runnable example, see the repository recipe
(`devdoc/adding-an-unfolding.md`) and its worked nearest-neighbour
tight-binding fixture; example data handling (committed manifests vs
the external runtime store) is described on the repository page
`devdoc/example-data.md`.
