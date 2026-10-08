# Adding a new unfolding to `unfolding`

Contributor recipe (story 048, shared-core epic 010). It documents the
*existing* seams — there is no framework to subclass. A worked example
lives in `tests/test_toy_tb_fixture.py` (nearest-neighbour TB
supercell through the production core).

## Decide what you are adding

**A new source** (a code/file format that produces bands on a
supercell): you need a *parser* only. Reuse one of the existing engines
and register a route.

**A new projection/physics** (a genuinely different weight definition,
e.g. a new quasiparticle type): you need one *kernel*. Consume the
shared modules below and keep the kernel engine-local.

## The seams you build on

| Seam | Module | Contract |
|---|---|---|
| Folding / cosets | `unfolding.cosets` | `fold(k, M)`, `unfold(K, M)`, `fold_kpoints(M)` (row convention `K = k @ M.T`; zero-first coset order), `sector(k, M, n_cells)`, `match_stored_k(stored, k, M, tol_k)` (L2 periodic-delta; WFSX/PAW keep their own metrics) |
| Degeneracy | `unfolding.degeneracy` | `energy_groups` (caller picks sorted-`order` vs in-place idiom), `assign_eigvalsh_groups` (Hermitian-block spectrum; `descending=False` for the PAW as-returned convention), `assign_total_groups` (magnon integer/fractional totals) |
| Results | existing `*Weights` dataclasses | Produce `kpoints (nk,3)`, energies/eigenvalues and `weights (nk,nb)`. **Do not** invent a common result base: LCAO/PW collapse degenerate groups and sum weights; magnon keeps all columns and averages. Keep each engine's semantics. |
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
- **Units**: the Dataset boundary is **eV** — `Dataset.from_result`
  has no unit argument (it defaults to eV); only `save_dataset`
  accepts an explicit `energy_unit`. Adapters convert at the boundary:
  e.g. the phonopy adapter converts THz → cm⁻¹ for its plot output,
  and a Dataset export of such a source must convert to eV (or pass
  `energy_unit` through `save_dataset`) instead of relying on
  `from_result`.
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
  `nspinor` axis; spinor LCAO keeps interlaced blocks.
- **Optional backends**: wrap imports in try/except naming the extra
  (`pip install unfolding[<extra>]`); the package core stays
  backend-free.

## Worked example: TB supercell through the production core

`tests/test_toy_tb_fixture.py` builds a one-orbital nn-TB 4×4×4
supercell with true-integer shell keys `T_sc = (D − d)/L` (D =
minimum-image displacement), diagonalizes at the commensurate grid,
and runs the character-sum engine + robust projector. It seals:
binary pristine weights, spectra = folded `eps(k+G)` over the coset,
group-total weight 1 at `eps(k)`, fractional weights under an onsite
defect, and a `Dataset.from_result` round-trip.

## Checklist for a new source

1. Parser → engine inputs (layout, units, frames per the table above).
2. Reuse an engine; add `SimpleNamespace`/dataclass result with the
   three contract fields.
3. `Dataset.from_result` export test (round-trip).
4. `RouteSpec` registration + CLI smoke (`unfolding <route> --help`).
5. Focused tests on commensurate + generic k, degeneracy planted.

## Checklist for a new projection

1. Derive the weight formula symbolically (SymPy) and seal numerically.
2. One engine module: kernel + result class; consume `cosets` and
   `degeneracy`; choose your aggregation strategy explicitly.
3. Old-vs-new or oracle equivalence evidence before review.
