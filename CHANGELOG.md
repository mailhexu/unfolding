# Changelog

## 0.2.0 (2026-09-20)

Renewal release: rewritten LCAO unfolding core with symbolic validation,
real SIESTA/phonopy seals, production packaging, docs, and a paper
pipeline. 23 commits, 81 files changed (+5365/−323).

### Added

- `LCAOUnfolder` + `RelabelMap`: general LCAO supercell unfolding for
  arbitrary supercell orientations, with dual-basis weights
  (`method="ring"`, the exact BvK-torus projection) and the standard
  Popescu-Zunger/Lee ideal weight (`method="ideal"`, generic momenta).
- `unfold_siesta`: one-call SIESTA adapter through HamiltonIO
  (collinear `spin="up"|"down"`, pre-parsed models supported).
- Spinor (noncollinear/SOC) unfolding: `RelabelMapSpinor`,
  `HamiltonIOModelSpinor`, `LCAOUnfolderSpinor` (ADR-003).
- Symbolic derivation suite (`derivations/`): weight identity, sum
  rule, and phonon projection identities derived with sympy on BvK
  rings and re-executed in CI.
- Validation suite (92 tests): primitive-table oracles, round-trip
  identities for arbitrary supercell matrices, folded-grid sum rules,
  a real SIESTA Si example (Γ and 2×2×2 k-grid archives, WFSX/EIG
  eigenvalue and eigenvector oracle seals), a real displaced-defect
  k-grid archive, a real FePt spin-orbit WFSX archive, and a synthetic
  27-shell multi-shell HSX fixture.
- `docgen/` figure pipeline (headless) with docs-build and link checks;
  rewritten tutorial (install extras, four adapter walkthroughs) and a
  curated API reference. `Unfolder` is deprecated in favor of
  `LCAOUnfolder` (ADR-003).

### Fixed

- Supercell-overlap assembly now accumulates every stored SR shell
  (the previous congruence filter silently dropped wrapped in-SC bonds
  on multi-shell archives).
- `phonon_unfolder.get_weight` crashed on `G=None` in the
  `phase=False` branch (the documented phonopy example path).
- ABINIT DDB Cu_fcc unfolding now uses gauge-robust Bloch-sum projectors
  and degenerate-block diagonalization; pristine weights are binary
  despite anaddb's real/cosine eigenvector gauge.
- Added regression seals for synthetic mixed gauges, real phonopy data,
  and the real Cu_fcc DDB example.
- Packaging: renewed `pyproject.toml`, optional extras
  (`phonopy`/`abipy`/`siesta`/`dev`), lazy backend imports, pytest
  suite, and GitHub Actions CI.

### Validation highlights

- Pristine supercells reproduce the primitive spectrum with unit
  weights at every grid momentum for arbitrary supercell orientations;
  the folded-grid sum rule holds to machine precision (pristine and
  defect).
- Real SIESTA Si: Γ weights split exactly 24×0 / 8×1 over 32 bands;
  WFSX-native eigenvectors independently reproduce the weights per
  eigenvalue run; the ideal weight is binary on the real 2×2×2 k-grid
  archive and SC bands match the independently parsed primitive
  k-grid bands within the converged-SCF tolerance (3.7 meV max).
