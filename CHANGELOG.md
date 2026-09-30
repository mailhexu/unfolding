# Changelog


## Unreleased

### Fixed
- Unfolded-band JSON/plotting review fixes: `Dataset` now validates the
  (nk,3) coordinate arrays, rejects non-finite values except NaN weights,
  and preserves route/unit/Fermi/provenance metadata when re-saved.
  `plot_dataset` follows machine-readable `energy_reference` (`absolute`
  vs `fermi`), shares spin-channel limits, masks NaN weights for rendering,
  uses unit-aware y padding, and handles square Dataset overlays correctly.
  Wannier JSON k-points are converted back to primitive fractional frame;
  direct SIESTA/OpenMX adapter `output=` saving is restored; Wannier now
  permits data-only runs (`[output] data` without a figure path).
- Dataset visualization/reference fixes after review: the Wannier path
  coordinates are mapped back with the exact inverse of the reader’s
  `kvectors @ sc_matrix` convention, with a regression that checks the
  non-Gamma path vertices; square tuple overlays now require an explicit
  orientation (`kpoints_first` / `bands_first`); WFSX JSON stores absolute
  eigenvalues plus its Fermi energy, so dataset rendering applies the same
  single Fermi shift as the route figure.

- Wannier90 readers (`unfolding.wannier_unfold`): parse real Wannier90
  output. `read_wannier90_hr` now skips the "written on ..." comment
  line Wannier90 >= 2.0 prepends to `*_hr.dat` (real files raised
  `ValueError: invalid literal for int()`), the win reader accepts
  species-leading `atoms_cart` rows (`Ti<TAB>x y z`) and unit keyword
  lines, and `Wannier90Model` reads the orbital positions from the
  `.wout` Final State when `*_centres.xyz` is stale (centre count
  different from `num_wann`) instead of refusing to run. Verified by
  unfolding the real SrTiO3 datasets (pristine + Ti-vacancy
  sqrt(2)xsqrt(2)x2 supercells, 56 Wannier functions); regression tests
  in `tests/test_wannier_readers.py`.
- Correct the in-package Wannier90 reader's tight-binding convention: divide each
  `*_hr.dat` block by its Wigner-Seitz degeneracy, include the embedded Wannier
  centre Bloch phases without wrapping center images, and return band-major
  eigenvectors. `plot_unfolded_band`, `run`, and the Wannier config/CLI expose
  the optional energy tolerance (default: raw weights). The standalone script,
  figure generator, bundle config, and real-data reproducer opt in to 0.1 eV
  grouping for pristine folding only; Ti-vacancy outputs preserve raw weights
  to avoid rotating physically distinct defect levels. Regression coverage
  compares vacancy bands to frozen historical-reader energies and weights.
- `unf`/`phonopy_unfold` (phonopy route): pass `sc_mat.T` to the
  translation maps. `ase.make_supercell` builds lattice points with the
  transposed convention relative to phonopy's `A_sc = sc_mat^T @ A_prim`,
  so non-diagonal `unfold_sc_mat` (e.g. R-centred hexagonal cells,
  `M = inv(primitive_matrix)`) scrambled the translation orbits and
  produced fractional weights on pristine supercells. Verified on the MDR
  phonon-database R-3m dataset (Rb3B12H12I): weights are binary again
  (84 unit-weight modes per q); regression test with a non-symmetric
  supercell matrix added.

### Documentation

- Wannier90 SrTiO3 example: restore the real-data unfolding as the
  page's headline. The page now documents the committed pristine and
  Ti-vacancy SrTiO3 Wannier90 datasets (sqrt(2)xsqrt(2)x2 supercells,
  56 Wannier functions: 12 O-2p + 4 Ti-3d shells), embeds the bundle's
  `unfold.toml` verbatim, and links the shared method/parameters/k-path
  guide; the previous synthetic t2g figures move out of the page (the
  tiny synthetic model ships on as a smoke fixture in the bundle).
  `wannier-sto.tar.gz` now carries the real inputs (~7 MB compressed,
  packed via the new `examples/wannier-sto/bundle.manifest`), and
  `docgen/fig_wannier_sto.py` regenerates both figures.
- ABINIT WFK example: document the Si:P path-coverage analysis. The
  committed Si:P WFK fixtures store only the four path corners
  (supercell momenta (0,0,0), (0,1,0), (0.5,1,0), (0.5,0.5,0.5)), so of
  the 305 requested path folds only the 7 corner ticks have a stored
  match and the published Si:P map is the corner-fallback render. The
  example page now carries the stored-vs-requested coverage table,
  embeds the bundle's `unfold.toml` verbatim, and links the shared
  method/parameters/k-path guide instead of repeating shared-option
  explanations.

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
