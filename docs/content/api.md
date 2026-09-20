---
title: "API Reference"
weight: 3
---

# API Reference

Curated reference for the public interfaces (see `unfolding/__init__.py`).

## Consumer entry points

### `unfold_siesta(fdf=None, model=None, prim_atoms=None, unfold_sc_mat=None, spin="up", kpts=None, knames=None, xqpts=None, Xqpts=None, tol_r=0.04, orb_counts_prim=None, efermi=0.0, axis=None, output=None, ...)`

One-call SIESTA LCAO unfolding. Parses `fdf` through HamiltonIO (or
takes a pre-parsed `model`), builds the relabel map from
`prim_atoms`/`unfold_sc_mat`, computes supercell eigenvalues and
unfolded weights on `kpts`, and plots weight-coded bands
(`LCAOWeights`-compatible). `spin` selects the spin channel of a
collinear run.

### `phonopy_unfold(sc_mat, unfold_sc_mat, force_constants, sposcar, qpts, qnames=None, xqpts=None, Xqpts=None)`

Phonon supercell unfolding from phonopy FORCE_CONSTANTS: builds the
supercell dynamical matrix, unfolds each mode onto the primitive
q-path, and plots weight-coded phonon bands.

### `DDB_unfolder(dbname, sc_mat, kpath_bounds, knames, ...)`

Unfolds phonon bands from an Abinit DDB (requires abipy + anaddb;
external dependencies: `abipy`, `nc_unfolder` for netCDF DDBs).

### `Unfolder` (deprecated)

Legacy generic Bloch-sum unfolding class. Superseded by
`LCAOUnfolder` + `RelabelMap` (ADR-003); kept only for backward
compatibility with old scripts.

## Building blocks (LCAO path)

### `RelabelMap.from_atoms(sc_atoms, prim_atoms, unfold_sc_mat, tol_r=0.04, orb_counts_sc=None, orb_counts_prim=None)`

Index map from supercell-orbital indices to primitive-orbital
translations: determines the orbital translation matrix
(sc-orbital -> prim-orbital + lattice translation) used by the
unfolding Gram. `orb_counts_*` may be given explicitly; otherwise
they are inferred from the ASE atoms (element -> orbital table).

### `HamiltonIOModel(model)`

Adapter exposing a HamiltonIO-parsed Hamiltonian
(`.HS_and_eigen`, `.SR`, `.HR`, `.atoms`) to the unfolder.

### `LCAOUnfolder(model, relabel_map, ndim3=1)`

`.compute(kpoints, method="ring") -> LCAOWeights` with
`LCAOWeights(kpoints, eigenvalues, weights)`. `method="ring"` (default)
is the exact supercell-torus projection (Parseval over the folded
grid); `method="ideal"` reproduces the standard Popescu-Zunger / Lee
weight at generic momenta and requires multi-shell (k-sampled)
archives.

### `spinor` module

Spinor (nspinor=2) variant: `RelabelMapSpinor.from_atoms`,
`HamiltonIOModelSpinor`, `LCAOUnfolderSpinor` (story 010).

## Plotting

### `plotphon.plot_band_weight(kslist, ekslist, wkslist, xticks, ylabel=..., ypad=..., ...)`

Weight-coded band plot used by all adapters: scatter-coloured bands
with x ticks at high-symmetry points.
