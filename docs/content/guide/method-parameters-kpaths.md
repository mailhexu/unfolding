---
title: "Method, parameters, and k-paths"
weight: 5
---

Every example page on this site runs the same underlying machinery: an
adapter reads a supercell calculation, an unfolding engine computes
spectral weights on primitive-cell momenta, and a weight-coded band
figure (and, optionally, a machine-readable dataset) is written. This
page defines the shared concepts once. Example pages cover only what is
specific to their code and fixture.

## The unfolding method in one paragraph

A supercell Bloch state at reduced momentum `K` contains primitive
momenta `k` that fold onto it modulo the reciprocal supercell lattice.
Each integer supercell reciprocal vector `G_s` maps to a primitive
fractional wavevector `w = M^-T @ (K + G_s)`. The spectral weight on
primitive momentum `k` is the total character carried by that reciprocal
coset:

```text
W_n(k) = sum_{G_s: frac(M^-T @ (K + G_s)) = k} |c_{nK}(G_s)|^2
```

The engines differ only in how that projector is evaluated:

| Engine | Route family | Weight definition |
|---|---|---|
| Plane wave | `abinit-wfk`, `gpaw` (pw), `abacus` (pw) | direct coset sum of squared plane-wave coefficients (orthonormal basis) |
| LCAO | `siesta`, `siesta-wfsx`, `openmx`, `gpaw`/`abacus` (lcao) | Lee et al. Eq. (25): dual-basis projector of the non-orthogonal tight-binding Hamiltonian |
| PAW projection | `abinit-paw`, `vasp-paw` | projection onto primitive-cell PAW bands |
| Magnon | `magnon` | Euclidean BdG projector sector sum on the spin-wave Hamiltonian |
| Phonon | `phonopy`, `abinit-ddb` | P. B. Allen et al. atom-translation projector |

## The supercell matrix

`unfold_sc_mat` (TOML `structure.supercell_matrix`, flag `--unfold-mat`)
is the integer matrix `M` with **row convention**:

```text
A_sc = M @ A_prim
```

i.e. row `i` of `M` expresses supercell lattice vector `i` in
primitive-lattice units. Example: the 8-atom conventional cubic cell of
fcc Si as a supercell of the 2-atom fcc primitive cell is
`M = [[-1,1,1],[1,-1,1],[1,1,-1]]`. For fcc conventional-cell DDBs the
matrix relates to the DDB cell instead (see the
[ABINIT DDB example](/examples/abinit-ddb/)).

## Requesting a k-path
- `[path] special_points = "GXWGLX"` builds a dense path through the
  special points of the **path frame cell** — by default the primitive
  cell of the route (or a derived cell for phonopy, magnon, DDB, and
  Wannier; override with `path_cell`). `npts` is the **total number of
  points on the path**, distributed over all path segments (Wannier keeps
  its existing `options.npoints` spelling).
- `[path] kpoints = [x1,y1,z1, ...]` lists explicit momenta instead
  (with optional `names`, `xcoords`, `xticks`).
- Internally each requested primitive `k` is mapped to the supercell
  momentum it folds onto, `K = k @ M.T`; every requested `k` must be
  present (within tolerance) in the stored supercell grid — a Γ-only
  supercell run, for instance, only serves the momenta that fold to
  supercell Γ.

## Shared parameters

| Parameter | Where | Meaning |
|---|---|---|
| `supercell_matrix` / `--unfold-mat` | `[structure]` | `M` as above |
| `spin` | `[options]` | collinear channel (`up`/`down`); one channel per run on spin-polarised data |
| `mode` | `[options]` (gpaw, abacus) | `pw` (plane-wave coefficients) or `lcao` (localized basis) |
| `method` | `[options]` (LCAO routes) | `ring` (exact torus projection, supercell-torus k-grids only) or `ideal` (arbitrary k-paths) |
| `resolve_degenerate` | `[options]` (PW/PAW and Wannier routes) | eV tolerance; reassigns sector/eigenvector weights within near-degenerate groups. Wannier defaults to raw weights; enable only for physically degenerate fold groups (e.g. pristine Wannier bands), not defect bands. |
| `npts` | `[path]` | total number of interpolated points across the full special-point path |
| `path_cell` | `[path]` | path-frame cell when the special points are meant in a frame other than the primitive cell |
| `output` | `[output]` | figure path (PNG) |
| `data` | `[output]` | unfolded-band dataset path (JSON, schema below) |

## Energy references

Figures show energies in the route's natural unit (eV for electronic
routes, meV for magnons, cm⁻¹ for phonons), with the reference recorded
explicitly in each JSON dataset. `energy_reference = "absolute"` means
energies are absolute (and may include `fermi_energy`);
`energy_reference = "fermi"` means the route already shifted energies to
E_F = 0. The default `plot_dataset` follows this marker and only shifts
absolute energies when a Fermi level is available.

## Persisting and re-plotting: the JSON dataset

Every route accepts `[output] data = "bands.json"` (flag `--data`).
The document (schema `unfolding.dataset/1`) stores the primitive
k-points, per-(k, band) energies and unfolded weights (NaN = masked
band, serialized as `null`), optional fold/supercell k-points, the
Fermi level, the energy unit, and a small provenance record of the
route parameters:

```python
from unfolding import load_dataset, plot_dataset

ds = load_dataset("bands.json")       # parse back to arrays
plot_dataset(ds)                       # weight-coded bands

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plot_dataset(ds, ax=ax1)                       # your own subplots
plot_dataset(ds, ax=ax2, overlay=ds_prim,      # primitive reference
             overlay_shift=-0.13)              # explicit alignment shift
```

`plot_dataset` renders through the same weight-coding as the route
figures, draws into any `matplotlib` Axes you supply, and can overlay
primitive reference bands (a second dataset or `(x, energies)` arrays).
The overlay shift is always explicit — aligning two runs is a physical
decision (e.g. the median potential offset between the runs), never
inferred.

## Is a primitive cell required?

| Route | `primitive` input | Role |
|---|---|---|
| `siesta`, `siesta-wfsx`, `openmx` | **required** | relabel map (weights) and default path cell |
| `gpaw`, `abacus` (`lcao`) | **required** | same |
| `gpaw`, `abacus` (`pw`) | not used | path cell derived from the run's cell |
| `abinit-wfk` | optional | default path cell for `special_points` only; weights never need it |
| `abinit-paw`, `vasp-paw` | **required** | primitive-cell wavefunctions the supercell is projected onto |
| `phonopy` | not used | path cell derived from `SPOSCAR` and `M` |
| `abinit-ddb` | not used | path frame defaults to the DDB cell |
| `magnon` | not used | path cell derived from the TB2J cell and `M` |
| `wannier` | not used | the Wannier Hamiltonian already knows its cell |
