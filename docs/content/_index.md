---
title: "Introduction"
weight: 1
---

`unfolding` maps band structures computed in a **supercell** back onto the
Brillouin zone of the **primitive cell** — the standard "band unfolding"
analysis for defect, distorted, or commensurately-modulated calculations.

It works for electrons and phonons, and reads the output of several popular
DFT and force-constant codes:

- **SIESTA** Hamiltonians and wavefunctions (LCAO, spinors supported)
- **ABINIT** planewave wavefunctions (ETSF netCDF WFK) and phonon DDB files
- **phonopy** force constants
- **Wannier90** tight-binding Hamiltonians

Weights follow the Popescu–Zunger / Lee *et al.* formulation; every adapter
returns a matplotlib `Axes` with weight-coded (alpha) bands.

## Where to go next

- [Installation](install/) — pip extras per code backend
- [Example gallery](examples/) — figures with entry points to each walkthrough
- [API reference](api/) — consumer entry points and building blocks

## The one idea behind unfolding

A supercell calculation folds several primitive-cell momenta
$\mathbf{k}_m = \mathbf{K} + \mathbf{G}_m$ onto each supercell momentum
$\mathbf{K}$. The unfolding weight

$$
W_n(\mathbf{k}_m) = \sum_{\mathbf{G}_s \in m}
\left|c_{n\mathbf{K}}(\mathbf{G}_s)\right|^2
$$

measures how much of supercell band $n$ lives in fold sector
$\mathbf{k}_m$. For a pristine crystal the weights are 0 or 1; defects and
distortures turn them fractional, which is exactly the diagnostic power of
the method.
