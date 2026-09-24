---
title: "SIESTA Si bands"
weight: 2
---

Unfold a SIESTA supercell calculation onto the primitive-cell band path
using the stored Hamiltonian. The example uses an 8-atom conventional-cubic
Si supercell unfolded onto the 2-atom primitive fcc cell.

## What you need

From your SIESTA runs:

- the supercell run with the Hamiltonian saved (`SaveHS true`), producing
  `.HSX`/fdf output the adapter can parse,
- the primitive-cell structure (any ASE-readable file).

The supercell archive should come from a k-sampled SCF run: a Gamma-only
`SaveHS` collapses all supercell image shells into a single R=0 block,
which cannot be unfolded at generic momenta.

## One-call unfolding

```python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.dft.kpoints import bandpath, get_special_points
from unfolding import unfold_siesta

prim_atoms = read('primitive.xsf')            # 2-atom primitive cell
points = get_special_points('fcc', prim_atoms.cell, eps=0.01)
kpts, x, X = bandpath([points[k] for k in 'GXWGLX'], prim_atoms.cell, 300)

ax = unfold_siesta(
    fdf='si_sc.fdf',                          # supercell SIESTA input
    prim_atoms=prim_atoms,
    unfold_sc_mat=np.array([[-1, 1, 1],        # 8-atom conventional cell
                            [1, -1, 1],        # = M @ primitive cell
                            [1, 1, -1]]),
    kpts=kpts, xqpts=x, Xqpts=X,
    knames=[r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L', 'X'],
)
```

`unfold_sc_mat` uses the row convention `supercell = M @ primitive`. The
adapter parses the fdf through
[HamiltonIO](https://github.com/aimatores/HamiltonIO)/sisl, matches every
supercell atom onto the primitive cell, computes the weights, and plots
weight-coded bands in eV.

{{< figure src="/images/si_unfolded.png" title="Unfolded Si bands with the independently computed primitive-cell bands overlaid in red" >}}

## Reading the figure

The red overlay is the primitive-cell band structure diagonalized
independently. Every weight-1 unfolded branch lies on a primitive band,
which is the practical validation of the unfolding.

## Weights at generic momenta

Two weight definitions are available through
`LCAOUnfolder.compute(kpts, method=...)`:

- `method="ring"` — exact torus projection; binary and Parseval-exact at
  momenta commensurate with the supercell torus,
- `method="ideal"` — the standard Popescu–Zunger/Lee weight for generic
  (off-grid) momenta; use it along arbitrary k-paths.

For a pre-parsed Hamiltonian, pass `model=` instead of `fdf=`; collinear
spin-polarized runs select the channel with `spin="up"` or `spin="down"`.

## Building blocks

For custom pipelines, the stages behind `unfold_siesta` are public:

```python
from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
from unfolding.mapping import RelabelMap

rm = RelabelMap.from_atoms(sc_atoms, prim_atoms, unfold_sc_mat)
unf = LCAOUnfolder(HamiltonIOModel(sc_model), rm)
result = unf.compute(kpts, method="ideal")   # -> LCAOWeights
```

## Reproduce this example

Download the [complete input bundle](/downloads/siesta-si.tar.gz)
(`siesta-si.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
