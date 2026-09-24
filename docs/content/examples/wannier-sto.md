---
title: "Wannier90 SrTiO₃"
weight: 8
---

Unfold tight-binding Hamiltonians produced by Wannier90 — pristine and
oxygen-vacancy supercells of SrTiO₃. The route is generic for any
Wannier-derived (or other) tight-binding model that can solve the
supercell generalized eigenproblem.

## What you need

- a Wannier90 run on the supercell producing at least `wannier90_hr.dat`
  (centres optional but useful),
- the reader from the
  [minimulti](https://github.com/minimulti/minimulti) package
  (`pip install minimulti`), which converts the Wannier output into a
  tight-binding model in the convention below.

## Run the unfolding

The convenience driver reads a Wannier90 directory and plots:

```python
from unfolding.wannier_unfold import run

ax = run(
    path='data', prefix='wannier90',
    labels=['O'], scmat=np.diag([2, 2, 2]),
    output_figure='sto_unfolded.png',
    kvectors=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
              [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'],
)
```

`labels` names the orbitals of the primitive model; `scmat` is the
supercell matrix; `kvectors` are the path vertices in primitive fractional
coordinates.

## Driving any tight-binding model directly

`WannierUnfolder` only needs a model object exposing `.atoms`,
`._orb` (scaled orbital positions), and
`.solve_all(k_list=..., eig_vectors=...)` returning supercell eigenpairs:

```python
from unfolding.wannier_unfold import WannierUnfolder

u = WannierUnfolder(tbmodel, labels=labels, sc_matrix=scmat)
weights = u.unfold(kpts)                 # weight matrix along the path
ax = u.plot_unfolded_band(kvectors=..., knames=...)
```

{{< figure src="/images/sto_nodefect.png" title="Pristine SrTiO₃ unfolded from a Wannier90 supercell model" >}}

## Defect supercell

With an oxygen vacancy (or any defect) in the supercell, the same call
shows fractional weights on defect-derived bands while the host bands
remain at weight 1 — the electronic analogue of the
[SIESTA dopant example](../siesta-p-doped/).

{{< figure src="/images/sto_defect.png" title="SrTiO₃ with an oxygen vacancy: defect bands appear at reduced weight" >}}

## Reproduce this example

Download the [complete input bundle](/downloads/wannier-sto.tar.gz)
(`wannier-sto.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
