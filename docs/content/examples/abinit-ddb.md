---
title: "ABINIT DDB phonons"
weight: 7
---

Unfold phonons from an ABINIT DDB (derivatives database) — the route runs
ABINIT's `anaddb` through abipy to obtain the supercell eigenvectors along
your path, then computes the unfolding weights.

## What you need

- a DDB from your supercell phonon calculation,
- `pip install unfolding[abipy]` and a working `anaddb` on your `PATH`
  (or configured in abipy's `manager.yml`).

## Cu in the conventional cell

The subtlety here is the **k-path frame**. ase's `get_special_points`
returns fcc special points in *primitive-cell* fractional coordinates
(X=(1/2,0,1/2), W=(1/2,1/4,3/4), L=(1/2,1/2,1/2)), while abipy reads
`qptbounds` in the fractional frame of the cell stored in the DDB — for a
conventional-cubic-cell DDB that is the conventional basis
(X=(0,1,0), W=(1/2,1,0), L=(1/2,1/2,1/2)). Convert with the supercell
matrix before passing them on:

```python
import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.dft.kpoints import get_special_points
from unfolding.DDB_unfolder import DDB_unfolder

# Conventional cell = sc_mat @ primitive cell
sc_mat = np.linalg.inv(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0)

atoms = bulk('Cu', 'fcc')
points = get_special_points(atoms.cell, eps=0.01)     # primitive frame!
knames = [r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L']
kpath_prim = [points[k] for k in 'GXWGL']
kpath_bounds = [np.dot(k, sc_mat) for k in kpath_prim]  # -> DDB frame

ax = DDB_unfolder('./out_DDB', sc_mat=sc_mat,
                  kpath_bounds=kpath_bounds, knames=knames)
```

Skipping the `k_prim @ sc_mat` multiplication silently plots the wrong
path — abipy interprets the raw numbers in the DDB's own frame.

{{< figure src="/images/cu_fcc_unfolded.png" title="Cu phonons from a conventional-cell DDB: pristine weights are binary (0 or 1)" >}}

For a pristine crystal the weights are exactly 0 or 1: each supercell mode
folds from a single primitive momentum. The plot resolves the character-sum
gauge of the DDB eigenvectors and degenerate crossings automatically.

## CaTiO₃

CaTiO₃'s ground state is the **Pnma** orthorhombic perovskite (a⁻b⁺a⁻
octahedral tilts): a 20-atom cell with $a \approx b \approx \sqrt{2}\,a_{pc}$,
$c \approx 2 a_{pc}$, four formula units of the 5-atom **pseudo-cubic**
perovskite. The DDB comes from that Pnma cell; unfolding maps its phonons
back onto the pseudo-cubic cell along the simple-cubic path Γ–X–M–Γ–R
(X, M, R in pseudo-cubic fractional coordinates):

```python
ax = DDB_unfolder('./out.DDB',
                  sc_mat=[[1, -1, 0], [1, 1, 0], [0, 0, 2]],
                  kpath_bounds=[[0, 0, 0], [0, .5, 0], [.5, .5, 0],
                                [0, 0, 0], [.5, .5, .5]],
                  knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'],
                  dipdip=0)
```

The supercell matrix rows are the Pnma axes in pseudo-cubic units
($(1,-1,0)$, $(1,1,0)$, $(0,0,2)$), i.e. $A_{Pnma} = M \cdot A_{pc}$.
`kpath_bounds` is in the fractional frame of the DDB cell; `dipdip`
toggles the dipole-dipole (LO-TO) treatment passed to anaddb.

Unlike pristine Cu, the weights are not binary: the anti-phase and
in-phase octahedral tilts mix the pseudo-cubic fold sectors, so modes
carry genuine fractional pseudo-cubic character — exactly the physics
the unfolding is meant to expose.

{{< figure src="/images/catio3_unfolded.png" title="CaTiO₃ Pnma phonons unfolded onto the pseudo-cubic cell: tilt-mixed modes carry fractional weight" >}}

## Weight conventions

Phonon eigenvectors come in different storage gauges (phonopy folds the
path momentum out; anaddb keeps the full Bloch phase, and real dynamical
matrices on mirror-symmetric paths return cosine mixtures of degenerate
sectors). The unfolder uses gauge-robust Bloch-sum projectors with
degenerate-group resolution, so pristine weights stay binary in either
gauge.
