---
title: "Phonopy Cu phonons"
weight: 1
---

Unfold the phonon band structure of a 3×3×3 fcc Cu supercell, computed with
phonopy, onto the primitive fcc Brillouin zone.

## What you need

A finished phonopy run on the supercell, producing:

- `FORCE_CONSTANTS` — the second-order force constants,
- `SPOSCAR` — the supercell structure.

## Run the unfolding

```python
import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.dft.kpoints import bandpath, get_special_points
from unfolding.phonopy_unfolder import phonopy_unfold

# 1. Primitive-cell q-path (fcc Cu, primitive fractional coordinates)
atoms = bulk('Cu', 'fcc', a=3.61)
points = get_special_points('fcc', atoms.cell, eps=0.01)
kpts, x, X = bandpath([points[k] for k in 'GXWGL'], atoms.cell, 300)

# 2. Unfold the 3x3x3 supercell calculation back onto the primitive cell
ax = phonopy_unfold(
    sc_mat=np.diag([1, 1, 1]),          # supercell phonopy used (SPOSCAR)
    unfold_sc_mat=np.diag([3, 3, 3]),   # supercell being unfolded
    force_constants='FORCE_CONSTANTS',
    sposcar='SPOSCAR',
    qpts=kpts, xqpts=x, Xqpts=X,
    qnames=[r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L'],
)
plt.savefig('unfolded_band_structure.png', dpi=300)
```

`sc_mat` describes the cell stored in `SPOSCAR` (here the identity: the
SPOSCAR itself is the unfolded cell); `unfold_sc_mat` describes the
supercell the force constants belong to. Frequencies are converted from
phonopy's THz to cm$^{-1}$.

{{< figure src="/images/phonopy_unfolded_band_structure.png" title="Unfolded Cu phonons: at each path point only the three modes folding onto that primitive momentum carry weight 1" >}}

## Reading the figure

For a pristine crystal every supercell mode folds from exactly one primitive
momentum, so weights are binary: three bold branches trace the primitive
dispersion while the other 78 folded copies stay invisible. In a defective
or distorted supercell the same plot would show fractional weights.

## Lower-level staging

`read_phonopy(sposcar, sc_mat, force_constants=...)` returns the phonopy
object and accepts `disp_yaml`/`force_sets` instead of `FORCE_CONSTANTS`;
`unf(phonon, sc_mat, qpoints, ...)` runs the unfolding step alone.
