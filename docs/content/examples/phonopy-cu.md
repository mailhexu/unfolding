---
title: "Phonopy Cu phonons"
weight: 1
---

Unfold the phonon band structure of a 3×3×3 fcc Cu supercell, computed with
phonopy, onto the primitive fcc Brillouin zone with the `phonopy_unfold`
adapter.

## Running the example

A finished phonopy run on the supercell provides two inputs:

- `FORCE_CONSTANTS` — the second-order force constants,
- `SPOSCAR` — the supercell structure.

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
SPOSCAR itself is the 3×3×3 cell); `unfold_sc_mat` describes the
supercell the force constants belong to. Frequencies are converted from
phonopy's THz to cm$^{-1}$.

{{< figure src="/images/phonopy_unfolded_band_structure.png" title="Unfolded Cu phonon branches along Γ-X-W-Γ-L; blue color intensity encodes the unfolding weight of each mode; frequencies in cm⁻¹" >}}

A complete input bundle is available as
[phonopy-cu.tar.gz](/downloads/phonopy-cu.tar.gz): input files, the
fixture data needed for the figure, a `reproduce.py` script, and a
`README.txt` with prerequisites and exact run instructions.

## Calculation background

The fixture is the committed phonopy calculation in `examples/phonopy/`
(`FORCE_CONSTANTS` + `SPOSCAR` for a 3×3×3 fcc Cu supercell, a = 3.61 Å,
phonopy default settings). The unfolding path is Γ-X-W-Γ-L with 300
points, in primitive reciprocal fractional coordinates (Setyawan–Curtarolo
special points via ase). Regenerate the figure headless with
`python docgen/fig_phonopy_cu.py [out.png]`, which runs the same workflow
as the user-facing copy in `examples/phonopy/run_unfold.py`;
`read_phonopy(sposcar, sc_mat, force_constants=...)` stages the phonopy
object and accepts `disp_yaml`/`force_sets` instead of `FORCE_CONSTANTS`.
