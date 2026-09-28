---
title: "ABINIT DDB phonons"
weight: 7
---

Unfold phonons from an ABINIT DDB (derivatives database) onto a primitive
cell with the `DDB_unfolder` adapter: the route runs ABINIT's `anaddb`
through abipy to obtain the supercell eigenvectors along your path, then
computes the unfolding weights. Two examples: fcc Cu from a
conventional-cubic-cell DDB, and CaTiO₃ from its 20-atom Pnma cell onto
the 5-atom pseudo-cubic cell.

## Running the example

You need a DDB from your supercell phonon calculation,
`pip install unfolding[abipy]`, and a working `anaddb` on your `PATH`
(or configured in abipy's `manager.yml`).

Cu in the conventional cell — mind the k-path frame: ase's
`get_special_points` returns fcc special points in *primitive-cell*
fractional coordinates, while abipy reads `qptbounds` in the fractional
frame of the cell stored in the DDB (for a conventional-cubic-cell DDB:
X=(0,1,0), W=(1/2,1,0), L=(1/2,1/2,1/2)). Convert with the supercell
matrix before passing them on — feeding primitive-frame points straight
to abipy plots the wrong path:

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

{{< figure src="/images/cu_fcc_unfolded.png" title="Cu phonons unfolded from a conventional-cubic-cell DDB onto the fcc primitive cell along Γ-X-W-Γ-L: spectral-weight map on the phonon branches (degenerate-group resolved), frequency axis from anaddb" >}}

CaTiO₃ — the DDB comes from the Pnma orthorhombic ground-state cell
(20 atoms, $a \approx b \approx \sqrt{2}\,a_{pc}$, $c \approx 2 a_{pc}$,
four formula units of the 5-atom pseudo-cubic perovskite); unfolding maps
its phonons onto the pseudo-cubic cell along Γ–X–M–Γ–R (X, M, R in
pseudo-cubic fractional coordinates). The supercell matrix rows are the
Pnma axes in pseudo-cubic units ($(1,-1,0)$, $(1,1,0)$, $(0,0,2)$), i.e.
$A_{Pnma} = M \cdot A_{pc}$; `kpath_bounds` is in the fractional frame of
the DDB cell; `dipdip` toggles the dipole-dipole (LO-TO) treatment passed
to anaddb:

```python
ax = DDB_unfolder('./out.DDB',
                  sc_mat=[[1, -1, 0], [1, 1, 0], [0, 0, 2]],
                  kpath_bounds=[[0, 0, 0], [0, .5, 0], [.5, .5, 0],
                                [0, 0, 0], [.5, .5, .5]],
                  knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'],
                  dipdip=0)
```

{{< figure src="/images/catio3_unfolded.png" title="CaTiO₃ phonons unfolded from the 20-atom Pnma cell onto the pseudo-cubic cell along Γ-X-M-Γ-R: spectral-weight map on the pseudo-cubic branches, no dipole-dipole term" >}}

## Calculation background

- Code: ABINIT DDB + `anaddb` (via abipy); the unfolder uses
  gauge-robust Bloch-sum projectors with degenerate-group resolution, so
  eigenvector storage gauges (phonopy folds the path momentum out,
  anaddb keeps the full Bloch phase) do not change the weights.
- Cu: DDB `examples/Cu_fcc/out_DDB`, computed for the conventional cubic
  fcc cell (natom 4); `sc_mat = inv([[0,1,1],[1,0,1],[1,1,0]]/2)`; path
  Γ–X–W–Γ–L. Figure: `python examples/Cu_fcc/unfold.py`.
- CaTiO₃: DDB `examples/CaTiO3_unfold/out.DDB`, Pnma cell; `sc_mat =
  [[1,-1,0],[1,1,0],[0,0,2]]`, `dipdip=0`; path Γ–X–M–Γ–R. Figure:
  `python examples/CaTiO3_unfold/unfold.py`.

Download the [complete input bundle](/downloads/abinit-ddb.tar.gz)
(`abinit-ddb.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
