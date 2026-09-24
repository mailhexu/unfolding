---
title: "TB2J magnons SrMnO3"
weight: 9
---

Downfold magnon band structures computed from TB2J exchange parameters:
the antiferromagnetic DFT cell gives a magnetic lattice larger than the
chemical primitive cell, so the magnon bands fold — unfolding assigns
each folded branch its primitive-cell momentum. The example: G-type AFM
SrMnO₃.

## The pipeline

DFT (any TB2J-supported code) → **TB2J** exchange parameters in the AFM
phase → magnon bands on the supercell BZ → **downfold** onto the
primitive-cell magnon BZ:

```text
SrMnO3 VASP/Wannier90 DFT (G-AFM)
  -> TB2J pickle: JR tensors, 10-atom sqrt2 x sqrt2 x sqrt2 cell, 2 Mn
  -> TB2J Magnon: BdG magnon bands (collinear reference, Q = 0)
  -> unfolding: weights on the 5-atom pseudo-cubic primitive cell
```

TB2J's own CLI already *folds* a primitive q-path into the supercell BZ
for plotting; this package supplies the missing unfold side.

## The magnetic cell

The SrMnO₃ example cell is the √2×√2×√2 G-AFM cell (10 atoms, 2 Mn with
moments ±2.81 μB, propagation Q=(½,½,½)). Its axes in pseudo-cubic units
are the rows of `M = [[0,1,1],[1,0,1],[1,1,0]]`, i.e. `A_afm = M · A_pc`
— the cell is twice the 5-atom pseudo-cubic primitive cell, so the
downfold target has a single Mn per cell:

```python
import numpy as np
from unfolding import unfold_tb2j

m_afm = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])   # A_afm = M @ A_pc

ax = unfold_tb2j(
    'TB2J_results',                # directory with TB2J.pickle
    m_afm,
    qpts,                          # pseudo-cubic primitive q-path
    knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'],
    xqpts=x, Xqpts=X,
)
```

The reference defaults to the collinear two-sublattice frame (Q=0,
quantization axis ẑ, moments from the pickle); spiral references are not
supported in v1.

## One command

```bash
pip install unfolding[tb2j]
unfolding-magnon TB2J_results --unfold-mat 0 1 1 1 0 1 1 1 0 \
    --kpath GXMGR --npts 200 --output magnon_unfolded.png
```

The primitive cell is derived from the TB2J cell through the unfold
matrix; the q-path special points come from that cell.

{{< figure src="/images/srmmo3_magnon_unfolded.png" title="G-AFM SrMnO3 magnons downfolded onto the pseudo-cubic cell: dark traces carry weight ≈ 1" >}}

## Reading the figure

For a pristine antiferromagnet the weights are binary: every folded
magnon branch belongs to exactly one primitive momentum. The G-AFM
dispersion is Q-periodic — ω(q) = ω(q+Q) — so the two folds sharing a
stored supercell momentum are exactly degenerate and the stored
eigenvector gauge mixes them; the degenerate-group presentation resolves
this to binary per-fold weights (the gauge-invariant content is the
group-total weight). A Goldstone mode sits at Γ (machine precision (< 1e-4 meV) here, set by
the small magnetic anisotropy in the exchange set). Defects or
non-collinear orders would produce genuinely fractional weights, exactly
as in the electronic examples.

## Reproduce this example

Download the [complete input bundle](/downloads/tb2j-magnon-srmmo3.tar.gz)
(`tb2j-magnon-srmmo3.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
