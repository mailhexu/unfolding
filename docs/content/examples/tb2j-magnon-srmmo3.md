---
title: "TB2J magnons SrMnO3"
weight: 9
---

Downfold magnon band structures computed from TB2J exchange parameters
onto the primitive magnon Brillouin zone with the `unfold_tb2j` adapter:
the antiferromagnetic DFT cell (√2×√2×√2 G-AFM SrMnO₃, 10 atoms) is
larger than the chemical primitive cell (5-atom pseudo-cubic, one Mn),
so the magnon bands fold and unfolding assigns each folded branch its
primitive-cell momentum.

## Running the example

Pipeline: DFT (any TB2J-supported code) → TB2J exchange parameters in the
AFM phase → magnon bands on the supercell BZ → downfold onto the
primitive-cell magnon BZ:

```text
SrMnO3 VASP/Wannier90 DFT (G-AFM)
  -> TB2J pickle: JR tensors, 10-atom sqrt2 x sqrt2 x sqrt2 cell, 2 Mn
  -> TB2J Magnon: BdG magnon bands (collinear reference, Q = 0)
  -> unfolding: weights on the 5-atom pseudo-cubic primitive cell
```

The example cell's axes in pseudo-cubic units are the rows of
`M = [[0,1,1],[1,0,1],[1,1,0]]`, i.e. `A_afm = M · A_pc`:

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
supported in v1. Or, as one command (`pip install unfolding[tb2j]`; the
primitive cell is derived from the TB2J cell through the unfold matrix
and the q-path special points come from that cell):

```bash
unfolding-magnon TB2J_results --unfold-mat 0 1 1 1 0 1 1 1 0 \
    --kpath GXMGR --npts 200 --output magnon_unfolded.png
```

{{< figure src="/images/srmmo3_magnon_unfolded.png" title="G-AFM SrMnO₃ magnons downfolded from the 10-atom AFM cell onto the 5-atom pseudo-cubic cell along Γ-X-M-Γ-R: line opacity encodes the unfolded weight per primitive momentum; energy axis in meV, dashed line at 0 meV" >}}

## Calculation background

- Source: TB2J exchange parameters from a VASP/Wannier90 G-AFM DFT run
  (√2×√2×√2 cell, 10 atoms, 2 Mn at ±2.81 μB, propagation Q=(½,½,½)),
  committed at `tests/data/tb2j_srmmo3/TB2J_results` (`TB2J.pickle` with
  the JR tensors).
- Unfold matrix `M = [[0,1,1],[1,0,1],[1,1,0]]` (rows = AFM axes in
  pseudo-cubic units); magnon eigenproblems are built by TB2J's
  `Magnon` (collinear reference, Q = 0) and unfolded along the
  200-point pseudo-cubic Γ–X–M–Γ–R path.
- Figure: `python docgen/fig_srmmo3_magnon.py` writes
  `docs/static/images/srmmo3_magnon_unfolded.png`.

Download the [complete input bundle](/downloads/tb2j-magnon-srmmo3.tar.gz)
(`tb2j-magnon-srmmo3.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
