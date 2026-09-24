---
title: "SIESTA WFSX route"
weight: 3
---

If the supercell run already stored its wavefunctions, you can unfold from
SIESTA's own eigenvectors and eigenvalues instead of diagonalizing the
Hamiltonian — useful for large cells where you want the exact SCF states.

## What you need

- the supercell Hamiltonian archive (`.HSX`, as in the
  [Si example](../siesta-si/)) — the weight is an orbital-overlap matrix,
  so the overlap shells are still required,
- a WFSX file written along the unfolding path: `SaveWFSX true` plus a
  `%block WaveFuncKPoints` list covering your path.

## Run the unfolding

```python
import numpy as np
from HamiltonIO.siesta.wfsx import SiestaWFSXParser
from unfolding.lcao_unfolder import HamiltonIOModel
from unfolding.mapping import RelabelMap
from unfolding.wfsx_unfolder import WFSXUnfolder

cell = np.asarray(sc_model.atoms.cell)
wfsx = SiestaWFSXParser("si_sc_path.selected.WFSX", cell=cell).read()
rm = RelabelMap.from_atoms(sc_model.atoms, prim_atoms, unfold_sc_mat)
unf = WFSXUnfolder(wfsx, HamiltonIOModel(sc_model), rm, sc_mat=unfold_sc_mat)
result = unf.compute(kpts, method="ideal")   # same path as the HSX route
```

## What changes versus the Hamiltonian route

- The eigen-solve is replaced by the stored coefficients; the spectrum is
  SIESTA's own (as written by the run).
- WFSX energies are returned exactly as SIESTA stores them — they are
  Fermi-shifted by the writing run. Subtract the `.EIG` header value if you
  need absolute eigenvalues.
- WFSX coefficients use SIESTA's orbital-position gauge; the unfolder
  converts them internally (a wrong phase convention here shows up as
  weights off by orders of magnitude, so it is pinned by tests).

{{< figure src="/images/si_wfsx_unfolded.png" title="Same Si spectrum as the Hamiltonian route, obtained from stored wavefunctions without diagonalizing H" >}}

## Reproduce this example

Download the [complete input bundle](/downloads/siesta-wfsx.tar.gz)
(`siesta-wfsx.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
