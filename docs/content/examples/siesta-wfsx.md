---
title: "SIESTA WFSX route"
weight: 3
---

If the supercell run already stored its wavefunctions, unfold from
SIESTA's own eigenvectors and eigenvalues instead of diagonalizing the
Hamiltonian: an 8-atom conventional-cubic Si supercell run
(`SaveWFSX true`) unfolded onto the 2-atom primitive fcc cell through
`WFSXUnfolder`.

## Running the example

- the supercell Hamiltonian archive (`.HSX`, as in the
  [Si example](../siesta-si/)) — the weight is an orbital-overlap matrix,
  so the overlap shells are still required,
- a WFSX file written along the unfolding path: `SaveWFSX true` plus a
  `%block WaveFuncKPoints` list covering your path.

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

The eigen-solve is replaced by the stored coefficients; the spectrum is
SIESTA's own (as written by the run). WFSX energies are returned exactly
as SIESTA stores them — they are Fermi-shifted by the writing run, so
subtract the `.EIG` header value for absolute eigenvalues.

{{< figure src="/images/si_wfsx_unfolded.png" title="Unfolded SIESTA Si$_8$ bands from the stored WFSX wavefunctions along Γ-X-W-Γ-L-X; blue color intensity encodes the unfolding weight, red curves are the independently computed primitive-cell bands; energies in eV with zero at the Fermi level" >}}

A complete input bundle is available as
[siesta-wfsx.tar.gz](/downloads/siesta-wfsx.tar.gz): input files,
pseudopotentials, the fixture data needed for the figure, a
`reproduce.py` script, and a `README.txt` with prerequisites and exact
run instructions.

## Calculation background

The committed fixture is the SIESTA path run
`tests/data/si_example/si_sc_path.selected.WFSX` (plus
`si_sc_path.EIG`), written by the same 8-atom supercell setup as the
[Si example](../siesta-si/) with `SaveWFSX true` and a
`%block WaveFuncKPoints` list covering the unfolding path. The figure
evaluates the weights on a 150-point Γ-X-W-Γ-L-W-X grid; WFSX
eigenvalues are shifted by the Fermi level in the `.EIG` header. The
weight still uses the `.HSX` overlap shells; only the eigen-solve is
replaced. WFSX coefficients use SIESTA's orbital-position gauge, which
the unfolder converts internally. Regenerate the figure headless with
`python docgen/fig_siesta_wfsx.py`.
