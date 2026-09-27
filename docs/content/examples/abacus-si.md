---
title: "ABACUS Si bands (LCAO)"
weight: 12
---

ABACUS LCAO runs can export the real-space Hamiltonian and overlap
(`out_mat_hs2 1` writes `data-HR-sparse_SPIN0.csr` / `data-SR-sparse_SPIN0.csr`).
This example unfolds an 8-atom conventional-cubic Si supercell (DZP
2s2p1d orbitals, 2x2x2 k-grid) onto the 2-atom fcc primitive cell,
exactly like the SIESTA example.

## What you need

- a supercell SCF run with `out_mat_hs2 1` (and `out_band 1` if you want
  ABACUS's own eigenvalues as an oracle),
- the primitive cell structure (any ASE-readable form) and the orbital
  files referenced by the STRU — ABACUS <=3.10 does not write
  `OUT.${suffix}/Orbital`, so the basis is reconstructed from the STRU.

## One-call unfolding

```python
import numpy as np
from HamiltonIO.abacus.abacus_wrapper import AbacusParser
from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
from unfolding.mapping import RelabelMap

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])   # conv = B @ prim
prim = AbacusParser(outpath='si_prim/OUT.si_prim').get_models()
sc = AbacusParser(outpath='si_conv/OUT.si_conv').get_models()

rm = RelabelMap.from_atoms(sc.atoms, prim.atoms, B,
                           orb_counts_sc=[13]*8, orb_counts_prim=[13, 13])
unf = LCAOUnfolder(HamiltonIOModel(sc), rm)
res = unf.compute(kpts, method="ideal")
```

`unfold_sc_mat` uses the row convention `supercell = M @ primitive`.
The parsed model exposes the full unfolding interface: `atoms` (ASE
Atoms of the supercell), `HR`/`SR` dicts keyed by integer lattice
translations (HR in eV), and `hs_and_eigen(k)`.

{{< figure src="/images/abacus_si_unfolded.png" title="ABACUS Si$_8$ LCAO unfolded onto the primitive-cell path with primitive bands overlaid in red" >}}

## Reading the figure

Energies are in eV with zero at the Fermi level parsed from the run's
`running_scf.log` (`model.efermi`; the same shift is applied to the
overlaid primitive bands). Host bands carry weight near 1 and lie on the
overlaid primitive-cell
band structure. With the 10 au (rc-limited) orbital range, ABACUS stores
the R shells it needs and no more; the ideal weight at generic momenta
therefore carries a small data-limit smear on this fixture — the SIESTA
example shows the exact 0/1 limit reached with a full BvK shell set.

## Si:P substitution

Replace one Si by P in the STRU and pass `match_species=False` so the
dopant folds onto the host site it replaces:

{{< figure src="/images/abacus_si_p_doped.png" title="Si$_7$P: host states keep high weight while impurity-derived states spread across fold sectors" >}}

## Reproduce this example

`examples/abacus_si/unfold.py` regenerates both figures from the
committed fixtures (`tests/data/abacus_example`), headless.
