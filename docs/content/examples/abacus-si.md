---
title: "ABACUS Si bands (LCAO)"
weight: 12
---

Unfold an ABACUS LCAO supercell calculation onto the primitive-cell band
path: an 8-atom conventional-cubic Si supercell (plus a Si:P variant)
with the real-space Hamiltonian and overlap exported (`out_mat_hs2 1`
writes `data-HR-sparse_SPIN0.csr` / `data-SR-sparse_SPIN0.csr`),
unfolded onto the 2-atom fcc primitive cell through
`HamiltonIO.abacus.abacus_wrapper.AbacusParser`, exactly like the
SIESTA example.

## Running the example

- a supercell SCF run with `out_mat_hs2 1` (and `out_band 1` if you want
  ABACUS's own eigenvalues as an oracle),
- the primitive cell structure (any ASE-readable form) and the orbital
  files referenced by the STRU — ABACUS <=3.10 does not write
  `OUT.${suffix}/Orbital`, so the basis is reconstructed from the STRU.

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

The parsed model exposes the full unfolding interface: `atoms` (ASE
Atoms of the supercell), `HR`/`SR` dicts keyed by integer lattice
translations (HR in eV), and `hs_and_eigen(k)`. For the doped
supercell, replace one Si by P in the STRU and pass
`match_species=False` so the dopant folds onto the host site it
replaces.

{{< figure src="/images/abacus_si_unfolded.png" title="ABACUS Si$_8$ LCAO unfolded onto the primitive-cell path; blue color intensity encodes spectral weight, red curves are the independently computed primitive-cell bands; energies in eV with zero at the Fermi level parsed from the run's running_scf.log" >}}

{{< figure src="/images/abacus_si_p_doped.png" title="ABACUS Si$_7$P LCAO unfolded along the same path, dopant mapped onto the host site (match_species=False); same encoding, energies referenced to the Si:P run's Fermi level" >}}

`examples/abacus_si/unfold.py` regenerates both figures from the
committed fixtures, headless.

## Calculation background

The committed fixtures live in `tests/data/abacus_example/`
(`si_prim`, `si_conv`, `si7p`): ABACUS LCAO runs (`basis_type lcao`,
`esolver_type ksdft`) with PBE, `ecutwfc 100`, `symmetry 0`, a
Gamma-centered 2×2×2 k-grid, and DZP numerical orbitals
`Si_gga_10au_100Ry_2s2p1d` (13 orbitals per Si atom; the Si:P fixture
adds `P_gga_10au_100Ry_2s2p1d`). The conventional cubic cell uses
`LATTICE_CONSTANT 10.2632` bohr (a = 5.43 Å). Energies are shifted to
the Fermi level parsed from the run's `running_scf.log`
(`model.efermi`); the same shift is applied to the overlaid primitive
bands. Regenerate both figures with
`python examples/abacus_si/unfold.py [docs/static/images]`.
