---
title: "OpenMX Si bands"
weight: 13
---

Unfold an [OpenMX](https://www.openmx-square.org/) supercell calculation
onto the primitive-cell band path: an 8-atom conventional-cubic Si
supercell (plus a Si:P variant) unfolded onto the 2-atom fcc primitive
cell from the binary `.scfout` file (real-space Hamiltonian and overlap,
written with `HS.fileout on`), parsed by the HamiltonIO OpenMX adapter
(pure python, no OpenMX runtime needed).

## Running the example

From your OpenMX run:

- the `.scfout` file (written with `HS.fileout on`),
- the primitive-cell structure (any ASE-readable file, or just the
  lattice vectors).

Use a k-sampled SCF (`scf.EigenvalueSolver band` plus `scf.Kgrid`): the
stored real-space tables then cover a symmetric cube of periodic images
(`atv_ijk` in OpenMX notation), which is what the unfolding sums over.

```python
import numpy as np
from ase import Atoms
from unfolding import unfold_openmx

a = 5.43
prim = Atoms('Si2',
             scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25)],
             cell=[[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]],
             pbc=True)

ax = unfold_openmx(
    scfout='si_sc.scfout',                     # 8-atom conventional cell
    prim_atoms=prim,                           # 2-atom fcc primitive cell
    unfold_sc_mat=np.array([[-1, 1, 1],        # conv cell = M @ prim
                            [1, -1, 1],
                            [1, 1, -1]]),
    kpts=kpts, knames=knames, xqpts=x, Xqpts=X,
    method='ideal',
)
```

`unfold_sc_mat` uses the row convention `supercell = M @ primitive`;
`method="ring"` selects the exact torus projection at commensurate
momenta instead of the default ideal weight. The adapter parses the
scfout through [HamiltonIO](https://github.com/aimatores/HamiltonIO)'s
OpenmxParser, matches every supercell atom onto the primitive cell,
computes the weights, and plots weight-coded bands in eV. For the doped
supercell, substitute one Si by P (same 13-orbital `P7.0-s2p2d1` basis)
and map the dopant onto the host site it replaces
(`match_species=False`), as in the
[SIESTA Si:P example](/examples/siesta-p-doped/).

{{< figure src="/images/openmx_si_unfolded.png" title="Unfolded OpenMX Si$_8$ bands along Γ-X-W-Γ-L-X; blue color intensity encodes the unfolding weight, red curves are the independently diagonalized OpenMX primitive-cell bands; energies in eV with zero at the run's Fermi level (ChemP)" >}}

{{< figure src="/images/openmx_si_p_doped.png" title="Unfolded OpenMX Si$_7$P bands along the same path, dopant mapped onto the host site (match_species=False); same encoding, energies referenced to the Si:P run's Fermi level" >}}

A runnable user-facing version of the workflow lives in
`examples/openmx_si/unfold.py`.

## Calculation background

The committed fixtures under `tests/data/si_example/` are the exact runs
used for the figures: `openmx_si_prim.scfout` (2-atom primitive cell,
4×4×4 k-grid) and `openmx_si_sc.scfout` / `openmx_si_sc_p.scfout`
(8-atom conventional cells, 2×2×2 k-grid), all GGA-PBE and spin
unpolarized with the Si7.0-s2p2d1 / P7.0-s2p2d1 basis and Si_PBE19
pseudopotentials (OpenMX 2019 data set, a = 5.43 Å). Conventions:
OpenMX builds `H(k) = sum_R exp(+2 pi i k.R) H(R)` with fractional
k-points and the integer image translations `R` from `atv_ijk`
(`EigenValue_Problem.c`), identical to HamiltonIO convention 2; scfout
energies are Hartree and the geometry Bohr, both converted on parse;
each panel is shifted so the parsed Fermi level (`ChemP`) of the run
being unfolded sits at zero. Regenerate the figures headless with
`python docgen/fig_openmx_si.py`.
