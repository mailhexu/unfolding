---
title: "OpenMX Si bands"
weight: 13
---

Unfold an [OpenMX](https://www.openmx-square.org/) supercell calculation
onto the primitive-cell band path. OpenMX stores the real-space
Hamiltonian and overlap in the binary `.scfout` file; the HamiltonIO
OpenMX adapter parses it directly (pure python, no OpenMX runtime
needed) and the weight computation runs on top.

## What you need

From your OpenMX run:

- the `.scfout` file (written with `HS.fileout on`),
- the primitive-cell structure (any ASE-readable file, or just the
  lattice vectors).

Use a k-sampled SCF (`scf.EigenvalueSolver band` plus `scf.Kgrid`): the
stored real-space tables then cover a symmetric cube of periodic images
(`atv_ijk` in OpenMX notation), which is what the unfolding sums over.

## One-call unfolding

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

`unfold_sc_mat` uses the row convention `supercell = M @ primitive`, and
the weights use the ideal (Popescu-Zunger / Lee) definition by default;
pass `method="ring"` for the exact torus projection at commensurate
momenta. The adapter parses the scfout through
[HamiltonIO](https://github.com/aimatores/HamiltonIO)'s OpenmxParser,
matches every supercell atom onto the primitive cell, computes the
weights, and plots weight-coded bands in eV.

{{< figure src="/images/openmx_si_unfolded.png" title="Unfolded Si8 bands with the independently diagonalized OpenMX primitive-cell bands overlaid in red" >}}

## Conventions

OpenMX builds `H(k) = sum_R exp(+2 pi i k.R) H(R)` with fractional
k-points and the integer image translations `R` from `atv_ijk`
(`EigenValue_Problem.c`); HamiltonIO convention 2 is identical. The
scfout energies are Hartree and the geometry Bohr - both are converted
on parse, and the eigenvalues are shifted so the parsed Fermi level
(`ChemP`) sits at zero: the dashed line marks E_F and 0 on the energy
axis is the Fermi level.

## Si:P doped supercell

Substituting one Si by P (same 13-orbital `P7.0-s2p2d1` basis) and
mapping the dopant onto the host site it replaces
(`match_species=False`) gives the donor picture: host bands stay near
unit weight while donor-derived states appear with reduced weight.

{{< figure src="/images/openmx_si_p_doped.png" title="Unfolded Si7P bands: donor-derived states drop below unit weight" >}}

## Reproduce this example

The committed fixtures under `tests/data/si_example/openmx_*` are the
exact runs used for the figures (Si7.0-s2p2d1 / P7.0-s2p2d1, GGA-PBE,
spin unpolarized): `openmx_si_prim` (2-atom cell, 4x4x4 grid) and
`openmx_si_sc` / `openmx_si_sc_p` (8-atom cells, 2x2x2 grid). Regenerate
the figures with `python docgen/fig_openmx_si.py`; a runnable
user-facing version of the workflow lives in
`examples/openmx_si/unfold.py`.
