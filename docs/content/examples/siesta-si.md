---
title: "SIESTA Si bands"
weight: 2
---

Unfold a SIESTA supercell calculation onto the primitive-cell band path
using the stored Hamiltonian: an 8-atom conventional-cubic Si supercell
(`SaveHS true`) unfolded onto the 2-atom primitive fcc cell with the
`unfold_siesta` adapter, which parses the run through
[HamiltonIO](https://github.com/aimatores/HamiltonIO)/sisl.

## Running the example

From your SIESTA runs:

- the supercell run with the Hamiltonian saved (`SaveHS true`), producing
  `.HSX`/fdf output the adapter can parse,
- the primitive-cell structure (any ASE-readable file).

The supercell archive should come from a k-sampled SCF run: a Gamma-only
`SaveHS` collapses all supercell image shells into a single R=0 block,
which cannot be unfolded at generic momenta.

```python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.dft.kpoints import bandpath, get_special_points
from unfolding import unfold_siesta

prim_atoms = read('primitive.xsf')            # 2-atom primitive cell
points = get_special_points('fcc', prim_atoms.cell, eps=0.01)
kpts, x, X = bandpath([points[k] for k in 'GXWGLX'], prim_atoms.cell, 300)

ax = unfold_siesta(
    fdf='si_sc.fdf',                          # supercell SIESTA input
    prim_atoms=prim_atoms,
    unfold_sc_mat=np.array([[-1, 1, 1],        # 8-atom conventional cell
                            [1, -1, 1],        # = M @ primitive cell
                            [1, 1, -1]]),
    kpts=kpts, xqpts=x, Xqpts=X,
    knames=[r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L', 'X'],
)
```

`unfold_sc_mat` uses the row convention `supercell = M @ primitive`. The
adapter matches every supercell atom onto the primitive cell, computes
the weights, and plots weight-coded bands in eV.

{{< figure src="/images/si_unfolded.png" title="Unfolded SIESTA Si$_8$ bands along Γ-X-W-Γ-L-X; blue color intensity encodes the unfolding weight, red curves are the independently computed primitive-cell bands; energies in eV with zero at the Fermi level" >}}

Two weight definitions are available through
`LCAOUnfolder.compute(kpts, method=...)`: `method="ring"` (exact torus
projection, binary at momenta commensurate with the supercell torus) and
`method="ideal"` (the standard Popescu–Zunger/Lee weight for generic
off-grid momenta, used along arbitrary k-paths). For a pre-parsed
Hamiltonian, pass `model=` instead of `fdf=`; collinear spin-polarized
runs select the channel with `spin="up"` or `spin="down"`. For custom
pipelines the stages behind `unfold_siesta` are public
(`RelabelMap.from_atoms`, `LCAOUnfolder(HamiltonIOModel(model), rm)`),
and the WFSX variant of this example is described in
[SIESTA WFSX route](../siesta-wfsx/).

A complete input bundle is available as
[siesta-si.tar.gz](/downloads/siesta-si.tar.gz): input files,
pseudopotentials, the fixture data needed for the figure, a
`reproduce.py` script, and a `README.txt` with prerequisites and exact
run instructions.

## Calculation background

The committed fixtures live in `tests/data/si_example/`: SIESTA runs
with a Si pseudopotential and single-zeta PAO basis (`PAO.BasisSize SZ`),
GGA-PBE, `MeshCutoff 100 Ry`, a 2×2×2 Monkhorst-Pack grid, and lattice
constant a = 5.430 Å — the 8-atom conventional-cubic supercell
(`si_sc.fdf` + `si_sc.HSX`) and the 2-atom primitive cell
(`si_prim.fdf`). The supercell matrix is the conventional cube in
primitive-lattice units, `B = [[-1,1,1],[1,-1,1],[1,1,-1]]`; the path is
Γ-X-W-Γ-L-X with 300 points. Regenerate the figure headless with
`python docgen/fig_siesta_si.py`.
