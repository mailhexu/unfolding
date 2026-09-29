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

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/).
Primitive cell input: required for the `openmx` route — it supplies the relabel map and the default path cell.

## Configuration

```toml
# OpenMX Si: 8-atom conventional-cubic supercell (committed fixture)
# unfolded onto the 2-atom fcc primitive cell along Gamma-X-W-Gamma-L-W-X.
# Companion of reproduce.py; see README.txt.
route = "openmx"

[input]
scfout = "data/openmx_si_sc.scfout"

[structure]
# 2-atom fcc primitive cell (ASE-readable); supercell = M @ primitive
primitive = "data/si_prim.vasp"
supercell_matrix = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]

[path]
# fcc special points on the primitive cell, 300 points
special_points = "GXWGLWX"
npts = 300

[options]
method = "ideal"
# efermi omitted: 0.0 means use the ChemP stored in the scfout

[output]
output = "openmx_si_unfolded.png"
```

## Running the example

Download the [openmx-si bundle](/downloads/openmx-si.tar.gz), unpack it,
and work from the unpacked directory:

```bash
tar xzf openmx-si.tar.gz && cd openmx-si
python reproduce.py                     # -> openmx_si_unfolded.png
```

The bundle ships the committed OpenMX outputs (`data/openmx_si_sc.scfout`
for the 8-atom supercell and the 2-atom primitive reference) plus the
`.dat` inputs, so the pristine figure needs no OpenMX installation —
only the `unfolding` package importable (`pip install -e <unfolding
repo checkout>`) with numpy, matplotlib, ase and HamiltonIO.

The same computation through the unified CLI:

```bash
unfolding --config unfold.toml          # the bundled config
# or, with explicit flags:
unfolding openmx --scfout data/openmx_si_sc.scfout \
  --primitive data/si_prim.vasp \
  --unfold-mat -1 1 1 1 -1 1 1 1 -1 \
  --special-points GXWGLWX --npts 300 --method ideal \
  --output openmx_si_unfolded.png
```

```python
# the same TOML through the Python entry points
from unfolding import load_config, run

run(load_config("unfold.toml"))
```

and the same computation from Python, either through the config file
(`unfolding.config.load_config` validates, `unfolding.run` executes) or
with explicit adapter parameters:

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
    scfout='data/openmx_si_sc.scfout',         # 8-atom conventional cell
    prim_atoms=prim,                           # 2-atom fcc primitive cell
    unfold_sc_mat=np.array([[-1, 1, 1],        # conv cell = M @ prim
                            [1, -1, 1],
                            [1, 1, -1]]),
    kpts=kpts, knames=knames, xqpts=x, Xqpts=X,
    method='ideal',
)
```

The doped figure (`openmx_si_p_doped.png`) needs the Si7P scfout, which
is not shipped (three scfouts exceed the 10 MB bundle cap): produce it
with OpenMX from the bundled `inputs/openmx_si_sc_p.dat` — copy your
`Si_PBE19.vps` / `P_PBE19.vps` pseudopotentials next to it, run
`openmx openmx_si_sc_p.dat`, place the resulting scfout in `data/`, then
`python reproduce.py --doped`.

{{< figure src="/images/openmx_si_unfolded.png" title="Unfolded OpenMX Si$_8$ bands along Γ-X-W-Γ-L-X; blue color intensity encodes the unfolding weight, red curves are the independently diagonalized OpenMX primitive-cell bands; energies in eV with zero at the run's Fermi level (ChemP)" >}}

{{< figure src="/images/openmx_si_p_doped.png" title="Unfolded OpenMX Si$_7$P bands along the same path, dopant mapped onto the host site (match_species=False); same encoding, energies referenced to the Si:P run's Fermi level" >}}

## Structures

- **Primitive cell**: 2-atom fcc, a = 5.43 Å —
  `cell = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]]`, Si at
  (0,0,0) and (1/4,1/4,1/4) fractional. Shipped as
  `data/si_prim.vasp` (any ASE-readable format works).
- **Supercell**: 8-atom conventional cubic cell, related by the
  supercell matrix `M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]`.
- **Basis**: Si7.0-s2p2d1 (13 orbitals per Si atom), Si_PBE19
  pseudopotentials (OpenMX 2019 data set), GGA-PBE, spin unpolarized.
- For Si:P, substitute one Si by P (same 13-orbital `P7.0-s2p2d1`
  basis) and map the dopant onto the host site it replaces
  (`match_species=false`, as in the
  [SIESTA Si:P example](/examples/siesta-p-doped/)).

## K-path

Γ–X–W–Γ–L–W–X, 300 points; the plot window is −13…8 eV.

## Parameters

| Parameter | Meaning |
|---|---|
| `scfout` | binary OpenMX output (needs `HS.fileout on`); parsed by HamiltonIO |
| `primitive` | primitive-cell structure (ASE-readable) the branches are labeled with |
| `match_species` | map supercell atoms onto same-species primitive sites (`false` for substitutional dopants) |
| `tol_r` | atom-matching tolerance in Å (default 0.04) |
| `efermi` | Fermi level in eV; the default 0.0 means "use ChemP from the scfout" |

## Calculation background

The committed fixtures are the exact runs behind the figures, produced
with OpenMX 3.9 (2019 data files), GGA-PBE, `scf.EigenvalueSolver
band`, `HS.fileout on`: a 2-atom primitive cell on a 4×4×4 k-grid and
8-atom conventional cells on 2×2×2 k-grids. A k-sampled SCF is
essential: a Γ-only run collapses all supercell image shells into a
single R = 0 block, which cannot be unfolded at generic momenta.
Conventions: OpenMX builds `H(k) = Σ_R exp(+2πi k·R) H(R)` with
fractional k-points and the integer image translations `R` from
`atv_ijk`; scfout energies are Hartree and the geometry Bohr, both
converted on parse. The supercell inputs ship in `inputs/` so every run
can be regenerated from scratch.
