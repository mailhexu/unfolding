---
title: "SIESTA Si bands"
weight: 2
---

Unfold a SIESTA supercell calculation onto the primitive-cell band path
using the stored Hamiltonian: an 8-atom conventional-cubic Si supercell
(`SaveHS true`) unfolded onto the 2-atom primitive fcc cell. Pristine
weights are binary, so the bold branches are the primitive band
structure and the independent primitive-cell calculation (red) checks
them branch for branch.

## The bundle

Download [siesta-si.tar.gz](/downloads/siesta-si.tar.gz) and unpack it:

```console
tar xf siesta-si.tar.gz && cd siesta-si
```

Everything needed ships in the bundle: the SIESTA fixtures
(`data/si_prim.fdf` + `si_prim.HSX`, `data/si_sc.fdf` + `si_sc.HSX` —
each parser input sits next to its Hamiltonian archive), the primitive
cell as a POSCAR (`data/si_prim.vasp`), the `unfold.toml` config, the
`reproduce.py` script, and this README-adjacent note on
pseudopotentials. Prerequisites: `pip install unfolding` plus
`pip install HamiltonIO sisl` (the SIESTA parser). No SIESTA run is
needed.

## Structure and k-path

| | |
|---|---|
| primitive cell | 2-atom fcc Si, a = 5.430 Å (`data/si_prim.vasp`) |
| supercell | 8-atom conventional cubic cell, `M = [[-1,1,1],[1,-1,1],[1,1,-1]]` |
| k-path | Γ-X-W-Γ-L-W-X, 300 points |

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/).
Primitive cell input: required for the `siesta` route — it supplies the relabel map and the default path cell.

## Configuration

```toml
# SIESTA Si: 8-atom conventional-cubic supercell -> 2-atom primitive fcc cell.
# Run from the unpacked bundle root:  unfolding --config unfold.toml
# or with explicit flags:
#   unfolding siesta --fdf data/si_sc.fdf --primitive data/si_prim.vasp \
#     --unfold-mat -1 1 1 1 -1 1 1 1 -1 --special-points GXWGLX \
#     --npts 300 --method ideal --output si_unfolded_cli.png
route = "siesta"

[input]
fdf = "data/si_sc.fdf"            # supercell run; the parser reads si_sc.HSX next to it

[structure]
primitive = "data/si_prim.vasp"   # 2-atom primitive fcc cell (a = 5.430 Ang)
supercell_matrix = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]   # conventional = M @ primitive

[path]
special_points = "GXWGLX"         # letters resolved on the primitive cell (Setyawan-Curtarolo)
npts = 300

[options]
method = "ideal"                  # generic-k-path weight; "ring" is torus-grid only

[output]
output = "si_unfolded_cli.png"
```

## Run it

From the unpacked bundle directory — one command with the shipped
config:

```console
unfolding --config unfold.toml          # -> si_unfolded_cli.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or with explicit flags (the same call the TOML encodes):

```console
unfolding siesta --fdf data/si_sc.fdf --primitive data/si_prim.vasp \
    --unfold-mat -1 1 1 1 -1 1 1 1 -1 --special-points GXWGLX \
    --npts 300 --method ideal --output si_unfolded_cli.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from ase.io import read
from unfolding import unfold_siesta

ax = unfold_siesta(
    fdf="data/si_sc.fdf",               # supercell run; reads si_sc.HSX
    prim_atoms=read("data/si_prim.vasp"),
    unfold_sc_mat=np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
    kpts=kpts, knames=knames, xqpts=x, Xqpts=X,
    method="ideal",
)
```

The published figure adds the primitive-cell overlay; `reproduce.py`
builds it:

```console
python reproduce.py                     # -> si_unfolded.png
```

The adapter matches every supercell atom onto the primitive cell and
computes the weights. For a pre-parsed Hamiltonian pass `model=`
instead of `fdf=`.

{{< figure src="/images/si_unfolded.png" title="Unfolded SIESTA Si$_8$ bands along Γ-X-W-Γ-L-X; blue color intensity encodes the unfolding weight, red curves are the independently computed primitive-cell bands; energies in eV with zero at the Fermi level" >}}

## The fixtures

Both runs are GGA-PBE, single-zeta PAO basis (`PAO.BasisSize SZ`),
`MeshCutoff 100 Ry`, `SaveHS true`: the primitive cell from a 4×4×4
k-grid SCF and the 8-atom supercell from a 2×2×2 grid matched to it. A
k-sampled SCF is essential — a Gamma-only `SaveHS` run collapses all
supercell image shells into a single R=0 block, which cannot be
unfolded at generic momenta. To re-run SIESTA from scratch you need
`Si.psf` (norm-conserving, from the SIESTA distribution's
Examples/Si_Optical set); it is not bundled and never read by the
unfolding (the shipped `.HSX` carries the Hamiltonian and overlap) —
see `pseudos/` in the bundle.
