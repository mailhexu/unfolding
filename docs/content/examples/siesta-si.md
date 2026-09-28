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
| supercell | 8-atom conventional cubic cell, `A_sc = M @ A_prim` with `M = [[-1,1,1],[1,-1,1],[1,1,-1]]` |
| k-path | Γ-X-W-Γ-L-W-X, 300 points, fcc special points in primitive reciprocal fractional coordinates (Setyawan–Curtarolo) |
| weight | `method = "ideal"` — the standard Popescu–Zunger/Lee weight for generic (off-grid) momenta; `"ring"` is the exact torus projection, defined only on the supercell torus grid |

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

`unfold_sc_mat` uses the row convention `supercell = M @ primitive`.
The adapter matches every supercell atom onto the primitive cell,
computes the weights, and plots weight-coded bands in eV with zero at
the Fermi level. For a pre-parsed Hamiltonian pass `model=` instead of
`fdf=`; collinear spin-polarized runs select the channel with
`spin="up"` or `spin="down"`.

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
