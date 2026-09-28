---
title: "SIESTA spinors"
weight: 5
---

Non-collinear (`nspin=4`, spinor) SIESTA supercell runs unfold through
the same machinery as the collinear example — the parse → relabel →
weight chain on the doubled basis (each PAO times two spin
components). Here: the 8-atom conventional-cubic Si supercell run
non-collinear, unfolded onto the 2-atom primitive fcc cell. The scalar
Si pseudopotential has no SOC channels, so the spinor bands equal the
scalar bands with Kramers degeneracy — a clean end-to-end seal of the
spinor machinery, not spinor physics.

## The bundle

Download [siesta-spinor.tar.gz](/downloads/siesta-spinor.tar.gz) and
unpack it:

```console
tar xf siesta-spinor.tar.gz && cd siesta-spinor
```

Shipped: the non-collinear fixtures (`data/si_sc_pso.fdf` +
`si_sc_pso.HSX` supercell, `data/si_prim_pso.fdf` +
`si_prim_pso.HSX` primitive, POSCAR copy in
`data/si_prim_pso.vasp`), `unfold.toml`, and `reproduce.py`.
Prerequisites: `pip install unfolding` plus `pip install HamiltonIO
sisl`. No SIESTA run is needed.

## Structure and k-path

| | |
|---|---|
| primitive cell | 2-atom fcc Si, a = 5.430 Å |
| supercell | 8-atom conventional cubic cell, `M = [[-1,1,1],[1,-1,1],[1,1,-1]]` (row convention) |
| orbitals | 8 spinor orbitals per atom (4 PAO × 2 spin components); derived automatically from the parsed model |
| k-path | Γ-X-W-Γ-L-W-X, 300 points, primitive reciprocal coordinates |
| weight | `method = "ideal"` |

## Run it

From the unpacked bundle directory, one command with the shipped
config:

```console
unfolding --config unfold.toml          # -> si_spinor_unfolded_cli.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or with explicit flags:

```console
unfolding siesta --fdf data/si_sc_pso.fdf --primitive data/si_prim_pso.vasp \
    --unfold-mat -1 1 1 1 -1 1 1 1 -1 --special-points GXWGLX \
    --npts 300 --method ideal --output si_spinor_unfolded_cli.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from ase.io import read
from unfolding import unfold_siesta

ax = unfold_siesta(
    fdf="data/si_sc_pso.fdf",           # non-collinear run; reads si_sc_pso.HSX
    prim_atoms=read("data/si_prim_pso.vasp"),
    unfold_sc_mat=np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
    kpts=kpts, knames=knames, xqpts=x, Xqpts=X,
    method="ideal",
)
```

Nothing spinor-specific needs passing: the doubled orbital counts are
read from the parsed model, and the same weight pipeline applies to the
64-orbital supercell basis. The published figure adds the primitive
spinor-cell overlay:

```console
python reproduce.py                     # -> si_spinor_unfolded.png
```

{{< figure src="/images/si_spinor_unfolded.png" title="Unfolded SIESTA spinor (nspin=4) Si$_8$ bands along Γ-X-W-Γ-L-X; blue color intensity encodes the unfolding weight, red curves are the independently computed primitive-cell spinor bands; energies in eV with zero at the Fermi level" >}}

## The fixtures

The same 8-atom Si setup as the collinear example run non-collinear
(`SpinPolarized true` + `Spin.Orbit true`, `nspin=4`) with a scalar Si
pseudopotential, so spin-orbit coupling is off. Sanity check for your
own spinor runs: every spinor band pair should be Kramers-degenerate
when SOC is off, and the weight sums over exactly-degenerate groups
should be integers. Spinor eigenvalues are Fermi-shifted by the writing
run, as usual in SIESTA.
