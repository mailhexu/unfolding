---
title: "SIESTA WFSX route"
weight: 3
---

If the supercell run already stored its wavefunctions, unfold from
SIESTA's own eigenvectors and eigenvalues instead of diagonalizing the
Hamiltonian: an 8-atom conventional-cubic Si supercell run
(`SaveWFSX true`) unfolded onto the 2-atom primitive fcc cell. Same
cell, same path, same weights as the
[Hamiltonian route](../siesta-si/) — only the eigen-solve is replaced;
the weight still uses the `.HSX` overlap shells.

## The bundle

Download [siesta-wfsx.tar.gz](/downloads/siesta-wfsx.tar.gz) and unpack
it:

```console
tar xf siesta-wfsx.tar.gz && cd siesta-wfsx
```

Shipped: the WFSX path run (`data/si_sc_path.selected.WFSX` plus its
`si_sc_path.EIG`), the Hamiltonian fixtures for the overlap shells and
the primitive reference (`data/si_prim.*`, `data/si_sc.*`), the
primitive POSCAR, the path-run deck (`inputs/si_sc_path.fdf`, with
`SaveWFSX true` and a `%block WaveFuncKPoints` list covering the path),
and `reproduce.py`. Prerequisites: `pip install unfolding` plus
`pip install HamiltonIO sisl` (needs HamiltonIO ≥ 0.3.6 for
`SiestaWFSXParser`). No SIESTA run is needed.

## Structure and k-path

| | |
|---|---|
| primitive cell | 2-atom fcc Si, a = 5.430 Å |
| supercell | 8-atom conventional cubic cell, `M = [[-1,1,1],[1,-1,1],[1,1,-1]]` (row convention) |
| k-path | Γ-X-W-Γ-L-W-X, **157 stored k-points** (nominal segment-density parameter 150); the WFSX matches only its written `%block WaveFuncKPoints` list, so `unfold.toml` carries those coordinates explicitly |
| weight | `"ideal"`, as in the HSX route |

## Run it

From the unpacked bundle directory, one command with the shipped
config:

```console
unfolding --config unfold.toml          # -> si_wfsx_unfolded_cli.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

The config carries the **exact grid SIESTA stored wavefunctions on**
(the segment-proportional Γ-X-W-Γ-L-W-X list of the run deck's
`%block WaveFuncKPoints`, 150 points nominal → 157 with repeated
junctions): the unfolder matches requested momenta to the stored
entries at 1e-6 tolerance, so a regenerated special-points path would
not hit the stored points. With explicit flags (kpoint list as in the
TOML):

```console
unfolding siesta-wfsx --wfsx data/si_sc_path.selected.WFSX \
    --hs-fdf data/si_sc.fdf --primitive data/si_prim.fdf \
    --unfold-mat -1 1 1 1 -1 1 1 1 -1 --kpoints ... \
    --names G X W G L W X --method ideal \
    --output si_wfsx_unfolded_cli.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from HamiltonIO.siesta import SislParser
from HamiltonIO.siesta.wfsx import SiestaWFSXParser
from unfolding.lcao_unfolder import HamiltonIOModel
from unfolding.mapping import RelabelMap
from unfolding.wfsx_unfolder import WFSXUnfolder

sc_model = SislParser("data/si_sc.fdf").get_model()
prim_model = SislParser("data/si_prim.fdf").get_model()

cell = np.asarray(sc_model.atoms.cell)
wfsx = SiestaWFSXParser("data/si_sc_path.selected.WFSX", cell=cell).read()
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
rm = RelabelMap.from_atoms(sc_model.atoms, prim_model.atoms, M)
unf = WFSXUnfolder(wfsx, HamiltonIOModel(sc_model), rm, sc_mat=M)
result = unf.compute(kpts, method="ideal")   # same path as the HSX route
```

The spectrum is SIESTA's own (as written by the run). WFSX energies are
returned exactly as SIESTA stores them — Fermi-shifted by the writing
run — so subtract the `.EIG` header value for absolute eigenvalues; the
route does this by default, and `reproduce.py` does it for the
published figure with the primitive-cell bands overlaid:

```console
python reproduce.py                     # -> si_wfsx_unfolded.png
```

{{< figure src="/images/si_wfsx_unfolded.png" title="Unfolded SIESTA Si$_8$ bands from stored WFSX wavefunctions along Γ-X-W-Γ-L-W-X; blue intensity encodes weight, red curves are primitive-cell bands; energies relative to E_F." >}}

## The fixtures

Same SIESTA setup as the pristine example (GGA-PBE, SZ PAO basis,
`MeshCutoff 100 Ry`, `SaveHS true`) plus the path run with
`SaveWFSX true`. WFSX coefficients use SIESTA's orbital-position gauge,
which the unfolder converts internally. For your own system: run the
supercell with `SaveHS true` (for the overlap shells) and a second run
with `SaveWFSX true` writing wavefunctions along your path.
