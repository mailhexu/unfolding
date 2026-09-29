---
title: "SIESTA Si:P dopant"
weight: 4
---

Unfold a substituted SIESTA supercell — one Si atom of the 8-atom
conventional-cubic cell replaced by P — onto the primitive fcc cell
band path. The substitution breaks the translation symmetry: host bands
keep weight 1 while donor-derived states appear with fractional weight.

## The bundle

Download [siesta-p-doped.tar.gz](/downloads/siesta-p-doped.tar.gz) and
unpack it:

```console
tar xf siesta-p-doped.tar.gz && cd siesta-p-doped
```

Shipped: the Si₇P supercell fixtures (`data/si_sc_p.fdf` +
`si_sc_p.HSX`), the pristine primitive cell (`data/si_prim.fdf` +
`si_prim.HSX`, POSCAR copy in `data/si_prim.vasp`), the P
pseudopotential (`pseudos/P.psml`), and `reproduce.py`. Prerequisites:
`pip install unfolding` plus `pip install HamiltonIO sisl`. No SIESTA
run is needed.

## Structure and k-path

Identical to the [pristine example](../siesta-si/): primitive 2-atom
fcc cell (a = 5.430 Å), supercell matrix
`M = [[-1,1,1],[1,-1,1],[1,1,-1]]`, path Γ-X-W-Γ-L-W-X with 300 points.

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/).
Primitive cell input: required for the `siesta` route — it supplies the relabel map and the default path cell.

## Configuration

```toml
# SIESTA Si:P dopant: one Si of the 8-atom conventional cell replaced by P.
# Run from the unpacked bundle root:  unfolding --config unfold.toml
# or with explicit flags:
#   unfolding siesta --fdf data/si_sc_p.fdf --primitive data/si_prim.vasp \
#     --unfold-mat -1 1 1 1 -1 1 1 1 -1 --special-points GXWGLX \
#     --npts 300 --method ideal --no-match-species \
#     --output si_p_doped_unfolded_cli.png
#
# match_species = false lets the P dopant map onto the host site it
# replaces (same position and orbital count): host bands keep weight 1
# while donor-derived states appear with fractional weight.
route = "siesta"

[input]
fdf = "data/si_sc_p.fdf"          # Si7P supercell run; reads si_sc_p.HSX

[structure]
primitive = "data/si_prim.vasp"   # pristine 2-atom primitive fcc cell (a = 5.430 Ang)
supercell_matrix = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]   # conventional = M @ primitive

[path]
special_points = "GXWGLX"
npts = 300

[options]
method = "ideal"
match_species = false             # substitutional dopant: ignore species when matching

[output]
output = "si_p_doped_unfolded_cli.png"
```

## Run it

The dopant needs one extra setting: atom matching must ignore species
(`match_species = false`), so the P atom maps onto the host site it
replaces (same position and orbital count). From the unpacked bundle
directory, one command with the shipped config:

```console
unfolding --config unfold.toml          # -> si_p_doped_unfolded_cli.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or with explicit flags (the same call the TOML encodes):

```console
unfolding siesta --fdf data/si_sc_p.fdf --primitive data/si_prim.vasp \
    --unfold-mat -1 1 1 1 -1 1 1 1 -1 --special-points GXWGLX \
    --npts 300 --method ideal --no-match-species \
    --output si_p_doped_unfolded_cli.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from ase.io import read
from HamiltonIO.siesta import SislParser
from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
from unfolding.mapping import RelabelMap

sc = SislParser("data/si_sc_p.fdf").get_model()
prim = SislParser("data/si_prim.fdf").get_model()
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

rm = RelabelMap.from_atoms(
    sc.atoms, prim.atoms, M,
    orb_counts_sc=[4] * 8, orb_counts_prim=[4, 4],
    match_species=False,   # the P dopant maps onto the Si site it replaces
)
unf = LCAOUnfolder(HamiltonIOModel(sc), rm)
result = unf.compute(kpts, method="ideal")
```

`reproduce.py` renders the published figure:

```console
python reproduce.py                     # -> si_p_doped_unfolded.png
```

{{< figure src="/images/si_p_doped_unfolded.png" title="Unfolded SIESTA Si$_7$P bands along Γ-X-W-Γ-L-X; blue color intensity encodes the unfolding weight, red curves are the pristine primitive-cell bands; energies in eV with zero at the Fermi level" >}}

## The fixtures

Same SIESTA setup as the pristine example (GGA-PBE, SZ PAO basis,
`MeshCutoff 100 Ry`, 2×2×2 k-grid SCF, a = 5.430 Å) with one
substitutional P. For your own doped supercell: relax the geometry as
you see fit, keep `SaveHS true` and a k-sampled SCF, and pass
`match_species=False` as above. The unperturbed host bands must stay at
weight 1 — if they do not, the dopant site is mapping to the wrong
host.
