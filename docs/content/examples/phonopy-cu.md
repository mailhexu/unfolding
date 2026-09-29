---
title: "Phonopy Cu phonons"
weight: 1
---

Unfold the phonon band structure of a 3×3×3 fcc Cu supercell, computed
with phonopy, onto the primitive fcc Brillouin zone. For a pristine
crystal every supercell phonon folds from exactly one primitive
momentum, so the weights are binary: three bold branches trace the
primitive Cu dispersion while the other 24 folded copies of the
27-atom cell stay invisible. In a defective or distorted supercell the
same plot shows fractional weights.

## The bundle

Download [phonopy-cu.tar.gz](/downloads/phonopy-cu.tar.gz) and unpack
it:

```console
tar xf phonopy-cu.tar.gz && cd phonopy-cu
```

Shipped: the phonopy run's `FORCE_CONSTANTS` (27×27) and `SPOSCAR`
(the 3×3×3 supercell, a = 3.61 Å), `unfold.toml`, and `reproduce.py`.
Prerequisites: `pip install "unfolding[phonopy]"` (phonopy, ase,
spglib). No DFT is involved.

## Structure and k-path

| | |
|---|---|
| supercell | `SPOSCAR`, the 3×3×3 cell itself; reading matrix `sc_mat = diag(1,1,1)` |
| unfolding matrix | `M = diag(3,3,3)`; the path cell is derived as `inv(M) @` SPOSCAR cell (the primitive fcc cell) |
| q-path | Γ-X-W-Γ-L, 300 points |
| units | frequencies read from phonopy in THz, plotted in cm⁻¹ |

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/).
Primitive cell input: not used by the `phonopy` route — the path cell is derived from `SPOSCAR` and the unfolding matrix.

## Configuration

```toml
# Phonopy Cu: 3x3x3 fcc supercell force constants -> primitive fcc phonons.
# Run from the unpacked bundle root:  unfolding --config unfold.toml
# or with explicit flags:
#   unfolding phonopy --force-constants FORCE_CONSTANTS --sposcar SPOSCAR \
#     --unfold-mat 3 0 0 0 3 0 0 0 3 --special-points GXWGL \
#     --npts 300 --output unfolded_band_structure.png
route = "phonopy"

[input]
force_constants = "FORCE_CONSTANTS"
sposcar = "SPOSCAR"               # the 27-atom 3x3x3 supercell itself

[structure]
supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]     # the unfolding: A_sc = M @ A_prim

[path]
special_points = "GXWGL"          # resolved on inv(M) @ SPOSCAR cell (the primitive cell)
npts = 300

[output]
output = "unfolded_band_structure.png"
```

## Run it

From the unpacked bundle directory, one command with the shipped
config:

```console
unfolding --config unfold.toml          # -> unfolded_band_structure.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or with explicit flags:

```console
unfolding phonopy --force-constants FORCE_CONSTANTS --sposcar SPOSCAR \
    --unfold-mat 3 0 0 0 3 0 0 0 3 --special-points GXWGL \
    --npts 300 --output unfolded_band_structure.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from ase.build import bulk
from ase.dft.kpoints import bandpath, get_special_points
from unfolding.phonopy_unfolder import phonopy_unfold

atoms = bulk("Cu", "fcc", a=3.61)
points = get_special_points("fcc", atoms.cell, eps=0.01)
kpts, x, X = bandpath([points[k] for k in "GXWGL"], atoms.cell, 300)

ax = phonopy_unfold(
    sc_mat=np.diag([1, 1, 1]),          # the cell stored in SPOSCAR
    unfold_sc_mat=np.diag([3, 3, 3]),   # the supercell the FC belong to
    force_constants="FORCE_CONSTANTS",
    sposcar="SPOSCAR",
    qpts=kpts, xqpts=x, Xqpts=X,
    qnames=[r"$\Gamma$", "X", "W", r"$\Gamma$", "L"],
)
ax.figure.savefig("unfolded_band_structure.png", dpi=300)
```

`reproduce.py` runs the same call and saves the same figure:

```console
python reproduce.py
```

{{< figure src="/images/phonopy_unfolded_band_structure.png" title="Unfolded Cu phonon branches along Γ-X-W-Γ-L; blue color intensity encodes the unfolding weight of each mode; frequencies in cm⁻¹" >}}

## Your own system

Point the `[input]` section (or the flags) at your phonopy run's
`FORCE_CONSTANTS`/`SPOSCAR` and set the supercell matrix you used; the
reader also accepts `phonopy.disp.yaml`/force-set style inputs through
the lower-level staging (`read_phonopy(sposcar, sc_mat,
force_constants=...)`, and `phonopy_unfold(phonon, sc_mat, qpoints,
...)` for an already-staged phonopy object). The two matrices matter:
`sc_mat` describes what is stored in `SPOSCAR` (the identity when the
SPOSCAR is the unfolded cell), `unfold_sc_mat` describes the supercell
being unfolded — primitive-frame q-points are multiplied by it before
the supercell dynamical matrix is evaluated.
