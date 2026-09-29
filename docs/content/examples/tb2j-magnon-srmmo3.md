---
title: "TB2J magnons SrMnO3"
weight: 9
---

Downfold magnon band structures computed from TB2J exchange parameters
onto the primitive magnon Brillouin zone: the antiferromagnetic DFT
cell (√2×√2×√2 G-AFM SrMnO₃, 10 atoms) is larger than the chemical
primitive cell (5-atom pseudo-cubic, one Mn), so the magnon bands fold
and unfolding assigns each folded branch its primitive-cell momentum.

```text
SrMnO3 VASP/Wannier90 DFT (G-AFM)
  -> TB2J pickle: JR tensors, 10-atom sqrt2 x sqrt2 x sqrt2 cell, 2 Mn
  -> TB2J Magnon: BdG magnon bands (collinear reference, Q = 0)
  -> unfolding: weights on the 5-atom pseudo-cubic primitive cell
```

## The bundle

Download
[tb2j-magnon-srmmo3.tar.gz](/downloads/tb2j-magnon-srmmo3.tar.gz) and
unpack it:

```console
tar xf tb2j-magnon-srmmo3.tar.gz && cd tb2j-magnon-srmmo3
```

Shipped: the TB2J exchange parameters (`data/TB2J_results/TB2J.pickle`,
from a VASP/Wannier90 G-AFM run: 10 atoms, 2 Mn at ±2.81 μB, propagation
Q=(½,½,½)), `unfold.toml`, and `reproduce.py`. Prerequisites:
`pip install "unfolding[tb2j]"` (TB2J).

## Structure and k-path

| | |
|---|---|
| unfold matrix | `M = [[0,1,1],[1,0,1],[1,1,0]]`, the 10-atom AFM cell in 5-atom pseudo-cubic units |
| primitive cell | derived as `inv(M) @` TB2J cell; the q-path special points are resolved on it |
| q-path | Γ-X-M-Γ-R, 200 points |
| reference | collinear two-sublattice frame: Q = 0, quantization axis ẑ, moments from the pickle (spiral references are not supported in v1) |

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/); this page only covers what is specific to this example.
Primitive cell input: not used by the `magnon` route — the path cell is derived from the TB2J cell and the unfolding matrix.

## Configuration

```toml
# TB2J magnons: G-AFM SrMnO3, 10-atom AFM cell -> 5-atom pseudo-cubic cell.
# Run from the unpacked bundle root:  unfolding --config unfold.toml
# or with explicit flags:
#   unfolding magnon --results data/TB2J_results \
#     --unfold-mat 0 1 1 1 0 1 1 1 0 --special-points GXMGR \
#     --npts 200 --output magnon_unfolded_cli.png
route = "magnon"

[input]
results = "data/TB2J_results"          # directory holding TB2J.pickle

[structure]
supercell_matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]   # A_afm = M @ A_pc

[path]
special_points = "GXMGR"               # resolved on inv(M) @ TB2J cell (pseudo-cubic)
npts = 200

[output]
output = "magnon_unfolded_cli.png"
```

## Run it

From the unpacked bundle directory, one command with the shipped
config:

```console
unfolding --config unfold.toml          # -> magnon_unfolded_cli.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or with explicit flags:

```console
unfolding magnon --results data/TB2J_results \
    --unfold-mat 0 1 1 1 0 1 1 1 0 --special-points GXMGR \
    --npts 200 --output magnon_unfolded_cli.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from unfolding import unfold_tb2j

m_afm = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])   # A_afm = M @ A_pc

ax = unfold_tb2j(
    "data/TB2J_results",              # directory with TB2J.pickle
    m_afm,
    qpts,                             # pseudo-cubic primitive q-path
    knames=[r"$\Gamma$", "X", "M", r"$\Gamma$", "R"],
    xqpts=x, Xqpts=X,
)
```

`reproduce.py` renders the published figure through the Python API
**and** checks the physics seals — a Goldstone mode at Γ
(< 0.05 meV) and binary unfolding weights (every weight 0 or 1 to
1e-8) for the pristine G-AFM:

```console
python reproduce.py                     # -> srmmo3_unfolded_api.png, srmmo3.png
```

{{< figure src="/images/srmmo3_magnon_unfolded.png" title="G-AFM SrMnO₃ magnons downfolded from the 10-atom AFM cell onto the 5-atom pseudo-cubic cell along Γ-X-M-Γ-R: line opacity encodes the unfolded weight per primitive momentum; energy axis in meV, dashed line at 0 meV" >}}

For your own system: run TB2J on the AFM supercell (any supported DFT
code), keep the `TB2J_results` directory, and set the unfold matrix
that relates your AFM cell to the primitive cell.
