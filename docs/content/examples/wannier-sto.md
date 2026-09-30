---
title: "Wannier90 SrTiO3"
weight: 8
---

Unfold a real Wannier90 tight-binding Hamiltonian of cubic SrTiO$_3$ —
the pristine $\sqrt{2}\times\sqrt{2}\times 2$ supercell (20 atoms, 56
Wannier functions) and, on the same footing, a Ti-vacancy supercell —
onto the 5-atom cubic primitive cell along Γ-X-M-Γ-R. The bundled
`wannier90_hr.dat` files are exactly as written by Wannier90; the
package's own reader (hr + win + Wannier-function centres) is all that
is needed, minimulti is not required.

## The bundle

Download [wannier-sto.tar.gz](/downloads/wannier-sto.tar.gz) and unpack
it:

```console
tar xf wannier-sto.tar.gz && cd wannier-sto
```

Shipped: the real Wannier90 outputs for the pristine
(`data/pristine/`) and Ti-vacancy (`data/ti_vacancy/`) supercells —
`wannier90.win`, `wannier90.wout` and the 38 MB `wannier90_hr.dat`
each (the compressed bundle is ~7 MB) — plus `reproduce.py`, and a tiny
synthetic t$_{2g}$ smoke fixture (`data/example_hr.dat`, documented in
the bundle README) so the route can be self-tested without the large
files. Prerequisites: `pip install unfolding` (numpy, matplotlib, ase).

## Structure and k-path

| | |
|---|---|
| primitive cell | cubic perovskite SrTiO$_3$, a = 3.9 Å, 5 atoms |
| supercell matrix | `M = [[1,-1,0],[1,1,0],[0,0,2]]` — the $\sqrt{2}\times\sqrt{2}\times 2$ tetragonal cell (20 atoms) |
| Wannier functions | 56: 12 O sites × (p_z, p_x, p_y) + 4 Ti sites × (d_z2, d_xy, d_yz, d_x2, d_xz); disentanglement with a frozen window on a 6×6×4 mp-grid |
| Ti-vacancy dataset | one Ti site vacant (dummy species `V` in the win); the vacancy-site d shell is kept, so labels and `M` are identical |
| orbital positions | final Wannier-function centres, read from the `.wout` |
| k-path | Γ-X-M-Γ-R, 200 points, vertices in primitive fractional coordinates: (0,0,0), (.5,0,0), (.5,.5,0), (0,0,0), (.5,.5,.5) |

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/). Primitive-cell input: **not used** — the Wannier90 Hamiltonian already carries its own cell (`structure.cell` or the `.win` file).


## Configuration

```toml
# Wannier90 SrTiO3: pristine 20-atom sqrt(2)xsqrt(2)x2 supercell
# (56 Wannier functions: O-2p + Ti-3d) unfolded onto the 5-atom cubic
# primitive cell.
# Run from the unpacked bundle root (requires pip install unfolding):
#   unfolding --config unfold.toml
# Ti-vacancy dataset: change path to "data/ti_vacancy" and omit the
# resolve_degenerate option to preserve raw defect weights.
route = "wannier"

[input]
path = "data/pristine"                 # wannier90 directory (win + hr + wout)
prefix = "wannier90"

[structure]
supercell_matrix = [[1, -1, 0], [1, 1, 0], [0, 0, 2]]   # A_sc = M @ A_prim, cubic a = 3.9 A
labels = [
    "pz", "px", "py", "pz", "px", "py", "pz", "px", "py",
    "pz", "px", "py", "pz", "px", "py", "pz", "px", "py",
    "pz", "px", "py", "pz", "px", "py", "pz", "px", "py",
    "pz", "px", "py", "pz", "px", "py", "pz", "px", "py",
    "dz2", "dxy", "dyz", "dx2", "dxz",
    "dz2", "dxy", "dyz", "dx2", "dxz",
    "dz2", "dxy", "dyz", "dx2", "dxz",
    "dz2", "dxy", "dyz", "dx2", "dxz",
]

[path]
kpoints = [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
names = ["Γ", "X", "M", "Γ", "R"]

[options]
npoints = 200
resolve_degenerate = 0.1  # pristine only; omit for Ti-vacancy

[output]
output = "sto_unfolded.png"
```

## Run it

From the unpacked bundle directory, one command with the shipped config
(the unit cell comes from `wannier90.win`, so nothing needs pinning):

```console
unfolding --config unfold.toml          # -> sto_unfolded.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or through the Python API with explicit parameters (`labels` takes one
label per supercell orbital, 56 here):

```python
from unfolding.wannier_unfold import run

ax = run(
    path='data/pristine', prefix='wannier90',   # win + hr + wout
    labels=['pz', 'px', 'py'] * 12 + ['dz2', 'dxy', 'dyz', 'dx2', 'dxz'] * 4,
    scmat=[[1, -1, 0], [1, 1, 0], [0, 0, 2]],
    output_figure='sto_unfolded.png',
    kvectors=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
              [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'],
    npoints=200,
    resolve_degenerate=0.1,  # pristine only; omit for a Ti-vacancy run
)
```
## Standalone real-data script

The repository example `examples/wannier_STO/wannier_unfold.py` uses only
the built-in `Wannier90Model` reader and `WannierUnfolder`; it does not
import pythtb or minimulti. Run from the repository root (with the package
dependencies installed):

```console
python examples/wannier_STO/wannier_unfold.py both --output-dir /tmp/sto-bands
```

Use `pristine` or `ti-vacancy` instead of `both` to render one dataset.
The script writes `sto_unfolded.png` and `sto_ti_vacancy.png` in the chosen
output directory.
For pristine, the script resolves translation-sector weights within 0.1 eV
energy groups to remove arbitrary eigenvector rotations at folded degeneracies.
It leaves Ti-vacancy weights unresolved, preserving defect mixing and parity
with the historical-reader reference.
`reproduce.py` draws both real datasets through the same reader (each
run parses the 38 MB hr file once, ~5 s):

```console
python reproduce.py --real pristine      # -> sto_unfolded.png
python reproduce.py --real ti_vacancy    # -> sto_ti_vacancy.png
```


{{< figure src="/images/wannier_sto_unfolded.png" title="Pristine SrTiO3: the √2×√2×2 supercell (56 Wannier functions) unfolded onto the 5-atom cubic cell along Γ-X-M-Γ-R. Every band folds from one primitive momentum, so the weights are binary: bold branches trace the O-2p valence complex and the Ti-3d conduction manifold (1.8 eV gap); folded copies stay invisible. Energies in eV." >}}

{{< figure src="/images/wannier_sto_ti_vacancy.png" title="Ti-vacancy SrTiO3 supercell unfolded on the same path and scale. The broken translation symmetry gives vacancy-derived states fractional weight — flat branches around 8.3–8.6 eV and at the conduction-band edge — while the host bands keep weight 1." >}}
