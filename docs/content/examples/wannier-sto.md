---
title: "Wannier90 synthetic t2g model"
weight: 8
---

Unfold a bundled synthetic Ti $t_{2g}$ Wannier90-format tight-binding
model: a 2×2×2 simple-cubic supercell (24 orbitals) onto a 3-orbital
primitive cell along Γ-X-M-Γ-R. The included `example_hr.dat` and
`example.win` are sufficient to run the package's built-in Wannier90
reader; minimulti is not required.

## The bundle

Download [wannier-sto.tar.gz](/downloads/wannier-sto.tar.gz) and unpack
it:

```console
tar xf wannier-sto.tar.gz && cd wannier-sto
```

The bundle ships `data/example_hr.dat` and `data/example.win` for a
synthetic Ti $t_{2g}$ nearest-neighbor model (3 orbitals per primitive
site, 24 in the 2×2×2 supercell, hopping 1 eV). The annotated
`inputs/seedname.win.example` shows how to prepare a real Wannier90
dataset. The figures below are generated from the **bundled** model.

## Structure and k-path

| | |
|---|---|
| supercell matrix | `scmat = diag(2,2,2)` of the primitive cell (row convention) |
| labels | one per supercell orbital: the bundled model uses d_xy/d_yz/d_zx × 8 sites (24) |
| k-path | Γ-X-M-Γ-R, 200 points, vertices in primitive fractional coordinates: (0,0,0), (.5,0,0), (.5,.5,0), (0,0,0), (.5,.5,.5) |

## Run it

From the unpacked bundle directory, one command with the shipped
config (the `cell` line pins the win unit cell of the synthetic model,
so the CLI runs as shipped — with a real Wannier90 run the cell is read
from the `.win` instead):

```console
unfolding --config unfold.toml          # -> sto_unfolded.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or with explicit flags (`--labels` takes one label per supercell
orbital, 24 here):

```console
unfolding wannier --path data --prefix example \
    --unfold-mat 2 0 0 0 2 0 0 0 2 \
    --labels d_xy d_yz d_zx d_xy d_yz d_zx d_xy d_yz d_zx d_xy d_yz d_zx \
            d_xy d_yz d_zx d_xy d_yz d_zx d_xy d_yz d_zx d_xy d_yz d_zx \
    --kpoints 0 0 0 0.5 0 0 0.5 0.5 0 0 0 0 0.5 0.5 0.5 \
    --names G X M G R --npoints 200 --output sto_unfolded.png
```

or through the Python API with explicit parameters:

```python
from unfolding.wannier_unfold import run

ax = run(
    path='data', prefix='example',     # data/example.win + data/example_hr.dat
    labels=['d_xy', 'd_yz', 'd_zx'] * 8,
    scmat=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    output_figure='sto_unfolded.png',
    kvectors=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
              [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'],
)
```

`reproduce.py` draws the bundled pristine and defect models;
`--defect` changes one site's hoppings to 30% and adds a +6 eV on-site
term.

```console
python reproduce.py --engine builtin              # pristine -> sto_unfolded.png
python reproduce.py --defect --engine builtin     # defect   -> sto_defect.png
```


{{< figure src="/images/sto_nodefect.png" title="Bundled synthetic t2g supercell unfolded onto its primitive cell along Γ-X-M-Γ-R; opacity encodes unfolded weight, gray curves are folded supercell bands; energies in eV." >}}

{{< figure src="/images/sto_defect.png" title="Bundled synthetic t2g supercell with one perturbed site, unfolded on the same path and scale; opacity encodes weight." >}}
