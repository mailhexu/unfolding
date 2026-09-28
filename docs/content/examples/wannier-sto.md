---
title: "Wannier90 SrTiO₃"
weight: 8
---

Unfold tight-binding Hamiltonians produced by Wannier90 — pristine and
oxygen-vacancy supercells of SrTiO₃ — onto the primitive cell with the
`unfolding.wannier_unfold` driver: the Wannier90 output is converted to a
tight-binding model (reader from the
[minimulti](https://github.com/minimulti/minimulti) package,
`pip install minimulti`), the supercell generalized eigenproblem is
solved along the path, and each supercell state is assigned its
primitive-cell spectral weight. The route is generic for any
Wannier-derived (or other) tight-binding model exposing
`.atoms`, `._orb`, and `.solve_all(k_list=..., eig_vectors=...)`.

## Running the example

You need a Wannier90 run on the supercell producing at least
`wannier90_hr.dat` (centres optional but useful). The convenience driver
reads a Wannier90 directory and plots:

```python
from unfolding.wannier_unfold import run

ax = run(
    path='data', prefix='wannier90',
    labels=['O'], scmat=np.diag([2, 2, 2]),
    output_figure='sto_unfolded.png',
    kvectors=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
              [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'],
)
```

`labels` names the orbitals of the primitive model; `scmat` is the
supercell matrix; `kvectors` are the path vertices in primitive
fractional coordinates.

Driving any tight-binding model directly:

```python
from unfolding.wannier_unfold import WannierUnfolder

u = WannierUnfolder(tbmodel, labels=labels, sc_matrix=scmat)
weights = u.unfold(kpts)                 # weight matrix along the path
ax = u.plot_unfolded_band(kvectors=..., knames=...)
```

{{< figure src="/images/sto_nodefect.png" title="Pristine SrTiO₃: supercell Wannier model unfolded onto the primitive cell along Γ-X-M-Γ-R; line opacity encodes the unfolded spectral weight, faint gray curves are the folded supercell bands; energy axis in eV, E_F = 0" >}}

{{< figure src="/images/sto_defect.png" title="SrTiO₃ with an oxygen vacancy in the supercell, unfolded with the same call and encoded as the pristine figure (opacity = unfolded spectral weight, gray curves = folded supercell bands, E_F = 0)" >}}

The committed runnable version with data lives in
`examples/wannier_STO` — see also the
[SIESTA dopant example](../siesta-p-doped/) for the DFT-side route.

## Calculation background

- Model: Wannier90 on a 20-atom SrTiO₃ supercell
  (`scmat = [[1,-1,0],[1,1,0],[0,0,2]]`, det 4, of the 5-atom
  pseudo-cubic cell); 32 orbitals = O-2p (`pz, px, py` × 12 O) + Ti-3d
  (`dz2, dxy, dyz, dx2, dxz` × 4 Ti); hoppings pruned with
  `min_hopping_norm=0.05`.
- Data (`examples/wannier_STO/`): `data_nodefect/` (pristine) and
  `data/` (one oxygen vacancy) hold the Wannier90 runs (`wannier90.eig`,
  `wannier90.win`, `wannier90.wout`).
- Figures: `test_nodefect()` / `test_defect()` in
  `examples/wannier_STO/wannier_unfold.py` write
  `STO_nodefect.png` / `STO_defect.png`
  (copies of these are `docs/static/images/sto_nodefect.png` /
  `sto_defect.png`). Requires `pip install minimulti pythtb` (the
  Wannier90 reader).

Download the [complete input bundle](/downloads/wannier-sto.tar.gz)
(`wannier-sto.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
