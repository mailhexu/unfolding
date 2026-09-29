---
title: "GPAW plane-wave Si and Si:P"
weight: 11
---

Unfold GPAW *pseudo* plane-wave coefficients by reciprocal cosets with
the backend-free plane-wave engine: an 8-atom conventional-cubic Si
supercell run in plane-wave mode (plus a Si:P Γ-point run) unfolded
onto the primitive-cell path through `HamiltonIO.gpaw.GpawPWParser` and
`PWUnfolder`. This does not reconstruct PAW all-electron weights.

Run parameters and shared options (`mode`, `spin`, `resolve_degenerate`,
`method`, supercell-matrix row convention, k-path coordinates,
energy-reference conventions) are explained together in the
[method, parameters, and k-path guide](/guide/method-parameters-kpaths/);
this page only covers what is specific to this example.

## Configuration

```toml
# GPAW plane-wave Si:P: the shipped Gamma-point restart unfolded into its
# four primitive folds of supercell Gamma (companion of
# `python reproduce.py --doped`; see README.txt).
route = "gpaw"

[input]
supercell = "data/si7p_pw.gpw"

[structure]
# supercell = M @ primitive (conventional cubic cell in primitive units)
supercell_matrix = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]

[path]
# the four primitive momenta folding to supercell Gamma
kpoints = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
names = ["Γ", "(0,½,½)", "(½,0,½)", "(½,½,0)"]
xticks = [0.0, 1.0, 2.0, 3.0]

[options]
mode = "pw"
resolve_degenerate = 0.001

[output]
output = "gpaw_si_p_pw_unfolded.png"
```

## Running the example

Download the [gpaw-si-pw bundle](/downloads/gpaw-si-pw.tar.gz), unpack
it, and work from the unpacked directory:

```bash
tar xzf gpaw-si-pw.tar.gz && cd gpaw-si-pw
python reproduce.py --doped             # -> gpaw_si_p_pw_unfolded.png
```

The Si7P Γ-point restart (`data/si7p_pw.gpw`, 1.1 MB) ships with the
bundle, so the doped figure runs out of the box — only the `unfolding`
package importable (`pip install -e <unfolding repo checkout>`) with
numpy, matplotlib and a recent GPAW (>= 25) is needed. The pristine
path figure additionally needs `data/si8_pw.gpw` (~290 MB, not
shipped): run the plane-wave fixture generator from the
[gpaw-si bundle](/examples/gpaw-si/) recipe (PBE, PW cutoff 340 eV, 24
bands, SCF directly on the path k-set) and drop the `.gpw` into
`data/`, then `python reproduce.py --pristine`.

The same computation through the unified CLI:

```bash
unfolding --config unfold.toml          # the bundled config (Si7P folds)
# or, with explicit flags:
unfolding gpaw --supercell data/si7p_pw.gpw \
  --unfold-mat -1 1 1 1 -1 1 1 1 -1 \
  --kpoints 0 0 0 0 0.5 0.5 0.5 0 0.5 0.5 0.5 0 \
  --names 'Γ' '(0,½,½)' '(½,0,½)' '(½,½,0)' \
  --xticks 0 1 2 3 --mode pw --resolve-degenerate 0.001 \
  --output gpaw_si_p_pw_unfolded.png
```

```python
# the same TOML through the Python entry points
from unfolding import load_config, run

run(load_config("unfold.toml"))
```

and from Python:

```python
import numpy as np
from HamiltonIO.gpaw import GpawPWParser
from unfolding.pw_unfolder import PWEigenData, PWUnfolder

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = B @ prim

data = GpawPWParser("data/si7p_pw.gpw").read()
eigendata = PWEigenData(
    kpoints=data.kpoints,
    gvecs=data.gvecs,
    coefficients=[c[None, :, None, :] for c in data.coefficients],
    eigenvalues=data.eigenvalues[:, None, :],
)
folds = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
result = PWUnfolder(eigendata, B).compute(folds, resolve_degenerate=1e-3)
```

{{< figure src="/images/gpaw_si_pw_unfolded.png" title="GPAW plane-wave Si$_8$ (8-atom conventional cell) unfolded onto the primitive Γ-X-W-Γ-L-W-X path; blue color intensity encodes the pseudo-wavefunction coset weight, crimson curves are the independently computed primitive-cell plane-wave bands; energies in eV with zero at the run's Fermi level" >}}

{{< figure src="/images/gpaw_si_p_pw_unfolded.png" title="GPAW plane-wave Si:P: pseudo-wavefunction weights at the four primitive momenta (Γ, (0,½,½), (½,0,½), (½,½,0)) folding to supercell Γ, presented deliberately as separate momentum columns — not a k-path, since the Γ-only run samples only supercell Γ; marker area and color intensity encode the weight; energies in eV relative to the run's Fermi level" >}}

## Structures

- **Primitive cell**: 2-atom fcc, a = 5.43 Å —
  `cell = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]]`.
- **Supercell**: 8-atom conventional cubic cell, supercell = **M @
  primitive** with `M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]`; the
  Si7P fixture substitutes P on the (¼,¼,¼) site.
- **Runs**: PBE plane-wave mode, PW cutoff 340 eV, `symmetry='off'`,
  `mode='all'` restarts (wavefunctions stored — without them there is
  nothing to project). The pristine fixture samples the 300-point path
  as an SCF on the fixed path k-set; the shipped Si7P fixture is a
  Γ-point-only SCF.

## K-path

Pristine figure: Γ–X–W–Γ–L–W–X, 300 points; fcc special points in
**primitive reciprocal fractional coordinates** (Setyawan–Curtarolo):
Γ (0,0,0), X (½,0,½), W (½,¼,¾), L (½,½,½). The unfolder maps each
requested primitive k to its stored supercell momentum `K = k @ M.T`
internally; every requested k must be present in the stored grid.

Doped figure: the four primitive momenta folding to supercell Γ —
Γ, (0,½,½), (½,0,½), (½,½,0) — are exactly the momenta for which
`K = k @ M.T` lands on supercell Γ. A Γ-point-only SCF samples only
that one supercell momentum, so the four weights are shown as separate
momentum columns by design — this is not a broken or sparsely sampled
k-path. Energies in eV relative to the run's Fermi level.

## Parameters

| Parameter | Meaning |
|---|---|
| Weight meaning | pseudo-wavefunction reciprocal-coset fractions: the basis is orthonormal, so pristine weights are exactly 0/1. GPAW normalizes in the PAW overlap metric; the parser renormalizes to Σ\|c\|² = 1, which leaves coset fractions unchanged. These are **not** PAW all-electron spectral weights |

## Calculation background

`GpawPWParser` reads a `mode='all'` plane-wave restart and aggregates
it into a `GpawPWData`: fractional supercell k-points (plus their
Cartesian form, 2π included), eigenvalues in eV, and per-k expansion
coefficients on the stored plane-wave grids together with their integer
G vectors (recovered exactly from GPAW's
`G_plus_k_Gv = (g + k) @ (2π icell)` and validated to lie inside the
cutoff sphere). In the pristine figure the 20 lowest bands are shown:
the four highest stored states converge loosely in the path SCF and are
not meaningful. The overlay shifts the independently computed
primitive-cell plane-wave bands by the median potential offset of the
two runs' references.
