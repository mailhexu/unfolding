---
title: "GPAW plane-wave Si and Si:P"
weight: 11
---

Unfold GPAW *pseudo* plane-wave coefficients by reciprocal cosets,
using the backend-free plane-wave engine. This does not reconstruct
PAW all-electron weights. The pristine path and the Si:P Gamma folds
illustrate respectively nearly binary and fractional pseudo weights.

## Reading a GPAW planewave restart

``HamiltonIO.gpaw.GpawPWParser`` reads a ``mode='all'`` ``.gpw``
restart (planewave mode, ``symmetry='off'``) and aggregates it into a
``GpawPWData`` object mirroring the SIESTA ``WFSXData`` surface:
fractional supercell k-points (plus their Cartesian form, 2*pi
included), eigenvalues in eV, and per-k expansion coefficients on the
stored plane-wave grids together with their integer G vectors.
GPAW normalizes wavefunctions in the PAW overlap metric. This parser
renormalizes the pseudo coefficients to `sum_G |c_G|^2 = 1`: resulting
coset fractions describe the *pseudo* wavefunctions and are not PAW
all-electron spectral weights. For defects, augmentation can change
the fractions; see the PAW method research for the missing overlap.

| Run setting | Why |
|---|---|
| ``mode='all'`` restart | Without wavefunctions there is nothing to project. |
| ``symmetry='off'`` | The reader consumes the stored k-points as written. |

Integer G vectors are recovered exactly from GPAW's
``G_plus_k_Gv = (g + k) @ (2 pi icell)`` and validated to lie inside the
cutoff sphere.

The locally generated pristine Si fixture uses an 8-atom conventional
cell (PBE, 340 eV cutoff, 24 bands) on the 300-point
Gamma-X-W-Gamma-L-W-X path in supercell coordinates;
`examples/gpaw_si/generate_fixtures.py` reproduces it.

## Run the unfolding

```python
import numpy as np
from HamiltonIO.gpaw import GpawPWParser
from unfolding.pw_unfolder import PWEigenData, PWUnfolder

data = GpawPWParser("si8_pw.gpw").read()
eigendata = PWEigenData(
    kpoints=data.kpoints,
    gvecs=data.gvecs,
    coefficients=[c[None, :, None, :] for c in data.coefficients],
    eigenvalues=data.eigenvalues[:, None, :],
)
unf = PWUnfolder(eigendata, np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]))
res = unf.compute(kpts, resolve_degenerate=1e-3)   # kpts: primitive path
```

The supercell matrix is the conventional cubic cell in primitive
lattice units (row convention), and the requested path points are
primitive-cell fractional coordinates — the unfolder maps each to its
stored supercell momentum ``K = k @ B.T`` internally. Every requested
k must be present in the stored grid; ``resolve_degenerate`` (eV)
reassigns gauge-invariant weights inside near-degenerate groups so
exact degeneracies render as clean lines.

## Pristine Si: a known answer

{{< figure src="/images/gpaw_si_pw_unfolded.png" title="GPAW planewave Si 8-atom conventional cell unfolded onto the primitive path; crimson curves: independently computed primitive-cell planewave bands" >}}

For the lowest 20 states of this pristine Si fixture, the largest
`|w(1-w)|` after resolving near-degeneracies is 3.3e-6; do not
describe raw computed weights as exact integers. The crimson overlay
is an independent primitive-cell plane-wave calculation, aligned by
the median potential-reference offset. Run
`examples/gpaw_si_pw/unfold.py` to reproduce the figure.

## Si:P at supercell Gamma

{{< figure src="/images/gpaw_si_p_pw_unfolded.png" title="GPAW plane-wave Si:P: pseudo-wavefunction weights at the four primitive momenta folding to supercell Gamma" >}}

`si7p_pw.gpw` is a real 340 eV PBE 8-atom Si7P Gamma-point
calculation. Its four primitive folds have one shared supercell energy
per band and weights summing to one within 2.5e-15; band 16 has fold
weights approximately (0.082, 0.306, 0.306, 0.306). This Gamma-only
fixture does **not** provide a continuous path. Run
`examples/gpaw_si_pw/unfold_doped.py`; regenerate input with
`examples/gpaw_si/generate_fixtures.py`.
