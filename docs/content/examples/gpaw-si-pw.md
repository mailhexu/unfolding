---
title: "GPAW plane-wave Si and Si:P"
weight: 11
---

Unfold GPAW *pseudo* plane-wave coefficients by reciprocal cosets with
the backend-free plane-wave engine: an 8-atom conventional-cubic Si
supercell run in plane-wave mode (plus a Si:P Gamma-point run) unfolded
onto the primitive-cell path through ``HamiltonIO.gpaw.GpawPWParser`` and
``PWUnfolder``. This does not reconstruct PAW all-electron weights.

## Running the example

``HamiltonIO.gpaw.GpawPWParser`` reads a ``mode='all'`` ``.gpw``
restart (planewave mode, ``symmetry='off'``) and aggregates it into a
``GpawPWData`` object: fractional supercell k-points (plus their
Cartesian form, 2*pi included), eigenvalues in eV, and per-k expansion
coefficients on the stored plane-wave grids together with their integer
G vectors (recovered exactly from GPAW's
``G_plus_k_Gv = (g + k) @ (2 pi icell)`` and validated to lie inside
the cutoff sphere).

| Run setting | Why |
|---|---|
| ``mode='all'`` restart | Without wavefunctions there is nothing to project. |
| ``symmetry='off'`` | The reader consumes the stored k-points as written. |

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
exact degeneracies render as clean lines. GPAW normalizes wavefunctions
in the PAW overlap metric; the parser renormalizes the pseudo
coefficients to `sum_G |c_G|^2 = 1`, so resulting coset fractions
describe the *pseudo* wavefunctions and are not PAW all-electron
spectral weights.

{{< figure src="/images/gpaw_si_pw_unfolded.png" title="GPAW plane-wave Si$_8$ (8-atom conventional cell) unfolded onto the primitive Γ-X-W-Γ-L-W-X path; blue color intensity encodes the pseudo-wavefunction coset weight, crimson curves are the independently computed primitive-cell plane-wave bands; energies in eV with zero at the run's Fermi level" >}}

For the doped fixture, `si7p_pw.gpw` is a Gamma-point-only run: its
four primitive folds are plotted as separate columns (marker area and
color intensity encode the weight), not as a continuous path.

{{< figure src="/images/gpaw_si_p_pw_unfolded.png" title="GPAW plane-wave Si:P: pseudo-wavefunction weights at the four primitive momenta (Γ, (0,½,½), (½,0,½), (½,½,0)) folding to supercell Gamma; marker area and color intensity encode the weight; energies in eV relative to the run's Fermi level" >}}

Reproduce the figures with ``python examples/gpaw_si_pw/unfold.py`` and
``python examples/gpaw_si_pw/unfold_doped.py``.

## Calculation background

The committed fixtures live in `tests/data/gpaw_example/`:
`si8_pw.gpw` is an 8-atom conventional-cell plane-wave run (PBE,
340 eV cutoff, 24 bands, ``mode='all'``, ``symmetry='off'``) with the
nscf path sampled on the 300-point Γ-X-W-Γ-L-W-X primitive path in
supercell coordinates; `si7p_pw.gpw` is a real 340 eV PBE 8-atom Si7P
Gamma-point run. Both come from
``examples/gpaw_si/generate_fixtures.py`` (gpaw >= 26 in the mydev
environment; the planewave runs are skipped when the committed fixtures
are staged into the workdir). Energies are referenced to each run's own
Fermi level.
