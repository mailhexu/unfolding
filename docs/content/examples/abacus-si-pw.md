---
title: "ABACUS Si and Si:P (plane waves)"
weight: 13
---

ABACUS planewave runs export wavefunction coefficients with
`out_wfc_pw 1` (`OUT.${suffix}/WAVEFUNC${K}.txt`: the G list plus
complex coefficients for every band) and eigenvalues with
`out_band 1` (`BANDS_1.dat`, eV). This example unfolds an 8-atom
conventional-cubic Si supercell — run with `basis_type pw` — onto the
primitive-cell path with the same reciprocal-coset engine as the ABINIT
WFK example.

## What you need

- a supercell SCF run (`basis_type pw`) writing the charge
  (`out_chg 1`),
- an `nscf` run with a Line-mode KPT along the primitive-cell path,
  `init_chg file`, `symmetry 0`, `out_wfc_pw 1` and `out_band 1`.

## Unfolding the coefficients

```python
import numpy as np
from HamiltonIO.abacus.pw_wfc import AbacusPWParser
from unfolding.pw_unfolder import PWEigenData, PWUnfolder

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # supercell = B @ prim
data = AbacusPWParser('OUT.si_pw').read()           # WAVEFUNC + BANDS + kpoints
core = PWEigenData(
    kpoints=data.kpoints,                           # supercell fractional
    gvecs=data.gvecs,                               # integer Miller indices
    coefficients=[c[None, :, None, :] for c in data.coefficients],
    eigenvalues=data.eigenvalues[:, None, :],
)
res = PWUnfolder(core, B).compute(kprim, resolve_degenerate=1e-4)
```

The parser recovers integer Miller indices from the FFT-box indices
ABACUS writes (the wavefunction FFT dimensions are read from the run
log), so sector labels are exact. Coefficients are per-k arrays — no
padding across plane-wave counts.

{{< figure src="/images/abacus_si_pw_unfolded.png" title="ABACUS planewave Si$_8$ unfolded onto the primitive path; weights are 0 or 1 for the pristine cell" >}}

## Reading the figure

Every band carries weight exactly 0 or 1: each supercell state is a
folded primitive band. Path segments that lie on a BZ face (X-W here)
hold degenerate mixtures of two sectors; `resolve_degenerate`
eigen-assigns the sector-projector Gram inside each degenerate group,
restoring branch weights that are invariant under that mixing — the
same treatment the ABINIT WFK example uses.

## Si:P at supercell Gamma

{{< figure src="/images/abacus_si_p_pw_unfolded.png" title="ABACUS Si:P plane-wave weights at the four primitive folds of supercell Gamma" >}}

The real Si7P 8-atom, 10 Ry PBE Gamma run stores 20 bands in
`tests/data/abacus_example/si7p_pw/OUT.si7p_pw`. The four
primitive folds partition each band's coefficient norm with maximum
error 2.9e-7 (the text WAVEFUNC precision); band 16 has weights
(0.042, 0.319, 0.319, 0.319). This is a Gamma-only example, not a
continuous band path. Recompute the fixture by running `abacus`
inside `si7p_pw`, and render it with
`python examples/abacus_si_pw/unfold_doped.py`.

## Reproduce this example

`examples/abacus_si_pw/unfold.py` regenerates the figure from the
committed fixture (`tests/data/abacus_example/si_pw_path`). The fixture
runs at ecutwfc 10 Ry to stay committable; rerun the two ABACUS jobs
with production settings for publication-quality spectra.
