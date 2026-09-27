---
title: "PAW plane-wave unfolding: ABINIT and VASP"
weight: 14
---

A PAW wavefunction is normalized in the overlap metric, not by the sum
of its pseudo plane-wave coefficient norms. GPAW's own `gpaw/unfold.py`
computes the **pseudo-only** reciprocal-coset ratio; it does not include
PAW augmentation. `unfolding.abinit_paw` and `unfolding.vasp_paw` evaluate
physical overlaps with the PAW metric
`S = I + Σ_aij |p_ai⟩ ΔS_aij ⟨p_aj|`. Only overlap augmentation `ΔS`
enters the spectral projection, not the nonlocal Hamiltonian `D`.

Both routes embed primitive reference wavefunctions into the supercell G
basis, compute `A = B† S C` and `G = B† S B`, then obtain each weight
from the diagonal of `A† G⁻¹ A`. The finite primitive reference band bank
is not complete in general: the sum across folds can differ from one for
a doped system, and close degeneracies require group-projector
eigenvalues rather than arbitrary per-band gauge assignments.

## ABINIT JTH PAW: real Si and Si:P

`HamiltonIO.abinit.read_paw_wfk` reads ABINIT 9/10 netCDF WFKs and JTH PAW
XML datasets (via `pypao`); the WFK stores pseudo coefficients but **not**
projector overlaps. `unfolding.abinit_paw.unfold_abinit_paw` evaluates the
reciprocal projectors (via `abinao`), atom-site overlap correction, and
reference-band weights. The ordinary `unfold_abinit` rejects PAW WFKs
rather than silently treating them as norm-conserving.

The committed inputs and small WFKs in `tests/data/abinit_paw` reproduce
8-atom pristine Si, 8-atom Si7P, and a 2-atom primitive reference at the
four momenta folding to supercell Gamma. Run ABINIT on each `.abi` input
from that directory to regenerate; `iomode 3`, `istwfk 1`, `ecut 10 Ha`,
`pawecutdg 20 Ha` and the bundled Si/P JTH XMLs are required.

```python
from pathlib import Path
from unfolding.abinit_paw import unfold_abinit_paw
base = Path('tests/data/abinit_paw')
xml = {symbol: base / f'{symbol}.xml' for symbol in ('Si', 'P')}
M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
result = unfold_abinit_paw(base/'si7p_gammao_WFK.nc',
                           base/'si_primitive_foldso_WFK.nc', xml, M)
```

The real PAW Si8 WFK has maximum `|⟨ψ̃|S|ψ̃⟩−1|` of 5.8×10⁻⁷. Its raw
fold weights sum to one within 2×10⁻⁶ per band; after resolving
near-degenerate groups (`resolve_degenerate=1e-5` Hartree), branch weights
are within 2×10⁻⁶ of zero or one. For Si7P, the maximum PAW-minus-pseudo
weight difference is 0.015, and the PAW normalization error is 1.4×10⁻⁶.

{{< figure src="/images/abinit_paw_si7p.png" title="ABINIT Si:P at the four primitive folds of supercell Gamma: PAW-metric reference weights (left) and pseudo-only coset fractions (right). Marker area encodes weight." >}}

Run `python examples/abinit_paw/unfold.py` to redraw the figure. A
Gamma-only calculation does not provide a continuous k-path.

## VASP PAW: licensed input stays private

`HamiltonIO.vasp.read_wavecar` parses standard full-sphere scalar
WAVECARs through pymatgen; `read_potcar_paw` reads the matching POTCAR's
reciprocal projector tables and AE/pseudo partial waves. The VASP
license prohibits redistributing POTCAR here. Run
`unfolding.vasp_paw.unfold_vasp_paw((WAVECAR, POSCAR),
(primitive_WAVECAR, primitive_POSCAR), POTCAR, M)` with matching datasets.
Gamma-half and noncollinear WAVECAR layouts are deliberately unsupported.

The private bcc-Fe primitive fixture validates the actual PAW metric:
its first-eight-band pseudo coefficient norms range from 0.325 to 1.099,
while the PAW S-norm residual is below 8×10⁻⁶ at the two sampled k-points.
An 11-fold *constructed* Fe supercell embedding of a real primitive
wavefunction verifies the non-diagonal lattice reciprocal mapping and
restores its PAW norm within 2×10⁻⁸; other folds carry <10⁻⁸ weight.
That constructed embedding is not an independent VASP supercell run.

To exercise the licensed fixture locally, set `UNFOLDING_VASP_FE_SEED` to
a private directory with `WAVECAR`, `POSCAR`, `POTCAR`, then run
`python -m pytest tests/test_vasp_paw.py`. No licensed bytes are stored
in this repository.
