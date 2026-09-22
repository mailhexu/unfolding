---
title: "ABINIT WFK Si and Si:P"
weight: 6
draft: true
---

# ABINIT WFK Si and Si:P

Unfold an ABINIT supercell calculation directly from its wavefunction file.
The planewave basis is orthonormal, so the weight is a pure reciprocal-coset
projection — no overlaps or atom maps are involved. Two examples: pristine
Si in the 8-atom conventional cell (validated against the primitive cell),
then P-doped Si in the same cell.

## Preparing the WFK

The reader accepts ABINIT 9/10 ETSF netCDF WFKs and enforces a full-storage
contract:

| ABINIT input setting | Why |
|---|---|
| `iomode 3` | Write a netCDF WFK (a plain Fortran-binary WFK is rejected with this remedy). |
| `istwfk 1` | Store the full G sphere; half-sphere storage cannot be reconstructed. |
| `prtwf 1` | Keep the raw coefficients. |
| `chkprim 0` | Needed whenever the supercell itself is non-primitive. |

A typical setup is a two-dataset input: an SCF dataset that writes the
density, then a frozen-density (`iscf -2`, `getden2 1`) dataset that samples
your k-path with `kptopt 0`, an explicit `kpt2` list, and `prtwf 1`. Two
practical points learned the hard way:

- Give the path dataset a few bands of headroom beyond the states you want
  to plot so edge-state mixing stays out of the window.
- Converge the non-SCF step properly (`nstep2` generous, `tolwfr2` tight):
  an iteration-starved run leaves scattered k-points slightly unconverged,
  which jags individual eigenvalues by tenths of an eV and renders as
  broken band lines.

## Pristine Si: a known answer

{{< figure src="/images/si8_abinit_unfolded.png" title="Si 8-atom conventional cell unfolded onto the primitive path (blue intensity = spectral weight); crimson curves: independently computed primitive-cell bands" >}}

Unfold the pristine 8-atom cell on the same Γ–X–W–Γ–L–W–X path as the
[SIESTA Si example](/examples/siesta-si/). For a pristine crystal every
state folds from a single primitive momentum, so the weights are binary
and the unfolded bands *are* the primitive band structure. The crimson
overlay is an independent primitive-cell calculation: after a single
constant potential-reference shift (the supercell run uses a Γ-only SCF
density, the primitive run a k-sampled one) the two agree to within
~0.08 eV everywhere along the path.

## Si:P: a defect in the same cell

{{< figure src="/images/si7p_abinit_unfolded.png" title="Si:P unfolded: host bands keep the dark weight; defect-hybridized states carry fractional weight and appear dimmer" >}}

Substituting one Si by P in the same 8-atom cell (12.5% concentration)
mixes the fold sectors: host bands keep weight near 1 and render as the
darkest traces, while states hybridized with impurity-scattered momenta
carry fractional weight and appear dimmer. Weights of a normalized state
over all fold sectors sum to 1.

## Run the unfolding

```python
import numpy as np
import matplotlib.pyplot as plt
from unfolding import unfold_abinit

matrix = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])   # = M @ primitive

# fcc special points in primitive reciprocal coordinates (Setyawan-
# Curtarolo), same path as the SIESTA example: Gamma-X-W-Gamma-L-W-X.
special = {'G': (0, 0, 0), 'X': (.5, 0, .5), 'W': (.5, .25, .75),
           'L': (.5, .5, .5)}
names = 'GXWGLWX'
# ... build `kpts` by interpolating between the special points, and pass
# the SAME primitive points to the non-SCF dataset as kpt2 @ matrix.T.

ax = unfold_abinit(
    'si7p_patho_DS2_WFK.nc',      # your netCDF WFK
    matrix,
    kpts,
    knames=[r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L', 'W', 'X'],
    resolve_degenerate=1e-3,       # eV; keeps degenerate branch weights 0/1
)
```

Each requested primitive k-point must be present in the WFK (the adapter
maps it to the stored supercell momentum internally: primitive
`(0, t/2, t/2)` is conventional-cell `(t, 0, 0)` here). Eigenvalues are
converted from Hartree to eV at this boundary and shifted by the WFK's
Fermi energy by default (`fermi_shift=False` for absolute energies).
`average_degenerate` (eV) optionally averages weights over near-degenerate
groups, and `resolve_degenerate` (eV) is recommended for plotting: exact
degeneracies may be stored as arbitrary unitary mixtures of their fold
sectors, which splits the per-band weights from k-point to k-point and
renders as dotted lines; resolving eigen-assigns gauge-invariant branch
weights. Collinear spin channels are selected with `spin=`.

## Weights without plotting

```python
from unfolding.abinit_unfold import read_wfk
from unfolding.pw_unfolder import PWUnfolder

data = read_wfk('si7p_patho_DS2_WFK.nc')   # -> immutable WFKData
result = PWUnfolder(data, matrix).compute(kpts)   # -> PWWeights
```

`read_wfk` rejects unreadable variants with actionable messages (the
`iomode 3` and `istwfk 1` remedies above).
