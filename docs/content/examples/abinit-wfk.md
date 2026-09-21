---
title: "ABINIT WFK Si:P"
weight: 6
---

# ABINIT WFK Si:P

Unfold an ABINIT supercell calculation directly from its wavefunction file.
The planewave basis is orthonormal, so the weight is a pure reciprocal-coset
projection — no overlaps or atom maps are involved. The example: P-doped Si
in the 8-atom conventional cell, unfolded onto the 1-atom primitive fcc
cell along Γ–X.

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
your k-path with `kptopt 0`, an explicit `kpt2` list, and `prtwf 1`. Give
the path dataset a few bands of headroom beyond the states you want to plot
so edge-state mixing stays out of the window.

## Run the unfolding

```python
import numpy as np
import matplotlib.pyplot as plt
from unfolding import unfold_abinit

matrix = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])   # = M @ primitive

# The SAME primitive path you gave the non-self-consistent dataset:
# primitive Gamma-X is k = (0, t/2, t/2), t = 0..1.
npts = 300
kpts = np.array([[0.0, t / 2.0, t / 2.0] for t in np.linspace(0.0, 1.0, npts)])

ax = unfold_abinit(
    'si7p_patho_DS2_WFK.nc',      # your netCDF WFK
    matrix,
    kpts,
    knames=[r'$\Gamma$', 'X'],
    xqpts=np.linspace(0.0, 1.0, npts),
    Xqpts=[0.0, 1.0],
    ylabel=r'Energy relative to $E_F$ (eV)',
    resolve_degenerate=1e-3,       # eV; keeps degenerate branch weights 0/1
    style='scatter',               # dense-k rendering without crossing artifacts
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

{{< figure src="/images/si7p_abinit_unfolded.png" title="Si:P unfolded: dark host bands at weight near 1, dimmer traces where the impurity mixes fold sectors" >}}

## Reading the figure

Host bands keep weight near 1 and render as the darkest traces. The donor
states introduced by the substitution, and host states hybridized with
impurity-scattered sectors at avoided crossings, carry fractional weight
and appear dimmer. Weights of a normalized state over all fold sectors sum
to 1.

## Weights without plotting

```python
from unfolding.abinit_unfold import read_wfk
from unfolding.pw_unfolder import PWUnfolder

data = read_wfk('si7p_patho_DS2_WFK.nc')   # -> immutable WFKData
result = PWUnfolder(data, matrix).compute(kpts)   # -> PWWeights
```

`read_wfk` rejects unreadable variants with actionable messages (the
`iomode 3` and `istwfk 1` remedies above).
