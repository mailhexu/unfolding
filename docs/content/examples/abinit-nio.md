---
title: "ABINIT WFK AFM NiO"
weight: 8
---

Antiferromagnets need a magnetic cell larger than the chemical one, and
the WFK stores two separate spin channels. This example unfolds
**type-II AFM NiO** (Ni moments alternating along [111]) from the
4-atom magnetic primitive cell onto the **2-atom rocksalt primitive
cell**. Because the AFM keeps inversion symmetry, the spin-up and
spin-down unfolded band structures coincide — so one channel is plotted,
selected with `spin=0` (the `spin=1` partner is identical).

## The magnetic cell

Type-II AFM ordering doubles the 2-atom rocksalt primitive cell along
[111]. With `M = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]` (rows = the magnetic
axes in primitive units, $A_{afm} = M \cdot A_{prim}$) the magnetic cell
holds two Ni (moments +m and −m along z) and two O:

```text
acell 3*7.8817                        # a = 4.171 Angstrom
rprim 1.0 0.5 0.5
      0.5 1.0 0.5
      0.5 0.5 1.0
natom 4
typat 1 1 2 2                         # Ni Ni O O
xred
  0.00 0.00 0.00                      # Ni +m
  0.50 0.50 0.50                      # Ni -m
  0.25 0.25 0.25                      # O
  0.75 0.75 0.75                      # O
nsppol 2
nspden 2
spinat 0 0 2.0  0 0 -2.0  0 0 0  0 0 0
```

The collinear WFK (`nsppol 2`) stores two independent spin channels; the
unfolding weight of a state is computed within its own channel, so no
spin mixing is involved.

## Unfolding one spin channel

The same reciprocal-coset machinery applies per channel: pass `spin=0`
(up) or `spin=1` (down) and the adapter unfolds that channel onto the
2-atom primitive cell along Γ–X–W–Γ–L–W–X, exactly as in the Si example:

```python
import numpy as np
from unfolding import unfold_abinit

m_afm = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])

ax = unfold_abinit(
    'nio_afm_patho_DS2_WFK.nc', m_afm, kpts,
    knames=[r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L', 'W', 'X'],
    xqpts=x, Xqpts=X,
    spin=0,                        # spin-up channel; spin=1 is identical here
    resolve_degenerate=1e-3,
)
```

{{< figure src="/images/nio_afm_unfolded.png" title="AFM NiO unfolded onto the 2-atom primitive cell (spin-up channel; spin-down is identical by symmetry)" >}}

## Reading the figure

Only the spin-up channel is plotted because the spin-down one is its
exact partner: type-II AFM NiO keeps inversion symmetry, and time
reversal followed by the sublattice-translation maps the spin-down
Hamiltonian onto the spin-up one at the same momentum. Both channels
have identical band energies *and* identical unfolding weights (verified
numerically: the channel eigenvalues agree to < 0.05 eV everywhere) —
for a collinear AFM with inversion, plotting the second channel adds no
information. Systems without that symmetry (ferrimagnets, AFMs without
inversion) do show distinct channels; the recipe is the same, with
`spin=1` for the partner panel.

The AFM character shows up differently: the bands are the *exchange-split*
branches of the two Ni sublattices (local moments ±1.36 μB on the two
Ni, totalling zero), folded from the magnetic cell onto the 2-atom
primitive cell. States of both fold sectors carry weight ≈ 1 where they
project on the primitive momentum; plain PBE makes the moment small and
the gap close (a Hubbard U restores the insulator without changing the
unfolding recipe). The Ni-3s semicore multiplet near −60 eV (the
pseudopotential carries 18 valence electrons) is out of the plotted
window.
