---
title: "ABINIT WFK AFM NiO"
weight: 8
---

Unfold type-II antiferromagnetic NiO from the 4-atom magnetic primitive
cell onto the 2-atom rocksalt primitive cell with the `unfold_abinit`
adapter. The magnetic cell (`A_afm = M @ A_prim`, `M = [[1,0,1],[0,1,1],
[1,1,0]]`) doubles the rocksalt primitive cell along [111] and carries
two Ni moments (+m and −m along z) and two O; the collinear WFK
(`nsppol 2`) stores two spin channels, selected with `spin=`.

## Running the example

Magnetic-cell deck (atomic positions in the magnetic cell):

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

Unfold one channel onto the 2-atom primitive cell along Γ–X–W–Γ–L–W–X,
as in the [Si example](/examples/abinit-wfk/) (`spin=0` up, `spin=1`
down):

```python
import numpy as np
from unfolding import unfold_abinit

m_afm = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])

ax = unfold_abinit(
    'nio_afm_patho_DS2_WFK.nc', m_afm, kpts,
    knames=[r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L', 'W', 'X'],
    xqpts=x, Xqpts=X,
    spin=0,
    resolve_degenerate=1e-3,
)
```

{{< figure src="/images/nio_afm_unfolded.png" title="AFM NiO unfolded onto the 2-atom rocksalt primitive cell along Γ-X-W-Γ-L-W-X, spin-up channel: Gaussian-smeared spectral-weight map (Blues scale, opacity = weight), energies relative to the WFK Fermi level" >}}

The unfolding weight of a state is computed within its own spin channel;
no spin mixing is involved. Systems whose channels differ are handled by
the same call with `spin=1` for the partner panel.

## Calculation background

- Code: ABINIT, `Ni.psp8` / `O.psp8` pseudopotentials (18-valence-electron
  Ni, so the Ni-3s semicore multiplet near −60 eV is present in the WFK
  but outside the plotted −16…8 eV window), `ixc 11`, `ecut 40` Ha,
  `nband 40`, `nsppol 2`, `nspden 2`, `spinat` as above, `tolwfr2 1e-16`.
- Cell: `acell 3*7.8817` bohr with the `rprim` above; magnetic =
  `M @` primitive with `M = [[1,0,1],[0,1,1],[1,1,0]]`.
- Decks (`tests/data/abinit_si/`): `nio_afm.abi` (dense Γ–X–W–Γ–L–W–X
  path) and `nio_afm_corners.abi` (sparse fallback); path WFKs are staged
  by `regenerate_on_nic6.sh` in the same directory.
- Figure: `python docgen/fig_abinit_nio.py` writes
  `docs/static/images/nio_afm_unfolded.png`.

Download the [complete input bundle](/downloads/abinit-nio.tar.gz)
(`abinit-nio.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
