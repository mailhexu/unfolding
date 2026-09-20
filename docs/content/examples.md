---
title: "Examples"
weight: 3
---

# Examples

## Phonopy Example

This example shows how to unfold the phonon band structure of a 3x3x3 FCC Cu structure to the primitive FCC structure.

Files used in this example:
- `SPOSCAR`: The supercell structure file
- `FORCE_CONSTANTS`: The force constants file from Phonopy

The example script `run_unfold.py` demonstrates how to use the phonopy_unfold function:

```python
import numpy as np

from ase.dft.kpoints import get_special_points, bandpath
import matplotlib.pyplot as plt
from unfolding.phonopy_unfolder import phonopy_unfold


def run_unfolding():
    # Generate k-path for FCC structure
    from ase.build import bulk
    atoms = bulk('Cu', 'fcc', a=3.61)
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    path_highsym = [points[k] for k in 'GXWGL']
    kpts, x, X = bandpath(path_highsym, atoms.cell, 300)
    names = [r'$\Gamma, r'X', r'W', r'$\Gamma, r'L']

    # here is the unfolding. Here is an example of 3*3*3 fcc cell.
    ax = phonopy_unfold(
        sc_mat=np.diag([1, 1, 1]),  # supercell matrix for phonopy.
        unfold_sc_mat=np.diag([3, 3, 3]),  # supercell matrix for unfolding
        force_constants='FORCE_CONSTANTS',  # FORCE_CONSTANTS file path
        sposcar='SPOSCAR',  # SPOSCAR file for phonopy
        qpts=kpts,  # q-points. In primitive cell!
        qnames=names,  # Names of high symmetry q-points in the q-path.
        xqpts=x,  # x-axis, should have the same length of q-points.
        Xqpts=X  # x-axis for high symmetry q-points
    )
    plt.savefig('unfolded_band_structure.png', dpi=300)
    plt.show()


run_unfolding()
```

The resulting unfolded band structure is saved as `unfolded_band_structure.png`.

{{< figure src="/images/phonopy_unfolded_band_structure.png" title="Phonopy Unfolded Band Structure" >}}

## Cu_fcc Example

This example demonstrates unfolding the phonon band structure of FCC Cu using Abinit DDB files.

Files used in this example:
- `out_DDB`: The DDB file from Abinit
- `unfold.py`: The script to perform the unfolding

The example script `unfold.py` shows how to use the DDB_unfolder function:

```python
import numpy as np
from unfolding.DDB_unfolder import nc_unfolder, DDB_unfolder
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.dft.kpoints import get_special_points, bandpath

def run_unfolding():
    # Generate k-path for fcc Cu.
    atoms = bulk('Cu','fcc')
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    knames='GXWGL'
    kpath_bounds= [points[k] for k in 'GXWGL']
    sc_mat = np.linalg.inv((np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0))
    ax=DDB_unfolder('./out_DDB', sc_mat=sc_mat, kpath_bounds=kpath_bounds, knames=knames) 
    plt.savefig('unfolded.png')
    plt.show()

run_unfolding()
```

The resulting unfolded band structure is saved as `unfolded.png`.

{{< figure src="/images/cu_fcc_unfolded.png" title="Cu_fcc Unfolded Band Structure" >}}

## CaTiO3_unfold Example

This example shows how to unfold the phonon band structure of CaTiO3 using Abinit DDB files.

Files used in this example:
- `out.DDB`: The DDB file from Abinit
- `unfold.py`: The script to perform the unfolding

The example script `unfold.py` demonstrates how to use the DDB_unfolder function:

```python
import numpy as np
from unfolding.DDB_unfolder import nc_unfolder, DDB_unfolder
import matplotlib.pyplot as plt

def test():
    ax=DDB_unfolder('./out.DDB',
            sc_mat=[[1,-1,0],[1,1,0],[0,0,2]], 
            kpath_bounds=[[0,0,0],[0,.5,0], [.5,.5,0],[0,0,0],[.5,.5,.5]],
            knames=[r'$\Gamma, 'X','M', r'$\Gamma, 'R'],
            dipdip=0) 
    plt.savefig('unfolded.png')
    plt.show()

test()
```

The resulting unfolded band structure is saved as `unfolded.png`.

{{< figure src="/images/catio3_unfolded.png" title="CaTiO3 Unfolded Band Structure" >}}
## SIESTA: Si diamond (8-atom supercell)

Unfold an 8-atom conventional-cell Si supercell calculation onto the
2-atom primitive-cell path Gamma-X-W-Gamma-L-X. The fixtures
(`si_sc.HSX`, `si_prim.HSX`) are committed under
`tests/data/si_example/`; the script
`docgen/fig_siesta_si.py` regenerates the figure without SIESTA
(the Hamiltonian is read through
[HamiltonIO](https://github.com/aimatores/HamiltonIO)'s sisl parser).

```python
import numpy as np
from HamiltonIO.siesta.sisl_wrapper import SislParser

from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
from unfolding.mapping import RelabelMap
from unfolding.plotphon import plot_band_weight

class TorusSislParser(SislParser):
    def read_Rlist(self, geom=None):
        return self.ham.lattice.sc_off

prim = TorusSislParser("si_prim.fdf").get_model()
sc   = TorusSislParser("si_sc.fdf").get_model()

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv = B @ prim
rm = RelabelMap.from_atoms(sc.atoms, prim.atoms, B,
                           orb_counts_sc=[4]*8, orb_counts_prim=[4, 4])
unf = LCAOUnfolder(HamiltonIOModel(sc), rm)
res = unf.compute(kpts, method="ideal")   # generic-path spectral weights
```

`kpts` are the fcc high-symmetry points in *primitive* reciprocal
fractional coordinates (Gamma-X-W-Gamma-L-W-X); the supercell momenta
are formed internally as K = scmat^T k. Overlaying the independently
diagonalized primitive-cell bands on the weight plot validates the
unfolded spectrum: for the pristine crystal every weight-1 band lies
on a primitive band to within the SCF k-grid agreement (~2 meV here).

The one-call variant is `unfold_siesta(fdf=..., prim_atoms=...,
unfold_sc_mat=..., kpts=...)` (see the API reference); it parses the
fdf through HamiltonIO directly.

The committed fixtures were produced by k-grid SCF runs (4x4x4 on the
primitive cell, 2x2x2 on the supercell — matched sampling): a
Gamma-only `SaveHS` run collapses all supercell images into a single
R=0 shell, leaving a k-independent H that cannot be unfolded at
generic k. Use `method="ideal"` along generic paths; the `ring`
method remains exact at torus-commensurate momenta.

{{< figure src="/images/si_unfolded.png" title="SIESTA Si: 8-atom supercell unfolded onto the primitive path, with the primitive-cell bands overlaid" >}}

### Substitutional dopant: Si7P

Replacing one Si with P (the next element) breaks the ideal crystal:
the unfolding weight now measures how much each supercell state
resembles the ideal Si crystal. Host bands stay at weight 1 while
donor-derived and folded impurity states appear at reduced weight.
The dopant site maps onto the host site it replaces via
`RelabelMap.from_atoms(..., match_species=False)`.

{{< figure src="/images/si_p_doped_unfolded.png" title="SIESTA Si7P: one Si substituted by P; the ideal weight separates host bands (weight 1) from impurity-derived states" >}}

### Wavefunction-driven unfolding: WFSX instead of diagonalizing H

If the supercell run already stored wavefunctions (`SaveWFSX true` plus
a `%block WaveFuncKPoints` list on the path), `WFSXUnfolder` evaluates
the same weight formula from SIESTA's own coefficients and
eigenvalues — no Hamiltonian diagonalization:

```python
from HamiltonIO.siesta.wfsx import SiestaWFSXParser
from unfolding.wfsx_unfolder import WFSXUnfolder

cell = np.asarray(sc.atoms.cell)
wfsx = SiestaWFSXParser("si_sc_path.selected.WFSX", cell=cell).read()
unf = WFSXUnfolder(wfsx, HamiltonIOModel(sc), rm, sc_mat=B)
res = unf.compute(kpts, method="ideal")   # kpts: the same path
```

The overlap shells from the SC `.HSX` are still required (the weight
is an AO-overlap matrix); only the eigen-solve is replaced. WFSX
energies are returned exactly as SIESTA stores them — Fermi-shifted
by the writing run (subtract the `.EIG` header value for absolute
eigenvalues). WFSX coefficients are stored in SIESTA's orbital-position
gauge; at generic k they are converted to the unfolder's convention
with the per-orbital phase `exp(+2 pi i K . tau_s)` — a wrong sign
deviates the weights by orders of magnitude and is pinned by
`test_gauge_conversion_is_pinned`.

{{< figure src="/images/si_wfsx_unfolded.png" title="SIESTA Si from the committed WFSX path run: same spectrum as the HSX path, obtained without diagonalizing H" >}}

### Spinor (nspin=4) unfolding

Non-collinear runs work through the same machinery: spinor orbital
counts are doubled (4 PAO x 2 spin components), everything else is
identical. The committed spinor fixtures use `Spin.Orbit true` with a
scalar Si pseudopotential, so the spinor bands equal the scalar bands
with Kramers degeneracy — the example certifies the spinor pipeline
(parse -> relabel -> weights), not spinor physics.

```python
rm = RelabelMap.from_atoms(sc.atoms, prim.atoms, B,
                           orb_counts_sc=[8]*8, orb_counts_prim=[8, 8])
unf = LCAOUnfolder(HamiltonIOModel(sc), rm)
res = unf.compute(kpts, method="ideal")
```

{{< figure src="/images/si_spinor_unfolded.png" title="SIESTA Si spinor run (nspin=4): the spinor pipeline reproduces the primitive-cell bands with Kramers degeneracy" >}}
