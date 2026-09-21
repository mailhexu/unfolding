---
title: "Tutorial"
weight: 2
---

# Tutorial

## Installation

- Python >= 3.9
- Core dependencies: numpy, scipy, matplotlib, ase
- Optional extras:

| Extra | Enables | Dependencies |
|---|---|---|
| `phonopy` | `phonopy_unfold` (phonon supercell unfolding) | phonopy |
| `abinit` | `read_wfk`, `unfold_abinit` (ABINIT ETSF WFK unfolding) | netCDF4 |
| `abipy` | `DDB_unfolder` (Abinit DDB unfolding) | abipy (anaddb required to produce DDB files) |
| `siesta` | `unfold_siesta` (SIESTA LCAO unfolding) | HamiltonIO >= 0.3.8, sisl |
| `dev` | test suite | pytest, sympy |

```bash
pip install unfolding[phonopy]     # phonopy unfolding
pip install unfolding[siesta]      # SIESTA unfolding
pip install unfolding[abinit]      # ABINIT ETSF WFK unfolding
pip install unfolding[abipy]       # Abinit DDB unfolding
```

## Unfold phonon bands with phonopy

Unfold the phonon of a 3x3x3 fcc Cu supercell onto the fcc primitive
cell. Files: `examples/phonopy` (FORCE_CONSTANTS, SPOSCAR).

```python
import numpy as np
from ase.build import bulk
from ase.dft.kpoints import bandpath, get_special_points
from unfolding.phonopy_unfolder import phonopy_unfold

atoms = bulk('Cu', 'fcc', a=3.61)
points = get_special_points('fcc', atoms.cell, eps=0.01)
kpts, x, X = bandpath([points[k] for k in 'GXWGL'], atoms.cell, 300)

ax = phonopy_unfold(
    sc_mat=np.diag([1, 1, 1]),
    unfold_sc_mat=np.diag([3, 3, 3]),
    force_constants='FORCE_CONSTANTS',
    sposcar='SPOSCAR',
    qpts=kpts, xqpts=x, Xqpts=X,
    qnames=[r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L'],
)
```

The result is a matplotlib Axes with weight-coded bands (see
`docs/static/images/phonopy_unfolded_band_structure.png`).

## Unfold electron bands from SIESTA

Prerequisite: `pip install unfolding[siesta]`. Run a SIESTA
supercell calculation (e.g. the committed 8-atom Si example under
`tests/data/si_example/`) and unfold onto the primitive cell.

```python
import numpy as np
from unfolding import unfold_siesta
from ase.io import read

prim_atoms = read('primitive.xsf')
kpts = ...   # primitive-cell path, (N, 3) fractional

ax = unfold_siesta(
    fdf='si_sc.fdf',            # SIESTA input; parsed via HamiltonIO
    prim_atoms=prim_atoms,      # 2-atom primitive cell
    unfold_sc_mat=np.eye(3, dtype=int) * 2,
    kpts=kpts, xqpts=x, Xqpts=X, knames=['G', 'X', 'M', 'G'],
)
```

Collinear spin-polarized runs expose `spin='up'|'down'`; a pre-parsed
HamiltonIO model can be passed as `model=` instead of `fdf=`. The
committed example (`tests/data/si_example/` + `docgen/fig_siesta_si.py`)
produces `si_unfolded.png` without running SIESTA.

## Unfold electron bands from ABINIT WFK

Prerequisite: `pip install unfolding[abinit]`. The adapter reads ABINIT's
ETSF netCDF WFK directly; it does not require abipy. The committed Si and
Si7P fixtures under `tests/data/abinit_si/` are regenerated on nic6 with:

| ABINIT input setting | Why it is required |
|---|---|
| `iomode 3` | Emit an ETSF netCDF WFK instead of this build's Fortran-binary default. |
| `istwfk 1` | Store the full G sphere; half-sphere reconstruction is intentionally unsupported. |
| `chkprim 0` | Allow the 8-atom conventional cubic Si cell. |
| `prtwf 1` and nband headroom | Preserve raw coefficients and keep edge-state mixing outside the seal. |

```bash
bash tests/data/abinit_si/regenerate_on_nic6.sh
```

The primitive path is always given in primitive fractional coordinates. For
the Si7P Gamma--X fixture, the supercell reciprocal path is formed internally
as `K = frac(k @ M.T)`: primitive `(0, t/2, t/2)` maps to conventional-cell
`(t, 0, 0)`. The adapter converts WFK Hartree eigenvalues to eV at the plotting
boundary and shifts by the stored Fermi energy by default.

```python
from pathlib import Path
import numpy as np
from unfolding import unfold_abinit

matrix = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
kpts = np.array([[0.0, t / 2.0, t / 2.0] for t in (0, .25, .5, .75, 1)])
wfk = Path('tests/data/abinit_si/si7p_gamma_x_patho_DS2_WFK.nc')

ax = unfold_abinit(
    wfk, matrix, kpts,
    knames=[r'$\Gamma$', 'X'], xqpts=np.linspace(0, 1, len(kpts)),
    Xqpts=[0, 1], ylabel=r'Energy relative to $E_F$ (eV)',
)
```

`read_wfk(path)` exposes the immutable `WFKData` record if weights are needed
without plotting. It rejects Fortran WFKs with an `iomode 3` remedy and rejects
`istwfk != 1` with the input setting required for the full-G route.

{{< figure src="/images/si7p_abinit_unfolded.png" title="ABINIT Si7P: Gamma--X unfolded spectral weight; red markers identify fractional donor-window states" >}}


## Unfold Abinit DDB results

Prerequisite: abipy with a working anaddb. Run the DDB example
(`examples/CaTiO3_unfold`, `examples/Cu_fcc`).

```python
from unfolding.DDB_unfolder import DDB_unfolder

ax = DDB_unfolder('./out.DDB', sc_mat=[[1,-1,0],[1,1,0],[0,0,2]],
                  kpath_bounds=[[0,0,0],[0,.5,0],[.5,.5,0],[0,0,0],[.5,.5,.5]],
                  knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'],
                  dipdip=0)
```

## Unfold from wannier90

The wannier example (`examples/wannier_STO`) unfolds
wannier90-derived Hamiltonians for a defect and a pristine SrTiO3
supercell; see `wannier_unfold.py` and `plot.py` in that directory.

## Legacy interface

The legacy `Unfolder` class is deprecated; use
`unfold_siesta`/`LCAOUnfolder` for LCAO calculations (ADR-003).
