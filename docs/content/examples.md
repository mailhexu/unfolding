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

![Phonopy Unfolded Band Structure](/images/phonopy_unfolded_band_structure.png)

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

![Cu_fcc Unfolded Band Structure](/images/cu_fcc_unfolded.png)

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

![CaTiO3 Unfolded Band Structure](/images/catio3_unfolded.png)