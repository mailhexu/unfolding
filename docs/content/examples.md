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
    names = [r'$\Gamma$', r'X', r'W', r'$\Gamma$', r'L']

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