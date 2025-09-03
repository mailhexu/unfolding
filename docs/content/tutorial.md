---
title: "Tutorial"
weight: 2
---

# Tutorial

## Installation

- Prerequisite : ase, numpy, scipy, matplotlib 
  For unfolding with phonopy: phonopy
  For unfolding with abinit/anaddb output: abinit and abipy. Abipy should be configured so anaddb can run.

- How to install:
   1. Download the package
   2.  use pip
   ``` 
   pip install -e .
   ```
   or
    ```
    pip install .
    ```

## How to use

### Unfold phonon band structure with phonopy

Here we use an example to show how to unfold the phonon band structure. The files can be found in the example/phonopy
directory. Here we unfold the phonon of a 3*3*3 FCC Cu structure to the FCC structure. 

1. Prepare the phonopy files. The files needed here are the supercell
file and the FORCE_CONSTANTS file ([link](https://atztogo.github.io/phonopy/input-files.html#structure-file-poscar) ). 
The FORCE_CONSTANTS file can be generated using the --fc option with phonopy from FORCE_SETS and other files.
 If you use DFPT in VASP, it can be directly generated using 
```
  phonopy --fc vasprun.xml 
```
In the example here, we use the VASP format for the supercell file (SPOSCAR).

2. Use the phonon_unfolder tool. 

We can create a python script like below. 

sc_mat: supercell matrix used by  phonopy to build the supercell for calculating the forces set.
unfold_sc_mat: the matrix which map the supercell (corresponding to the phonon) to a primitive cell
force_constatnts: the FORCE_CONSTANTS file.
sposcar: the SPOSCAR file (or other file formats supported.)
qpts: the q-points along a path. (Note: in the primitive BZ. There is no need to map it to supercell BZ unit.)
qnames: labels of the high symmetry q-points.
xqpts: the x-axis in the plot, each q-point has a x value.
Xqpts: the x-axis corresponding to the high symmetry points. 

```python
import numpy as np

from ase.dft.kpoints import get_special_points, bandpath
import matplotlib.pyplot as plt
from unfolding.phonopy_unfolder import phonopy_unfold


def run_unfolding():
    # Generate k-path for FCC structure
    # It is not necessary to generate the k-path with ase.
    from ase.build import bulk
    atoms = bulk('Cu', 'fcc', a=3.61)
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    path_highsym = [points[k] for k in 'GXWGL']
    kpts, x, X = bandpath(path_highsym, atoms.cell, 300)
    names = ['$\Gamma$', 'X', 'W', '$\Gamma$', 'L']
 
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
    plt.show()


run_unfolding()
```

### Unfold phonon band structure with abipy

Before using abipy to do the unfolding, anaddb ( a part of abinit), and abipy should be already installed and configured. (see the abipy documentation.)
Now again, let's take BCC Cu as an example. With the out_DDB file, we can plot the unfolded band using the DDB_unfolder class from unfolding.DDB_unfolder module. 

Now use the python script below, the band can be unfolded. 

```python
import numpy as np
from unfolding.DDB_unfolder import nc_unfolder, DDB_unfolder
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.dft.kpoints import get_special_points, bandpath

def run_unfolding():
    # Generate k-path for fcc Cu with ase. 
    # other package can be used as well. Or build it by hand.
    atoms = bulk('Cu','fcc')
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    knames='GXWGL'   # names
    kpath_bounds= [points[k] for k in 'GXWGL']  # points with names

    # Here is the supercell matrix
    sc_mat = np.linalg.inv((np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0))

    ax=DDB_unfolder('./out_DDB', # DDB file name
            sc_mat=sc_mat,       # supercell matrix
            kpath_bounds=kpath_bounds,  # special kpoints in PRIMITIVE CELL.
            knames=knames)              # names of special k-points.
    plt.savefig('unfolded.png')         # save to an png file
    plt.show()                          # show on screen.

run_unfolding()
```

### Unfold electron band structure from Wannier90

Documentation to be added. See minimulti examples.

### Unfold electron band structure from PROCAR file

See pyprocar.

### Unfold magnon band structure from minimulti.

See minimulti doc.