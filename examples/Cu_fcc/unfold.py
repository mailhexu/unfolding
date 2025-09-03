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
