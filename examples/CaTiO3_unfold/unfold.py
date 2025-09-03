import numpy as np
from unfolding.DDB_unfolder import nc_unfolder, DDB_unfolder
import matplotlib.pyplot as plt

def test():
    ax=DDB_unfolder('./out.DDB',
            sc_mat=[[1,-1,0],[1,1,0],[0,0,2]], 
            kpath_bounds=[[0,0,0],[0,.5,0], [.5,.5,0],[0,0,0],[.5,.5,.5]],
            knames=['$\Gamma$', 'X','M', '$\Gamma$', 'R'],
            dipdip=0) 
    plt.savefig('unfolded.png')
    plt.show()

test()
