#!/usr/bin/env python
try:
    import abipy.abilab as abilab
except ModuleNotFoundError as exc:
    if exc.name not in ("abipy", "abipy.abilab"):
        raise  # a different dependency is missing: surface the real error
    raise ImportError(
        "abipy is required for the Abinit DDB adapter. "
        "Install it with: pip install unfolding[abipy]"
    ) from exc
import numpy as np
from ase.build import bulk
from ase.dft.kpoints import get_special_points, bandpath
from unfolding.phonon_unfolder import phonon_unfolder
from unfolding.plotphon import plot_band_weight
from unfolding.units import EV_TO_CM
import matplotlib.pyplot as plt

from ase.data import atomic_masses  # indexed by atomic number Z


def displacement_cart_to_evec(displ_cart, masses, scaled_positions, qpoint=None, add_phase=True):
    """
    displ_cart: cartisien displacement. (atom1_x, atom1_y, atom1_z, atom2_x, ...)
    masses: masses of atoms.
    scaled_postions: scaled postions of atoms.
    qpoint: if phase needs to be added, qpoint must be given.
    add_phase: whether to add phase to the eigenvectors.

    Mass-weights the displacement by sqrt(m) and always normalizes the
    result to unit Euclidean norm; the phase, when requested, is applied
    before normalization.
    """
    m = np.sqrt(np.kron(masses, [1, 1, 1]))
    if add_phase and qpoint is None:
        raise ValueError('qpoint must be given if adding phase is needed')
    evec=displ_cart *m
    if add_phase:
        phase = [np.exp(-2j*np.pi*np.dot(pos,qpoint)) for pos in scaled_positions]
        phase = np.kron(phase,[1,1,1])
        evec*=phase
    evec /= np.linalg.norm(evec)
    return evec




def DDB_unfolder(DDB_fname, kpath_bounds, sc_mat, knames=None, kx=None, dipdip=1):
    """
    Unfold phonon bands from an Abinit DDB file along a k-path.

    Args:
        DDB_fname: DDB file name.
        kpath_bounds: Path vertices **in fractional reciprocal coordinates of
            the DDB structure** (the cell stored in the DDB, e.g. the
            conventional cubic cell for an fcc DDB computed with natom=4).
            If your special points come from ase's ``get_special_points`` on
            the *primitive* cell, convert them first, e.g.
            ``k_conv = k_prim @ sc_mat`` (see examples/Cu_fcc/unfold.py).
        sc_mat: Supercell matrix in units of the DDB-cell lattice vectors
            (rows convention ``S = A_ddb @ sc_mat``); its translation lattice
            supplies the phase interference used for the spectral weights.
        knames: Labels for the path vertices; defaults to the coordinates.
    """
    DDB = abilab.abiopen(DDB_fname)
    struct = DDB.structure
    atoms = DDB.structure.to_ase_atoms()
    scaled_positions = struct.frac_coords

    #cell = struct.lattice_vectors()
    cell = struct.lattice
    numbers = struct.atomic_numbers
    masses = [atomic_masses[Z] for Z in numbers]

    #print numbers
    #print cell
    #print scaled_positions

    #print kpath_bounds

    phbst, phdos = DDB.anaget_phbst_and_phdos_files(
        nqsmall=2,
        asr=1,
        chneut=1,
        dipdip=dipdip,
        verbose=1,
        ndivsm=40,
        lo_to_splitting=True,
        qptbounds=kpath_bounds,
        )
    #phbst.plot_phbands()
    qpoints = phbst.qpoints.frac_coords
    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evals = np.zeros([nqpts, nbranch])
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    #positions=np.kron(scaled_positions,[1,1,1])
    
    for iqpt, qpt in enumerate(qpoints):
        for ibranch in range(nbranch):
            phmode = phbst.get_phmode(qpt, ibranch)
            evals[iqpt, ibranch] = phmode.freq
            #evec=phmode.displ_cart *m
            #phase = [np.exp(-2j*np.pi*np.dot(pos,qpt)) for pos in scaled_positions]
            #phase = np.kron(phase,[1,1,1])
            #evec*=phase
            #evec /= np.linalg.norm(evec)
            evec=displacement_cart_to_evec(phmode.displ_cart, masses, scaled_positions, add_phase=False)
            evecs[iqpt,:,ibranch] = evec
            
    uf = phonon_unfolder(atoms,sc_mat,evecs,qpoints,phase=False)
    weights = uf.get_weights()
    x=np.arange(nqpts)
    freqs=evals
    xpts=[]
    for ix in range(len(x)):
        for q in kpath_bounds:
            if np.sum((np.array(qpoints[ix])-np.array(q))**2)<0.00001 and ix not in xpts:
                xpts.append(ix)
    if knames is None:
        knames=[str(k) for k in kpath_bounds]

    #names = ['$\Gamma$', 'X', 'W', '$\Gamma$', 'L']
    #ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*33.356,weights[:,:].T*0.98+0.01,xticks=[names,X],axis=ax)
    ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*EV_TO_CM,weights[:,:].T*0.99+0.001,xticks=[knames,xpts],style='alpha')
    #ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*EV_TO_CM,weights[:,:].T*0.98+0.000001,xticks=[knames, kx],style='alpha' )

    #plt.show()
    return ax

def nc_unfolder(fname, sc_mat, kx=None, knames=None, plot_width=False, weight_multiplied_by=None):
    ncfile=abilab.abiopen(fname)
    struct = ncfile.structure
    atoms = ncfile.structure.to_ase_atoms()
    scaled_positions = struct.frac_coords

    cell = struct.lattice_vectors()
    numbers = struct.atomic_numbers
    masses = [atomic_masses[Z] for Z in numbers]

    #print numbers
    #print cell
    #print scaled_positions



    #print kpath_bounds

    phbst = ncfile.phbands
    #phbst.plot_phbands()
    qpoints = phbst.qpoints.frac_coords
    nqpts = len(qpoints)
    nbranch = 3 * len(numbers)
    evals = np.zeros([nqpts, nbranch])
    evecs = np.zeros([nqpts, nbranch, nbranch], dtype='complex128')

    #positions=np.kron(scaled_positions,[1,1,1])
    freqs=phbst.phfreqs
    displ_carts=phbst.phdispl_cart
    
    for iqpt, qpt in enumerate(qpoints):
        for ibranch in range(nbranch):
            #phmode = ncfile.get_phmode(qpt, ibranch)
            #print(2)
            evals[iqpt, ibranch] = freqs[iqpt, ibranch]
            #evec=phmode.displ_cart *m
            #phase = [np.exp(-2j*np.pi*np.dot(pos,qpt)) for pos in scaled_positions]
            #phase = np.kron(phase,[1,1,1])
            #evec*=phase
            #evec /= np.linalg.norm(evec)
            evec=displacement_cart_to_evec(displ_carts[iqpt, ibranch,: ], masses, scaled_positions, add_phase=False)
            evecs[iqpt,:,ibranch] = evec
            
    uf = phonon_unfolder(atoms,sc_mat,evecs,qpoints,phase=False)
    weights = uf.get_weights()
    if plot_width:
        weights=(weights*(1.0-weights))**(0.5)
    if weight_multiplied_by is not None:
        weights=weights*weight_multiplied_by
    x=np.arange(nqpts)
    freqs=evals
    #names = ['$\Gamma$', 'X', 'W', '$\Gamma$', 'L']
    #ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*33.356,weights[:,:].T*0.98+0.01,xticks=[names,X],axis=ax)
    ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*EV_TO_CM,weights[:,:].T*0.98+0.000001,xticks=[knames, kx],style='alpha' )
    #plt.show()
    return ax

def main():
    #sc_mat = np.linalg.inv((np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0))
    #sc_mat=np.array([[0,1,1],[1,0,1],[1,1,0]])
    # Generate k-path for fcc Cu
    atoms = bulk('Cu','fcc')
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    GXW = [points[k] for k in 'GXWGL']
    kpts, x, X = bandpath(GXW, atoms.cell, 700)
    names = [r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L']
    return kpts, x, X, names, GXW

    sc_mat=np.eye(3)
    #points = kpath()[-1]
    points=np.array([(0,0,0),(0,.5,0),(0.5,0.5,0),[.5,.5,.5],[0,0,0]])
    DDB_unfolder(DDB_fname='out_DDB', kpath_bounds = [np.dot(k, sc_mat) for k in points],sc_mat=sc_mat)


