try:
    from phonopy import Phonopy
    from phonopy.file_IO import parse_FORCE_CONSTANTS, parse_disp_yaml, parse_FORCE_SETS
    from phonopy.structure.atoms import PhonopyAtoms
except ModuleNotFoundError as exc:
    if exc.name != "phonopy":
        raise  # a different dependency is missing: surface the real error
    raise ImportError(
        "phonopy is required for the Phonopy adapter. "
        "Install it with: pip install unfolding[phonopy]"
    ) from exc

from ase.atoms import Atoms
from ase.io import read
import numpy as np
from numpy.linalg import inv

from ase.dft.kpoints import bandpath, get_special_points
import matplotlib.pyplot as plt

from unfolding.phonon_unfolder import phonon_unfolder
from unfolding.plotphon import plot_band_weight
from unfolding.units import THZ_TO_CM

def read_phonopy( sposcar='SPOSCAR', sc_mat=np.eye(3),force_constants=None,  disp_yaml=None, force_sets=None):
    if force_constants is None and (disp_yaml is None or force_sets is None):
        raise ValueError("Either FORCE_CONSTANTS or (disp.yaml&FORCE_SETS) file should be provided.")
    atoms=read(sposcar)
    #vesta_view(atoms)
    primitive_matrix=inv(sc_mat)
    bulk = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            scaled_positions=atoms.get_scaled_positions(),
            cell=atoms.get_cell())
    phonon = Phonopy(
        bulk,
        supercell_matrix=np.eye(3),
        primitive_matrix=primitive_matrix,
        #factor=factor,
        #symprec=symprec
    )

    if disp_yaml is not None:
        disp=parse_disp_yaml(filename=disp_yaml)
        phonon.set_displacement_dataset(disp)
    if force_sets is not None:
        phonon.dataset = parse_FORCE_SETS(filename=force_sets)
        phonon.produce_force_constants()
    if force_constants is not None:
        phonon.force_constants = parse_FORCE_CONSTANTS(force_constants)

    return phonon

def unf(phonon, sc_mat, qpoints, knames=None, x=None, xpts=None,
        return_result=False):
    prim=phonon._primitive
    prim=Atoms(symbols=prim.symbols, cell=prim.cell, positions=prim.positions)
    sc_qpoints=np.array([np.dot(q, sc_mat) for q in qpoints])
    phonon.run_qpoints(sc_qpoints, with_eigenvectors=True)
    qpoint_phonons=phonon.get_qpoints_dict()
    #freqs, eigvecs = phonon.get_qpoints_phonon()
    freqs=qpoint_phonons['frequencies']
    eigvecs=qpoint_phonons['eigenvectors']
    # Conventions for non-diagonal sc_mat (e.g. R-centred hexagonal cells):
    # phonopy builds the supercell as A_sc = sc_mat^T @ A_prim, so the q-map
    # q_sc = q_prim @ sc_mat is exact; ASE's make_supercell used inside
    # phonon_unfolder generates translation lattice points with the
    # transposed convention, so the maps need sc_mat^T.  Verified against a
    # pristine R-3m supercell (MDR Rb3B12H12I): identical matrices scramble
    # the translation orbits; this pairing gives exact binary weights.
    uf=phonon_unfolder(atoms=prim, supercell_matrix=sc_mat.T, eigenvectors=eigvecs, qpoints=sc_qpoints, phase=False)
    # phonopy eigenvectors carry only the fold label (q folded out of the
    # gauge); the robust weights reproduce the shipped character-sum values
    # bit-for-bit on pure gauges while also surviving mixed/real gauges.
    weights = uf.get_weights_robust(freqs, gauge="fold")

    ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*THZ_TO_CM,weights[:,:].T*0.99+0.001,xticks=[knames,xpts],style='alpha')
    if return_result:
        from types import SimpleNamespace
        return ax, SimpleNamespace(
            kpoints=np.asarray(qpoints, dtype=float),
            eigenvalues=np.asarray(freqs) * THZ_TO_CM,
            weights=np.asarray(weights),
        )
    return ax

def phonopy_unfold(sc_mat=np.diag([1,1,1]), unfold_sc_mat=np.diag([3,3,3]),force_constants='FORCE_CONSTANTS', sposcar='SPOSCAR', qpts=None, qnames=None, xqpts=None, Xqpts=None, return_result=False):
    phonon=read_phonopy(sc_mat=sc_mat, force_constants=force_constants, sposcar=sposcar)
    return unf(phonon, sc_mat=unfold_sc_mat, qpoints=qpts, knames=qnames,x=xqpts, xpts=Xqpts, return_result=return_result )




from ase.build import bulk
def kpath():
    atoms = bulk('Cu', 'fcc', a=3.61)
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    GXW = [points[k] for k in 'GXWGL']
    kpts, x, X = bandpath(GXW, atoms.cell, 300)
    names = [r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L']
    return kpts, x, X, names


def test():
    phonon=read_phonopy(sc_mat=np.diag([1,1,1]), force_constants='FORCE_CONSTANTS', sposcar='SPOSCAR')
    qpts, x, X, names=kpath()
    ax=unf(phonon, sc_mat=np.diag([3,3,3]), qpoints=qpts, knames=names,x=x, xpts=X )
    plt.show()


if __name__=="__main__":
    test()
