import warnings

import numpy as np
import matplotlib.pyplot as plt

from unfolding.plotphon import plot_band_weight
from unfolding.unfolder import Unfolder

# MyTB (Wannier90 reader) lives in the separate minimulti package; it is only
# needed by the run() convenience driver below, not by WannierUnfolder itself.


class WannierUnfolder(object):
    """Unfold a Wannier/tight-binding supercell band structure.

    `tbmodel` is any object with `.atoms` (ASE Atoms of the primitive cell),
    `._orb` (scaled orbital positions) and `.solve_all(k_list=, eig_vectors=)`
    returning `(evals [nband, nk], evecs [nband, nk, norb])` from the
    generalized eigenproblem of the *supercell* tight-binding model.
    (This matches the minimulti MyTB convention.)
    """

    def __init__(self, tbmodel, labels, sc_matrix):
        self.model = tbmodel
        self.labels = labels
        self.sc_matrix = sc_matrix
        self.cell = self.model.atoms.get_cell()
        self.positions = self.model._orb

    def unfold(self, kpts):
        self.evals, self.evecs = self.model.solve_all(
            k_list=kpts, eig_vectors=True)
        positions = self.model._orb
        # tbmodel: evecs[iband, ikpt, iorb]
        # unfolder: [ikpt, iorb, iband]
        with warnings.catch_warnings():
            # Unfolder is our deprecated legacy core; wannier adaptation keeps
            # using it until the orbital-based rewrite (internal use).
            warnings.simplefilter("ignore", DeprecationWarning)
            self.unf = Unfolder(
                cell=self.cell,
                basis=self.labels,
                positions=positions,
                supercell_matrix=self.sc_matrix,
                eigenvectors=np.swapaxes(np.swapaxes(self.evecs, 0, 1), 1, 2),
                qpoints=kpts)
        return self.unf.get_weights()

    def plot_unfolded_band(
            self,
            kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
                               [0, 0, 0], [.5, .5, .5]]),
            knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'],
            npoints=200,
            ax=None, ):
        """Plot the unfolded bands with weight alpha along a k-path."""
        if ax is None:
            fig, ax = plt.subplots()
        from ase.dft.kpoints import bandpath
        kvectors = [np.dot(k, self.sc_matrix) for k in kvectors]
        cell_sc = np.dot(self.sc_matrix, self.cell)
        path = bandpath(kvectors, cell_sc, npoints)
        kpts = path.kpts
        x, X, _ = path.get_linear_kpoint_axis()
        kslist = [x] * len(self.positions)
        wkslist = self.unfold(kpts).T * 0.98 + 0.01
        ekslist = self.evals  # [nband, nk]: one row per band, as plot_band_weight expects
        ax = plot_band_weight(
            kslist,
            ekslist,
            wkslist=wkslist,
            efermi=None,
            yrange=None,
            output=None,
            style='alpha',
            axis=ax,
            ylabel='Energy (eV)',
            ypad=float(np.ptp(self.evals) * 0.05 + 1e-3),
            xticks=[knames, X])
        return ax


def run(path, prefix, labels, scmat, output_figure, kvectors, knames, npoints=200):
    """Convenience driver reading a Wannier90 directory via minimulti's MyTB.

    Requires the `minimulti` package (pip install minimulti). See
    examples/wannier_STO for a runnable version with data. Hopping pruning,
    if needed, is configured on the MyTB model before unfolding.
    """
    try:
        from minimulti.unfolding.wannier.myTB import MyTB
    except ImportError as exc:
        if exc.name not in ("minimulti", "minimulti.unfolding", "minimulti.unfolding.wannier", "minimulti.unfolding.wannier.myTB"):
            raise
        raise ImportError(
            "minimulti is required for the Wannier90 reader (MyTB). "
            "Install it with: pip install minimulti"
        ) from exc
    tb = MyTB.read_from_wannier_dir(path=path, prefix=prefix)
    u = WannierUnfolder(tb, labels=labels, sc_matrix=scmat)
    ax = u.plot_unfolded_band(kvectors=kvectors, knames=knames, npoints=npoints)
    plt.savefig(output_figure)
    plt.show()
    return ax


if __name__ == "__main__":
    run(path='data_nodefect',
        prefix='wannier90',
        labels=['pz', 'px', 'py'] * 12 + ['dz2', 'dxy', 'dyz', 'dx2', 'dxz'] * 4,
        scmat=[[1, -1, 0], [1, 1, 0], [0, 0, 2]],
        output_figure='STO_nodefect.png',
        kvectors=np.array([[0, 0, 0], [.5, 0, 0], [.5, .5, 0], [0, 0, 0], [.5, .5, .5]]),
        knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'])
