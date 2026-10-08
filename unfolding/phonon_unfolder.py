"""
Phonon unfolding: Reciprocal space method. The method is described in
P. B. Allen et al. Phys Rev B 87, 085322 (2013).
This method should be also applicable to other bloch waves on discrete grid, eg. electrons wave function in wannier basis set, magnons, etc. Now only phonon istested.
"""
from ase.build import make_supercell
from ase.atoms import Atoms
import numpy as np

from . import degeneracy

class phonon_unfolder:
    """ phonon unfolding class"""

    def __init__(self, atoms, supercell_matrix, eigenvectors, qpoints, tol_r=0.04, ndim=3,labels=None,compare=None,phase=False):
        """
        phase=False is the correct pairing for phonopy-gauge eigenvectors
        (the canonical input from phonopy and the DDB adapters), where the
        sector decomposition is carried by the exp(-2*pi*i*G*r) factor in
        get_weight. Only pass phase=True for eigenvectors that carry no
        Bloch phase information at all.

        Params:
        ===================
        atoms: The structure of supercell.
        supercell matrix: The matrix that convert the primitive cell to supercell.
        eigenvectors: The phonon eigenvectors. format np.array() index=[ikpts, ifreq, 3*iatoms+j]. j=0..2
        qpoints: list of q-points, note the q points are in the BZ of the supercell.
        tol_r: tolerance. If abs(a-b) <r, they are seen as the same atom.
        ndim: number of dimensions. For 3D phonons, use ndim=3. For electrons(no spin), ndim=1. For spinors, use ndim=2 (TODO: spinor not tested. is it correct?).
        labels: labels of the basis. for 3D phonons, ndim can be set to 1 alternately, with labels set to ['x','y','z']*natoms. The labels are used to decide if two basis are identical by translation. (Not used for phonon)
        compare: how to decide the basis are identical (Not used for phonon)
        """
        self._atoms = atoms
        self._scmat = supercell_matrix
        self._evecs = eigenvectors
        self._qpts = qpoints
        self._tol_r = tol_r
        self._ndim = ndim
        self._labels=labels
        self._trans_rs = None
        self._trans_indices = None
        self._make_translate_maps()
        self._phase=phase
        return

    def _make_translate_maps(self):
        """
        find the mapping between supercell and translated cell.
        Returns:
        ===============
        A N * (ndim*natoms) array.
        index[i] is the mapping from supercell to translated supercell so that
        T(r_i) psi = psi[indices[i]].
        
        TODO: vacancies/add_atoms not supported. How to do it? For vacancies, a ghost atom can be added. For add_atom, maybe we can just ignore them? Will it change the energy spectrum?
        """
        a1 = Atoms(symbols='H', positions=[(0, 0, 0)], cell=[1, 1, 1])
        sc = make_supercell(a1, self._scmat)
        rs = sc.get_scaled_positions()
        positions = np.array(self._atoms.get_scaled_positions())
        nat = len(positions)
        ndim = self._ndim
        indices = np.zeros([len(rs), nat * ndim], dtype='int32')
        atom_ids = np.arange(nat)
        for i, ri in enumerate(rs):
            # target j_atom: (positions[j] + r_i) - positions[i] integer
            dpos = (positions + np.asarray(ri))[None, :, :] - positions[:, None, :]
            mask = np.all(np.abs(dpos - np.round(dpos)) < self._tol_r, axis=-1)
            # last matching i_atom per j_atom (loop-overwrite semantics);
            # unmatched columns keep the initialized zeros
            last = np.where(mask, atom_ids[:, None], -1).max(axis=0)
            match = np.repeat(mask.any(axis=0), ndim)
            row = np.repeat(np.where(mask.any(axis=0), last, 0) * ndim, ndim) \
                + np.tile(np.arange(ndim), nat)
            indices[i] = np.where(match, row, 0)

        self._trans_rs = rs
        self._trans_indices = indices

    def get_weight(self, evec, qpt, G=None):
        """
        get the weight of a mode which has the wave vector of qpt and eigenvector of evec.
        W= sum_1^N < evec| T(r_i)exp(-I (K+G) * r_i| evec>, here G=0. T(r_i)exp(-I K r_i)| evec> = evec[indices[i]]
        """
        if G is None:
            G = np.zeros(len(qpt))
        evec = np.asarray(evec)
        N = len(self._trans_rs)
        G = np.asarray(G, dtype=float)
        if self._phase:
            ph = np.exp(-1j * 2 * np.pi * (self._trans_rs @ (np.asarray(qpt, dtype=float) + G))) / N
        else:
            ph = np.exp(-1j * 2 * np.pi * (self._trans_rs @ G)) / N
        E2 = evec[self._trans_indices]                       # (N, ndof)
        weight = ((E2 @ np.conj(evec)) @ ph).real
        if weight < 0.0:
            weight = 0.0
        elif weight > 1.0:
            weight = 1.0
        return weight

    def get_weights(self):
        """
        Get the weight for all the modes.
        """
        evecs = np.asarray(self._evecs)
        nqpts, nfreqs = evecs.shape[0], evecs.shape[1]
        weights = np.zeros([nqpts, nfreqs])
        N = len(self._trans_rs)
        for iqpt in range(nqpts):
            qpt = np.asarray(self._qpts[iqpt], dtype=float)
            E = evecs[iqpt]                                 # (ndof, nfreqs)
            Ec = np.conj(E)
            if self._phase:
                ph = np.exp(-1j * 2 * np.pi * (self._trans_rs @ qpt)) / N
            else:
                ph = np.full(N, 1.0 / N)
            w = np.zeros(nfreqs, dtype=complex)
            for it, ind in enumerate(self._trans_indices):
                w += ph[it] * np.sum(Ec * E[ind, :], axis=0)
            weights[iqpt] = np.clip(w.real, 0.0, 1.0)

        self._weights = weights
        return self._weights

    def _home_orbit_columns(self):
        """DOF columns of one representative atom per translation orbit.

        The translation group partitions the supercell atoms into orbits
        (one per primitive-cell atom); the orbit representatives supply the
        "home" degrees of freedom on which the Bloch sums used by
        ``get_weights_robust`` are built.
        """
        nat = self._trans_indices.shape[1] // self._ndim
        seen = set()
        columns = []
        for j in range(nat):
            if j in seen:
                continue
            seen.update(int(i) for i in self._trans_indices[:, j * self._ndim] // self._ndim)
            columns.extend(range(j * self._ndim, (j + 1) * self._ndim))
        return np.array(columns, dtype=int)

    def get_weights_robust(self, frequencies, gauge="fold", degenerate_tol=1e-6):
        """Gauge-robust unfolding weights at the identity target k* = q.

        For every stored supercell q-point, the weight of each mode is the
        squared projection of its eigenvector onto the Bloch sums

        $$
        |kappa, k*> = (1/sqrt(N)) sum_r exp(-2 pi i (k* - q_carry).r) T_r |kappa>,
        $$

        built on the translation group {r} and the home atoms. The weight
        matrix H = A^dagger A (with A the projection of the eigenvector
        matrix onto those Bloch sums) is Hermitian positive semidefinite
        for any eigenvector gauge, and diagonalizing it within degenerate
        frequency groups yields per-branch weights that are exactly binary
        (0 or 1) for a pristine supercell.

        Two eigenvector gauges are supported:

        - ``"fold"``: the eigenvectors carry only the fold label (phonopy's
          convention); q_carry = q.
        - ``"bloch"``: the eigenvectors carry the full Bloch momentum q+g
          (anaddb's convention); q_carry = 0.

        This repairs the historical fractional-weights failure of the
        Abinit DDB route: on mirror-symmetric q-paths the real dynamical
        matrix returns eigenvectors that are cosine mixtures of the +k and
        -k fold sectors, which the bare character sum ``get_weights``
        cannot resolve. ``frequencies`` are the mode eigenvalues in the
        array's native unit (eV for the DDB route, THz for the phonopy
        route); ``degenerate_tol`` groups modes in that unit.
        """
        if gauge not in ("fold", "bloch"):
            raise ValueError("gauge must be 'fold' or 'bloch'")
        evecs = np.asarray(self._evecs)
        nqpts = evecs.shape[0]
        R = np.asarray(self._trans_rs, dtype=float)
        home = self._home_orbit_columns()
        N = len(R)
        weights = np.zeros_like(np.asarray(frequencies, dtype=float))
        for iqpt in range(nqpts):
            qpt = np.asarray(self._qpts[iqpt], dtype=float)
            qcarry = qpt if gauge == "fold" else np.zeros_like(qpt)
            ph = np.exp(-2j * np.pi * (R @ (qpt - qcarry)))
            E = evecs[iqpt]
            # one gather of the home columns for all translations + contraction
            A = np.tensordot(ph, E[self._trans_indices[:, home], :], axes=1)
            A = A / np.sqrt(N)
            H = A.conj().T @ A
            weights[iqpt] = _block_weights(H, frequencies[iqpt], degenerate_tol)
        self._weights = weights
        return weights


def _block_weights(H, freqs, tol):
    """Per-branch weights from a Hermitian weight matrix ``H``.

    Modes are grouped by (near-)degenerate frequencies and ``H`` is
    diagonalized inside each group: degenerate modes may be stored in any
    unitary mixture (a real dynamical matrix on mirror-symmetric paths
    mixes the +k and -k fold sectors), and only the group-resolved
    eigenvalues are gauge invariant. Sorted eigenvalues are assigned to
    the group's branches (any assignment is equivalent inside a
    degenerate group).
    """
    freqs = np.asarray(freqs, dtype=float)
    # singletons carry the diagonal; degenerate groups receive the
    # sorted-descending block spectrum (any assignment is equivalent
    # inside a degenerate group)
    w = np.asarray(np.diag(H)).real.astype(float).copy()
    order = np.argsort(freqs)
    degeneracy.assign_eigvalsh_groups(w, H, freqs, tol, order=order,
                                      clip=(0.0, 1.0))
    return np.clip(w, 0.0, 1.0)
