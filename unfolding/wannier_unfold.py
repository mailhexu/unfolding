from types import SimpleNamespace

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
        # the axis may either collapse a revisited vertex (ase dedupe ->
        # one tick per label) or split the path there (duplicate tick);
        # mirror whichever it did so tick count always matches
        tick_labels = list(knames)
        revisits = [i for i, vertex in enumerate(kvectors)
                    if any(np.allclose(vertex, prev) for prev in kvectors[:i])]
        for i in revisits[: max(0, len(list(X)) - len(knames))]:
            tick_labels.insert(i + 1, knames[i])
        if len(tick_labels) != len(list(X)):
            tick_labels = None  # never crash on an unmappable path
        kslist = [x] * len(self.positions)
        weights = self.unfold(kpts)
        wkslist = weights.T * 0.98 + 0.01
        ekslist = self.evals  # [nband, nk]: one row per band, as plot_band_weight expects
        self.last_result = SimpleNamespace(
            kpoints=kpts, eigenvalues=self.evals.T, weights=weights)
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
            xticks=[tick_labels, X] if tick_labels is not None else None)
        return ax


_BOHR_TO_ANG = 0.529177210903


def read_wannier90_hr(path):
    """Parse a Wannier90 ``*_hr.dat`` file.

    Returns ``(Rs, H)``: integer R vectors ``(NR, 3)`` (rows sorted) and
    the real-space Hamiltonian blocks ``H[i]`` for ``Rs[i]`` (eV, complex,
    ``(num_wann, num_wann)``). Handles the standard degeneracy preamble
    and the 15-values-per-line wrapping; spin-polarized ``*_hr.dat``
    files (``num_wann`` doubled) parse unchanged.
    """
    def is_int(token):
        return token.lstrip("+-").isdigit()

    with open(path) as fh:
        lines = fh.readlines()
    if not lines:
        raise ValueError(f"empty hr file: {path}")
    # Wannier90 >= 2.0 prepends a "written on ..." comment line before
    # num_wann (and num_r); older files start with num_wann directly.
    norb = None
    for line in lines:
        tok = line.split()
        if tok and is_int(tok[0]):
            norb = int(tok[0])
            break
    if norb is None:
        raise ValueError(f"{path}: no num_wann line")
    blocks = {}
    ndata = 0
    for line in lines[1:]:
        tok = line.split()
        if len(tok) != 7 or not all(is_int(t) for t in tok[:5]):
            continue  # degeneracy preamble / blank / wrapped header
        rx, ry, rz, m, n = (int(t) for t in tok[:5])
        blocks.setdefault((rx, ry, rz), np.zeros((norb, norb), dtype=complex)
                          )[m - 1, n - 1] = float(tok[5]) + 1j * float(tok[6])
        ndata += 1
    if not blocks:
        raise ValueError(f"no R-matrix entries found in {path}")
    if ndata % (norb * norb):
        raise ValueError(
            f"truncated hr file: {ndata} data lines for {norb} orbitals")
    Rs = np.array(sorted(blocks), dtype=float)
    if len(blocks) != ndata // (norb * norb):
        raise ValueError(
            f"inconsistent hr file: {len(blocks)} blocks for "
            f"{ndata // (norb * norb)} R vectors")
    return Rs, np.stack([blocks[tuple(R)] for R in Rs])


def _win_unit_block(lines, begin, unit_line_index, col=(0, 3), alat=None):
    """Matrix rows of a win block with an Ang/Bohr/Alat unit line.

    The unit line, when present, is wannier90's keyword (``ang``/``bohr``
    /``alat``); any other first row means the rows start right away and
    are read as Angstrom (wannier90's default). ``col`` selects the three
    coordinate tokens per row (the species token of ``atoms_cart`` rows
    is skipped with ``col=(1, 4)``). ``alat`` is the |a1| reference for
    ``alat`` blocks; without it the block's own first row is used (the
    ``unit_cell_cart`` case, where that row is a1).
    """
    unit = lines[unit_line_index].strip().lower().split()
    if unit[:1] in (["ang"], ["bohr"], ["alat"]):
        keyword, row_start = unit[0], unit_line_index + 1
    else:
        keyword, row_start = "ang", unit_line_index
    factor = {"ang": 1.0, "bohr": _BOHR_TO_ANG}.get(keyword)
    if factor is None:
        # alat: multiples of the reference lattice vector's length
        if alat is not None:
            factor = float(alat)
        else:
            first = lines[row_start].split()
            factor = float(np.linalg.norm([float(v) for v in first[:3]]))
    c0, c1 = col
    rows = []
    for line in lines[row_start:]:
        tok = line.split()
        if tok and tok[0].lower() in ("end", "begin"):
            break
        if len(tok) >= c1:
            try:
                rows.append([factor * float(v) for v in tok[c0:c1]])
            except ValueError:
                break
        if len(rows) == 3:
            break
    if len(rows) != 3:
        raise ValueError(f"incomplete {begin} block")
    return np.asarray(rows, dtype=float)


def read_wannier90_win(path):
    """Parse ``unit_cell_cart`` and ``atoms_frac``/``atoms_cart`` from a win.

    Returns ``(cell (3,3) Ang, sites (nsites, 3) fractional or None)``.
    """
    with open(path) as fh:
        lines = [line.split("#")[0] for line in fh]
    lowered = [line.lower() for line in lines]

    cell = None
    idx = next((i for i, t in enumerate(lowered) if "unit_cell_cart" in t), None)
    if idx is not None:
        cell = _win_unit_block(lines, "unit_cell_cart", idx + 1)

    sites = None
    idx = next((i for i, t in enumerate(lowered) if "atoms_frac" in t), None)
    if idx is not None:
        rows = []
        for line in lines[idx + 1:]:
            tok = line.split()
            if not tok or tok[0].lower() in ("end", "begin"):
                break
            try:
                rows.append([float(v) for v in tok[1:4]])
            except ValueError:
                break
        if rows:
            sites = np.asarray(rows, dtype=float)
    else:
        idx = next((i for i, t in enumerate(lowered) if "atoms_cart" in t), None)
        if idx is not None:
            cart = _win_unit_block(
                lines, "atoms_cart", idx + 1, col=(1, 4),
                alat=None if cell is None
                else float(np.linalg.norm(cell[0])))
            if cell is not None:
                sites = cart @ np.linalg.inv(cell)
    if cell is None:
        raise ValueError(f"{path}: no unit_cell_cart block")
    return cell, sites


def _sc_site_positions(prim_frac, scmat):
    """Supercell-fractional copies of the primitive sites under ``scmat``."""
    det = int(round(abs(np.linalg.det(scmat))))
    inv = np.linalg.inv(np.asarray(scmat, dtype=float))
    out = []
    for f in np.asarray(prim_frac, dtype=float):
        for n0 in range(-2, 3):
            for n1 in range(-2, 3):
                for n2 in range(-2, 3):
                    cand = (f + np.array([n0, n1, n2], dtype=float)) @ inv
                    cand = np.mod(cand, 1.0)
                    cand = np.where(np.abs(cand - 1.0) < 1e-6, 0.0, cand)
                    if not any(np.allclose(cand, c, atol=1e-6) for c in out):
                        out.append(cand)
    if len(out) != det * len(prim_frac):
        raise ValueError(
            f"site expansion gave {len(out)} positions, expected "
            f"{det * len(prim_frac)} (supercell matrix inconsistent)")
    return np.asarray(out, dtype=float)


class Wannier90Model:
    """Duck-typed WannierUnfolder model from a Wannier90 directory.

    Reads ``<prefix>_hr.dat``; the unit cell and atomic positions come
    from ``<prefix>.win`` (``cell`` overrides the win cell). Orbital
    positions (``_orb``, supercell-fractional) are the WF centres from
    ``<prefix>_centres.xyz`` when present, else each orbital sits on its
    atom's supercell position (uniform per-site orbital counts).
    Exposes minimulti's MyTB surface: ``.atoms`` (unit cell), ``._orb``,
    ``.solve_all(k_list=, eig_vectors=)``.
    """

    def __init__(self, model_dir, prefix, cell=None, scmat=None):
        from ase import Atoms
        from ase.io import read as ase_read

        import os

        hr = os.path.join(model_dir, f"{prefix}_hr.dat")
        if not os.path.isfile(hr):
            raise FileNotFoundError(f"Wannier90 hr file not found: {hr}")
        self.Rs, self.H = read_wannier90_hr(hr)
        self.norb = self.H.shape[1]

        win = os.path.join(model_dir, f"{prefix}.win")
        win_cell, sites = read_wannier90_win(win) if os.path.isfile(win) \
            else (None, None)
        cell = np.asarray(cell, dtype=float) if cell is not None else win_cell
        if cell is None:
            raise ValueError(
                f"{win}: no unit_cell_cart; pass the unit cell explicitly")
        self.atoms = Atoms(cell=cell, pbc=True)

        centres = os.path.join(model_dir, f"{prefix}_centres.xyz")
        if os.path.isfile(centres):
            cart = ase_read(centres).positions
            self._orb = np.mod(cart @ np.linalg.inv(cell), 1.0)
            if len(self._orb) != self.norb:
                raise ValueError(
                    f"{centres}: {len(self._orb)} centres for "
                    f"{self.norb} orbitals")
            return
        if sites is None:
            sites = np.zeros((1, 3))  # one-site primitive cell at the origin
        scmat = np.eye(3) if scmat is None else np.asarray(scmat, dtype=float)
        site_pos = _sc_site_positions(sites, scmat)
        if self.norb % len(site_pos):
            raise ValueError(
                f"{self.norb} orbitals cannot be split uniformly over "
                f"{len(site_pos)} supercell sites; provide "
                f"{prefix}_centres.xyz")
        self._orb = np.repeat(site_pos, self.norb // len(site_pos), axis=0)

    def solve_all(self, k_list, eig_vectors=False):
        k = np.atleast_2d(np.asarray(k_list, dtype=float))
        phase = np.exp(2j * np.pi * k @ self.Rs.T)
        hk = np.einsum("kr,rmn->kmn", phase, self.H)
        hk = 0.5 * (hk + np.conj(np.transpose(hk, (0, 2, 1))))
        evals, evecs = np.linalg.eigh(hk)
        if not eig_vectors:
            return evals.T
        return evals.T, np.transpose(evecs, (1, 0, 2))


def run(path, prefix, labels, scmat, output_figure, kvectors, knames,
        npoints=200, cell=None, return_result=False):
    """Convenience driver reading a Wannier90 directory.

    Uses minimulti's MyTB reader when that (older) API is available,
    else the built-in :class:`Wannier90Model` reader, which needs only
    ``<prefix>_hr.dat`` plus a ``<prefix>.win`` (or an explicit ``cell``)
    and drives the same :class:`WannierUnfolder`. Hopping pruning, if
    needed, is configured on the model before unfolding.

    With ``return_result=True`` returns ``(ax, result)`` where ``result``
    carries ``kpoints`` (supercell path points), ``eigenvalues`` (eV) and
    ``weights`` arrays of the drawn figure.
    """
    tb = None
    try:
        from minimulti.unfolding.wannier.myTB import MyTB
    except ImportError as exc:
        # fall back to the built-in reader when minimulti is absent OR its
        # own package internals are broken on this install; unrelated
        # ImportErrors still surface
        root = (getattr(exc, "name", "") or "").split(".")[0]
        if root != "minimulti":
            raise
    else:
        tb = MyTB.read_from_wannier_dir(path=path, prefix=prefix)
    if tb is None:
        tb = Wannier90Model(path, prefix, cell=cell, scmat=scmat)
    u = WannierUnfolder(tb, labels=labels, sc_matrix=scmat)
    ax = u.plot_unfolded_band(kvectors=kvectors, knames=knames, npoints=npoints)
    plt.savefig(output_figure)
    plt.show()
    if return_result:
        return ax, u.last_result
    return ax


if __name__ == "__main__":
    run(path='data_nodefect',
        prefix='wannier90',
        labels=['pz', 'px', 'py'] * 12 + ['dz2', 'dxy', 'dyz', 'dx2', 'dxz'] * 4,
        scmat=[[1, -1, 0], [1, 1, 0], [0, 0, 2]],
        output_figure='STO_nodefect.png',
        kvectors=np.array([[0, 0, 0], [.5, 0, 0], [.5, .5, 0], [0, 0, 0], [.5, .5, .5]]),
        knames=[r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R'])
