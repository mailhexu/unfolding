"""Supercell-to-normal-cell relabel map (story 006).

Builds the unambiguous correspondence behind Eq. 25's relabeling
$M \\mapsto (m(M), \\mathbf r_0(M))$: every supercell orbital is assigned

* the normal-cell orbital index ``m`` it folds onto, and
* the integer cell offset ``r0`` (in primitive-lattice vectors) of the
  normal cell it belongs to,

including defect supercells (vacancies raise; atoms displaced by up to
``tol_r`` -- inclusively -- still match).

Matching conventions
--------------------
* Positions are compared in **Cartesian** space: a supercell atom at
  primitive-fractional coordinate ``f`` is matched against periodic
  images of the primitive atoms; the cell offset is the integer image
  ``r0`` used by the successful match (never an independent ``floor``,
  so a displacement that crosses a cell face while staying within
  ``tol_r`` of its atom does not flip the label). The Cartesian
  tolerance keeps ``tol_r`` physically meaningful (the same number means
  the same displacement in any cell), unlike the fractional-component
  convention of the phonon path's ``_make_translate_maps``.
* The minimum image is searched over the 3x3x3 neighbouring lattice
  images in the Cartesian metric, which is exact for skew primitive
  cells at any tolerance small compared to the lattice constants.
* Species must match exactly; per-atom orbital counts (HamiltonIO
  ``orb_dict``) must match between paired atoms.
* ``unfold_sc_mat`` is validated in full against the supplied cells
  (row-vector convention ``sc_cell = unfold_sc_mat @ prim_cell``), not
  merely by determinant.
"""
from __future__ import annotations

import itertools

import numpy as np


class RelabelMapError(ValueError):
    """Raised when the supercell cannot be relabeled onto the primitive cell."""


def _as_counts(counts, n_atoms, which):
    """Normalize per-atom orbital counts.

    Accepts a sequence of positive integers or a HamiltonIO-style dict
    ``{atom_index: [orbital names]}``.
    """
    if counts is None:
        return [1] * n_atoms
    if isinstance(counts, dict):
        missing = set(range(n_atoms)) - set(counts)
        if missing:
            raise RelabelMapError(
                f"orb_dict_{which} is missing atoms {sorted(missing)}"
            )
        return [len(counts[i]) for i in range(n_atoms)]
    counts = list(counts)
    if len(counts) != n_atoms or any(int(c) < 1 for c in counts):
        raise RelabelMapError(
            f"orb_counts_{which} must give >= 1 orbital for each of the "
            f"{n_atoms} atoms"
        )
    return [int(c) for c in counts]


class RelabelMap:
    """Orbital-level relabel map of a supercell onto its primitive cell.

    Attributes
    ----------
    orb_to_m : (N_orb_sc,) int array
        Normal-cell orbital index of each supercell orbital.
    orb_to_r0 : (N_orb_sc, 3) int array
        Primitive-lattice cell offset of each supercell orbital.
    offsets : (n_cells, 3) int array
        Lexicographically sorted distinct cell offsets.
    orb_to_cell : (N_orb_sc,) int array
        Index of each orbital's offset within ``offsets``.
    rep_orbital : (N_orb_prim, n_cells) int array
        Inverse map: supercell orbital representing (prim orbital, cell).
    representatives : (N_orb_prim,) int array
        Canonical representative supercell orbital of each primitive
        orbital, taken in the canonical cell ``offsets[0]``.
    atom_to_m, atom_to_r0 : atom-level maps (before orbital expansion).
    orb_counts_sc, orb_counts_prim : per-atom orbital counts used.
    """

    def __init__(
        self,
        sc_atoms,
        prim_atoms,
        atom_to_m,
        atom_to_r0,
        orb_counts_sc,
        orb_counts_prim,
    ):
        self._sc_atoms = sc_atoms
        self._prim_atoms = prim_atoms
        self.atom_to_m = np.asarray(atom_to_m, dtype=int)
        self.atom_to_r0 = np.asarray(atom_to_r0, dtype=int)
        self.orb_counts_sc = list(orb_counts_sc)
        self.orb_counts_prim = list(orb_counts_prim)

        n_orb_prim = int(sum(orb_counts_prim))
        offsets = np.unique(self.atom_to_r0, axis=0)
        self.offsets = offsets
        n_cells = len(offsets)
        cell_index = {tuple(off): i for i, off in enumerate(offsets)}

        # orbital-level expansion, atom-major blocks in atoms-list order
        starts_sc = np.concatenate([[0], np.cumsum(orb_counts_sc)[:-1]])
        starts_prim = np.concatenate([[0], np.cumsum(orb_counts_prim)[:-1]])

        n_orb_sc = int(sum(orb_counts_sc))
        self.orb_to_m = np.zeros(n_orb_sc, dtype=int)
        self.orb_to_r0 = np.zeros((n_orb_sc, 3), dtype=int)
        self.orb_to_cell = np.zeros(n_orb_sc, dtype=int)
        for a in range(len(sc_atoms)):
            m = self.atom_to_m[a]
            base_prim = int(starts_prim[m])
            for local in range(orb_counts_sc[a]):
                o = int(starts_sc[a] + local)
                self.orb_to_m[o] = base_prim + local
                self.orb_to_r0[o] = self.atom_to_r0[a]
                self.orb_to_cell[o] = cell_index[tuple(self.atom_to_r0[a])]

        self.rep_orbital = np.full((n_orb_prim, n_cells), -1, dtype=int)
        for o in range(n_orb_sc):
            self.rep_orbital[self.orb_to_m[o], self.orb_to_cell[o]] = o
        if np.any(self.rep_orbital < 0):
            raise RelabelMapError("unfilled (m, r0) slot: supercell is incomplete")
        # canonical-cell representative of each primitive orbital
        self.representatives = self.rep_orbital[:, 0].copy()

    @classmethod
    def from_atoms(
        cls,
        sc_atoms,
        prim_atoms,
        unfold_sc_mat,
        tol_r: float = 0.04,
        orb_counts_sc=None,
        orb_counts_prim=None,
    ):
        """Match every supercell atom to (primitive atom, cell offset).

        Parameters
        ----------
        sc_atoms, prim_atoms : ase.Atoms
            Supercell and primitive cell (same species sets).
        unfold_sc_mat : (3, 3) int array
            Supercell matrix; row-vector convention
            ``sc_cell = unfold_sc_mat @ prim_cell`` is validated.
        tol_r : float
            Cartesian matching tolerance (Angstrom); a displacement of
            exactly ``tol_r`` still matches.
        orb_counts_sc, orb_counts_prim : optional
            Per-atom orbital counts, either a sequence of positive ints
            or a HamiltonIO-style dict ``{atom index: [orbital names]}``.
            Default: one orbital per atom. Matched atoms must carry
            equal counts.
        """
        prim_cell = np.asarray(prim_atoms.cell)
        sc_cell = np.asarray(sc_atoms.cell)
        n_prim, n_sc = len(prim_atoms), len(sc_atoms)

        mat = np.asarray(unfold_sc_mat, dtype=float)
        if mat.shape != (3, 3) or not np.allclose(mat, np.round(mat), atol=1e-8):
            raise RelabelMapError("unfold_sc_mat must be a 3x3 integer matrix")
        ratio = int(round(abs(np.linalg.det(mat))))
        if ratio == 0:
            raise RelabelMapError("unfold_sc_mat is singular")
        if n_sc != n_prim * ratio:
            raise RelabelMapError(
                f"atom counts inconsistent: {n_sc} supercell atoms cannot "
                f"fold onto {n_prim} primitive atoms x {ratio} cells"
            )
        if not np.allclose(mat @ prim_cell, sc_cell, rtol=1e-6, atol=1e-6):
            raise RelabelMapError(
                "unfold_sc_mat @ prim_cell does not reproduce the supplied "
                "supercell cell (row-vector convention)"
            )

        counts_sc = _as_counts(orb_counts_sc, n_sc, "sc")
        counts_prim = _as_counts(orb_counts_prim, n_prim, "prim")

        prim_frac = prim_atoms.get_scaled_positions(wrap=True)
        symbols_sc = np.array(sc_atoms.get_chemical_symbols())
        symbols_prim = np.array(prim_atoms.get_chemical_symbols())

        # supercell positions in primitive fractional coordinates
        f_sc = np.asarray(sc_atoms.positions) @ np.linalg.inv(prim_cell)

        atom_to_m = np.full(n_sc, -1, dtype=int)
        atom_to_r0 = np.zeros((n_sc, 3), dtype=int)
        used = {}

        # candidate offsets: the 3x3x3 shell around the naive nearest image.
        # Distance is measured UNWRAPPED for each candidate offset, so an
        # atom displaced across a cell face stays matched to its own copy
        # (the wrapped minimum image would falsely match the neighbouring
        # copy), while the shell search still finds the true Cartesian
        # nearest image for skew primitive cells.
        shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))

        for a in range(n_sc):
            best = None
            for m in range(n_prim):
                if symbols_sc[a] != symbols_prim[m]:
                    continue
                if counts_sc[a] != counts_prim[m]:
                    continue
                n_center = np.rint(f_sc[a] - prim_frac[m]).astype(int)
                for s in shifts:
                    n = n_center + s
                    d_frac = f_sc[a] - n - prim_frac[m]
                    dist = float(np.linalg.norm(d_frac @ prim_cell))
                    if dist <= tol_r and (best is None or dist < best[0]):
                        best = (dist, m, n)
            if best is None:
                raise RelabelMapError(
                    f"supercell atom {a} ({symbols_sc[a]} at cartesian "
                    f"{np.round(sc_atoms.positions[a], 4).tolist()}) matches no "
                    f"primitive atom within tol_r={tol_r}"
                )
            _, m, n = best
            key = (m, tuple(n))
            if key in used:
                raise RelabelMapError(
                    f"supercell atoms {used[key]} and {a} claim the same "
                    f"(m, r0) = {key}"
                )
            used[key] = a
            atom_to_m[a] = m
            atom_to_r0[a] = n

        # bijection completeness: all offsets present, each fully populated
        n_offsets = len({tuple(r) for r in atom_to_r0})
        if n_offsets != ratio:
            raise RelabelMapError(
                f"expected {ratio} distinct cell offsets, found {n_offsets} "
                "(vacancy or extra atom in the supercell)"
            )
        for m in range(n_prim):
            matched = sum(1 for (mm, _) in used if mm == m)
            if matched != ratio:
                raise RelabelMapError(
                    f"primitive atom {m} matched by {matched} supercell "
                    f"atoms, expected {ratio}"
                )

        # the offsets must tile the supercell: in supercell-fractional
        # coordinates their residues mod 1 must be pairwise distinct
        cart = atom_to_r0 @ prim_cell
        u = cart @ np.linalg.inv(sc_cell)
        keys = {tuple(np.round(v - np.floor(v), 6)) for v in u}
        if len(keys) != n_offsets:
            raise RelabelMapError(
                "cell offsets collide modulo the supercell lattice; "
                "check unfold_sc_mat against the supplied cells"
            )

        self = cls(sc_atoms, prim_atoms, atom_to_m, atom_to_r0,
                   counts_sc, counts_prim)
        self.scmat = np.asarray(unfold_sc_mat, dtype=int)
        # translation lookup keyed by (cell c, cell cp): the supercell
        # integer triple T whose translation carries cell c onto cell cp
        # on the torus (scmat @ T + offsets[c] - offsets[cp] ~= 0)
        inv_scmat = np.linalg.inv(self.scmat.astype(float))
        offs = np.unique(atom_to_r0, axis=0)
        scmat_keys = {}
        for c in range(len(offs)):
            for cp in range(len(offs)):
                dd = offs[c] - offs[cp]
                T = np.round(dd @ inv_scmat).astype(int)
                scmat_keys[(c, cp)] = tuple(int(v) for v in T)
        self.scmat_keys = scmat_keys
        return self
