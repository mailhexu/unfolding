"""Backend-free magnon band-structure unfolding engine (story 022,
PRD Rev 3 FR-019, ADR-011/012).

The engine consumes supercell magnon eigendata in the canonical BdG
amplitude convention and computes unfolding weights with the
reciprocal-coset Euclidean projector derived and sealed in
``derivations/magnon_weight.py``:

    tau_prim = tau_sc @ M.T          (per magnetic atom)
    m = tau_prim mod 1               (primitive sublattice)
    n = tau_prim - m in Z^3/M^T Z^3  (coset label)
    W_q(nu) = sum_{s in {u,v}} sum_m |sum_{a in m} chi_q(n_a) psi^s_a|^2 / N

with chi_q(n) = exp(-2 pi i q . n), N = det M, and psi_nu the
Euclidean-normalized canonical amplitudes (Psi = K X; both sectors of a
clean fold state carry the same fold character). Near-degenerate mode
groups (G-AFM doublets are exact) get the same eigen-assignment
treatment as the electronic engines: within a group the invariant
content is the eigenvalue spectrum of the projected Gram, assigned
sorted-descending onto the energy-sorted group.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np

__all__ = ["MagnonEigenData", "MagnonUnfolder", "MagnonWeights"]


@dataclass(frozen=True)
class MagnonEigenData:
    """Supercell magnon eigendata on stored supercell momenta.

    Parameters
    ----------
    kpoints
        (nk, 3) stored supercell momenta, supercell fractional.
    energies
        (nk, nmode) positive magnon energies (same unit as
        ``degen_tolerance``).
    wavefunctions
        (nk, nmode, 2 nmag) canonical BdG amplitudes, u block then v
        block, Euclidean-normalized rows (``Psi = K @ X`` from the
        Cholesky frame; see derivations/magnon_weight.py).
    positions
        (nmag, 3) magnetic-atom fractional positions, supercell frame.
    """

    kpoints: np.ndarray
    energies: np.ndarray
    wavefunctions: np.ndarray
    positions: np.ndarray

    def __post_init__(self):
        kpoints = np.asarray(self.kpoints, dtype=float)
        energies = np.asarray(self.energies, dtype=float)
        wavefunctions = np.asarray(self.wavefunctions)
        positions = np.asarray(self.positions, dtype=float)
        for name, arr in (
            ("kpoints", kpoints),
            ("energies", energies),
            ("wavefunctions", wavefunctions),
            ("positions", positions),
        ):
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (nk, 3)")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (nmag, 3)")
        nmag = positions.shape[0]
        if energies.shape != (len(kpoints), wavefunctions.shape[1]):
            raise ValueError("energies must have shape (nk, nmode)")
        if wavefunctions.shape != (len(kpoints), energies.shape[1], 2 * nmag):
            raise ValueError(
                "wavefunctions must have shape (nk, nmode, 2*nmag)"
            )


@dataclass(frozen=True)
class MagnonWeights:
    """Unfolded magnon weights along a primitive-cell q-path."""

    kpoints: np.ndarray
    energies: np.ndarray
    weights: np.ndarray
    sc_kpoints: np.ndarray

    def average_degenerate(self, tolerance):
        """Average weights inside near-degenerate energy groups."""
        energies = np.asarray(self.energies)
        weights = np.asarray(self.weights).copy()
        for ik in range(len(energies)):
            order = np.argsort(energies[ik], kind="stable")
            energies_sorted = energies[ik][order]
            start = 0
            for stop in range(1, len(energies_sorted) + 1):
                if (
                    stop == len(energies_sorted)
                    or energies_sorted[stop] - energies_sorted[stop - 1] > tolerance
                ):
                    if stop - start > 1:
                        weights[ik][order[start:stop]] = weights[ik][
                            order[start:stop]
                        ].mean(axis=0)
                    start = stop
        return MagnonWeights(
            self.kpoints, self.energies, weights, self.sc_kpoints
        )

    def plot(self, xqpts=None, **kwargs):
        """Weight-coded magnon bands through :func:`plot_band_weight`."""
        from .plotphon import plot_band_weight

        if xqpts is None:
            xqpts = np.arange(len(self.kpoints), dtype=float)
        kslist = [xqpts for _ in range(self.energies.shape[1])]
        ekslist = [self.energies[:, i] for i in range(self.energies.shape[1])]
        wkslist = [self.weights[:, i] for i in range(self.weights.shape[1])]
        return plot_band_weight(kslist, ekslist, wkslist, **kwargs)


class MagnonUnfolder:
    """Reciprocal-coset magnon unfolding engine.

    Parameters
    ----------
    eigendata
        :class:`MagnonEigenData` with canonical BdG amplitudes.
    unfold_sc_mat
        Integer (3, 3) matrix, row convention ``A_sc = M @ A_prim``.
    tol_k
        Periodic distance tolerance for locating stored supercell
        momenta.
    """

    def __init__(self, eigendata: MagnonEigenData, unfold_sc_mat, tol_k: float = 1e-6):
        if not isinstance(eigendata, MagnonEigenData):
            raise TypeError("eigendata must be a MagnonEigenData instance")
        matrix = np.asarray(unfold_sc_mat, dtype=int)
        if matrix.shape != (3, 3):
            raise ValueError("unfold_sc_mat must have shape (3, 3)")
        if round(np.linalg.det(matrix)) == 0:
            raise ValueError("unfold_sc_mat must be nonsingular")
        if tol_k <= 0:
            raise ValueError("tol_k must be positive")
        self.eigendata = eigendata
        self.unfold_sc_mat = matrix
        self.tol_k = float(tol_k)
        self._analyze_sites()

    # -- site bookkeeping -------------------------------------------------

    def _analyze_sites(self):
        """Per magnetic atom: primitive sublattice and coset label."""
        M = self.unfold_sc_mat.astype(float)
        tau_prim_raw = self.eigendata.positions @ M.T
        tau_prim = np.mod(tau_prim_raw, 1.0)
        n_vec = tau_prim_raw - tau_prim
        if np.max(np.abs(n_vec - np.round(n_vec))) > 1e-6:
            raise ValueError(
                "magnetic-atom positions are not commensurate with "
                "unfold_sc_mat (tau_prim = tau_sc @ M.T must decompose "
                "into sublattice + integer cell)"
            )
        keys = [tuple(np.round(t, 6)) for t in tau_prim]
        sublattices = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        self._sublattice = np.array([sublattices[k] for k in keys])
        self._coset = np.round(n_vec).astype(int)
        self._tau_prim = tau_prim
        self.nsublattice = len(sublattices)
        nfold = round(abs(np.linalg.det(self.unfold_sc_mat)))
        counts = np.bincount(self._sublattice)
        if np.any(counts != nfold):
            raise ValueError(
                "magnetic-atom positions are not a supercell of the "
                "primitive magnetic basis: every primitive sublattice "
                f"must appear det(M) = {nfold} times, got {counts.tolist()}"
            )

    @staticmethod
    def fold_kpoints(unfold_sc_mat) -> np.ndarray:
        """Primitive fractional coset representatives (house convention)."""
        matrix = np.asarray(unfold_sc_mat, dtype=int)
        nfold = round(abs(np.linalg.det(matrix)))
        max_entry = int(np.abs(matrix).max())
        reps = {}
        for bound in range(0, max_entry + 2):
            for n in product(range(bound + 1), repeat=3):
                frac = np.mod(
                    np.linalg.solve(matrix, np.asarray(n, dtype=float)), 1.0
                )
                key = tuple(np.rint(frac * 1e12).astype(np.int64) % 10**12)
                reps.setdefault(key, frac)
            if len(reps) == nfold:
                break
        if len(reps) != nfold:
            raise RuntimeError("could not enumerate all reciprocal fold sectors")
        values = list(reps.values())
        values.sort(key=lambda v: (not np.allclose(v, 0.0), *v.tolist()))
        return np.asarray(values)

    # -- unfolding ---------------------------------------------------------

    def _stored_kpoint(self, kpoint: np.ndarray) -> int:
        target = np.mod(kpoint @ self.unfold_sc_mat.T, 1.0)
        delta = (
            np.mod(self.eigendata.kpoints - target + 0.5, 1.0) - 0.5
        )
        distances = np.linalg.norm(delta, axis=1)
        ik = int(np.argmin(distances))
        if distances[ik] > self.tol_k:
            raise ValueError(
                "target supercell momentum "
                f"{target.tolist()} has no stored match within "
                f"tol_k={self.tol_k:g}; nearest stored "
                f"{self.eigendata.kpoints[ik].tolist()} "
                f"(periodic distance {distances[ik]:.3g})"
            )
        return ik

    def _fold_projected(self, ik: int, fold):
        """(nmode, 2*nmag) amplitudes projected onto one fold.

        The fold projector acts per sublattice and per BdG sector as
        the character average over the coset labels; the projected
        vectors are the overlap objects whose Gram carries the
        gauge-invariant group content.
        """
        M = self.unfold_sc_mat.astype(float)
        N = round(abs(np.linalg.det(M)))
        nmag = self.eigendata.positions.shape[0]
        wf = self.eigendata.wavefunctions[ik]
        chi = np.exp(-2j * np.pi * self._coset @ np.mod(fold, 1.0))  # (nmag,)
        out = np.zeros_like(wf)
        for m in range(self.nsublattice):
            sel = self._sublattice == m
            for s_block in (slice(0, nmag), slice(nmag, 2 * nmag)):
                amps = wf[:, s_block][:, sel]  # (nmode, n_m)
                proj = (amps @ chi[sel])[:, None] * chi[sel].conj()[None, :] / N
                out[:, s_block][:, sel] = proj
        return out

    def _fold_weights(self, ik: int, fold) -> np.ndarray:
        """(nmode,) weights of stored supercell momentum ik on fold q."""
        return np.sum(np.abs(self._fold_projected(ik, fold)) ** 2, axis=1)

    def compute(self, kpoints, resolve_degenerate: Optional[float] = None) -> MagnonWeights:
        """Unfold primitive ``kpoints`` (one may lie in each fold).

        For each requested primitive momentum q the stored supercell
        momentum is frac(q @ M.T); the returned weight columns follow
        the ``fold_kpoints`` order. ``resolve_degenerate`` is an energy
        tolerance (same unit as ``energies``) for eigen-assigning
        weights inside near-degenerate mode groups.
        """
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.ndim == 1:
            kpoints = kpoints[None, :]
        if kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (nk, 3)")
        if resolve_degenerate is not None and resolve_degenerate < 0:
            raise ValueError("resolve_degenerate must be non-negative")

        nk, nmode = len(kpoints), self.eigendata.energies.shape[1]
        energies = np.empty((nk, nmode))
        weights = np.empty((nk, nmode))
        sc_kpoints = np.empty_like(kpoints)

        for i, q in enumerate(kpoints):
            ik = self._stored_kpoint(q)
            energies[i] = self.eigendata.energies[ik]
            # the fold label IS the requested primitive momentum: the
            # character chi_q acts directly, no canonical-rep matching
            weights[i] = self._fold_weights(ik, q)
            sc_kpoints[i] = self.eigendata.kpoints[ik]
            if resolve_degenerate:
                order = np.argsort(energies[i], kind="stable")
                esorted = energies[i][order]
                start = 0
                for stop in range(1, nmode + 1):
                    if (
                        stop == nmode
                        or esorted[stop] - esorted[stop - 1] > resolve_degenerate
                    ):
                        idx = order[start:stop]
                        if stop - start > 1:
                            # the gauge-invariant group content at this
                            # fold is the group-total weight; canonical
                            # BdG modes are g-orthogonal, not
                            # Euclidean-orthogonal, so per-mode spectra
                            # inside a degenerate group are not
                            # presentation-stable. Assign the invariant
                            # total: integer totals become b(1, .. 1, 0,
                            # .. 0) onto the energy-sorted group; a
                            # fractional total (defect-hybridized
                            # degeneracies) is split equally.
                            total = weights[i][idx].sum()
                            k = round(total)
                            if abs(total - k) < 1e-6:
                                assign = np.zeros(stop - start)
                                assign[:k] = 1.0
                            else:
                                assign = np.full(stop - start, total / (stop - start))
                            weights[i][idx] = assign
                        start = stop
        return MagnonWeights(kpoints, energies, weights, sc_kpoints)

