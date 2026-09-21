"""Backend-free planewave unfolding weights (story 017).

A supercell plane wave at SC reduced momentum ``K`` and integer SC
G-vector ``n`` has primitive fractional wavevector
``w = M^-T (K + n)``, where rows of ``M`` are the supercell lattice
vectors in primitive-lattice units. Its fractional part is the
primitive translation sector. Thus the spectral weight on primitive
``k`` is the direct orthonormal projector norm

    W_n(k) = sum_{G_s: frac(M^-T (K + n_G)) = frac(k)} |c_nK(G_s)|^2.

This module deliberately knows nothing about ABINIT, netCDF, or units.
Adapters provide :class:`PWEigenData`; the WFK reader arrives in story
018. The only dependencies here are NumPy and the existing plotting
helper, imported lazily by :meth:`PWWeights.plot`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


def _owned_array(value, dtype) -> np.ndarray:
    """Copy a public-array boundary and freeze the stored value."""
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PWEigenData:
    """Planewave eigendata normalized for :class:`PWUnfolder`.

    Parameters are per stored supercell k-point so variable plane-wave
    counts are represented without padding:

    ``gvecs[ik]``
        Integer SC reciprocal vectors, shape ``(npw_k, 3)``.
    ``coefficients[ik]``
        Complex coefficients, shape ``(nspin, nband, nspinor, npw_k)``.
    ``eigenvalues[ik]``
        Energies in the source unit, shape ``(nspin, nband)``.

    The core selects one collinear ``spin`` channel at a time and sums
    the ``nspinor`` axis into the physical weight. Unit conversion and
    WFK-specific metadata stay in the adapter layer.
    """

    kpoints: np.ndarray
    gvecs: Sequence[np.ndarray]
    coefficients: Sequence[np.ndarray]
    eigenvalues: Sequence[np.ndarray]

    def __post_init__(self):
        kpoints = _owned_array(self.kpoints, float)
        if kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (nk, 3)")
        if len(kpoints) == 0:
            raise ValueError("at least one stored k-point is required")
        if not (len(self.gvecs) == len(self.coefficients) == len(self.eigenvalues) == len(kpoints)):
            raise ValueError("kpoints, gvecs, coefficients, and eigenvalues need one entry per k-point")

        gvecs, coefficients, eigenvalues = [], [], []
        model_shape = None
        for ik, (gvec, coeff, eig) in enumerate(zip(self.gvecs, self.coefficients, self.eigenvalues)):
            gvec = _owned_array(gvec, int)
            coeff = _owned_array(coeff, complex)
            eig = _owned_array(eig, float)
            if gvec.ndim != 2 or gvec.shape[1] != 3 or len(gvec) == 0:
                raise ValueError(f"gvecs[{ik}] must have nonzero shape (npw, 3)")
            if coeff.ndim != 4:
                raise ValueError(
                    f"coefficients[{ik}] must have shape (nspin, nband, nspinor, npw)"
                )
            if eig.ndim != 2:
                raise ValueError(f"eigenvalues[{ik}] must have shape (nspin, nband)")
            if coeff.shape[:2] != eig.shape:
                raise ValueError(f"coefficients/eigenvalues band dimensions differ at k-point {ik}")
            if coeff.shape[-1] != len(gvec):
                raise ValueError(f"coefficients/gvecs npw dimensions differ at k-point {ik}")
            if 0 in coeff.shape[:3]:
                raise ValueError(f"coefficients[{ik}] requires positive nspin, nband, and nspinor")
            if model_shape is None:
                model_shape = coeff.shape[:3]
            elif coeff.shape[:3] != model_shape:
                raise ValueError(
                    f"coefficients[{ik}] has (nspin, nband, nspinor)={coeff.shape[:3]}, "
                    f"expected {model_shape}"
                )
            gvecs.append(gvec)
            coefficients.append(coeff)
            eigenvalues.append(eig)

        object.__setattr__(self, "kpoints", kpoints)
        object.__setattr__(self, "gvecs", tuple(gvecs))
        object.__setattr__(self, "coefficients", tuple(coefficients))
        object.__setattr__(self, "eigenvalues", tuple(eigenvalues))

    @property
    def nspin(self) -> int:
        return self.coefficients[0].shape[0]


@dataclass(frozen=True)
class PWWeights:
    """Unfolded planewave weights along a primitive-cell k-path.

    ``kpoints`` preserves the user-supplied primitive coordinates;
    ``fold_kpoints`` is the same path reduced modulo the primitive
    reciprocal lattice, and ``sc_kpoints`` records the stored SC point
    used for each row. Eigenvalue units are inherited from the source.
    """

    kpoints: np.ndarray
    eigenvalues: np.ndarray
    weights: np.ndarray
    fold_kpoints: np.ndarray
    sc_kpoints: np.ndarray

    def __post_init__(self):
        kpoints = _owned_array(self.kpoints, float)
        eigenvalues = _owned_array(self.eigenvalues, float)
        weights = _owned_array(self.weights, float)
        fold_kpoints = _owned_array(self.fold_kpoints, float)
        sc_kpoints = _owned_array(self.sc_kpoints, float)
        if kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (nk, 3)")
        if eigenvalues.ndim != 2 or weights.shape != eigenvalues.shape:
            raise ValueError("eigenvalues and weights must have matching shape (nk, nband)")
        if eigenvalues.shape[0] != len(kpoints):
            raise ValueError("eigenvalues need one row per k-point")
        if fold_kpoints.shape != kpoints.shape or sc_kpoints.shape != kpoints.shape:
            raise ValueError("fold_kpoints and sc_kpoints must match kpoints shape")
        object.__setattr__(self, "kpoints", kpoints)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "fold_kpoints", fold_kpoints)
        object.__setattr__(self, "sc_kpoints", sc_kpoints)

    def average_degenerate(self, tol_e: float) -> "PWWeights":
        """Collapse near-degenerate bands into gauge-invariant group weights.

        Each k-point is energy-sorted before grouping. The returned group
        weights are *sums*, not averages: a two-dimensional degenerate
        subspace carrying one full sector's character has weight two.
        """
        if tol_e < 0:
            raise ValueError("tol_e must be non-negative")

        grouped_e, grouped_w = [], []
        max_groups = 0
        for eig, weight in zip(self.eigenvalues, self.weights):
            order = np.argsort(eig, kind="stable")
            eig = eig[order]
            weight = weight[order]
            out_e, out_w = [], []
            start = 0
            for stop in range(1, len(eig) + 1):
                if stop == len(eig) or eig[stop] - eig[stop - 1] > tol_e:
                    out_e.append(eig[start:stop].mean())
                    out_w.append(weight[start:stop].sum())
                    start = stop
            grouped_e.append(np.asarray(out_e))
            grouped_w.append(np.asarray(out_w))
            max_groups = max(max_groups, len(out_e))

        eig = np.zeros((len(grouped_e), max_groups))
        weight = np.zeros_like(eig)
        for ik, (out_e, out_w) in enumerate(zip(grouped_e, grouped_w)):
            eig[ik, : len(out_e)] = out_e
            weight[ik, : len(out_w)] = out_w
        return PWWeights(self.kpoints, eig, weight, self.fold_kpoints, self.sc_kpoints)

    def plot(self, xqpts=None, *, style="alpha", axis=None, ylabel="Energy", **kwargs):
        """Plot weight-coded bands through :func:`plot_band_weight`.

        ``xqpts`` defaults to integer path indices. Adapters can pass
        physically accumulated path distances while retaining this
        backend-free result object for programmatic callers.
        """
        from .plotphon import plot_band_weight

        if xqpts is None:
            xqpts = np.arange(len(self.kpoints), dtype=float)
        xqpts = np.asarray(xqpts, dtype=float)
        if xqpts.shape != (len(self.kpoints),):
            raise ValueError("xqpts must have one coordinate per k-point")

        kslist = [xqpts for _ in range(self.eigenvalues.shape[1])]
        ekslist = [self.eigenvalues[:, iband] for iband in range(self.eigenvalues.shape[1])]
        wkslist = [self.weights[:, iband] for iband in range(self.weights.shape[1])]
        return plot_band_weight(
            kslist,
            ekslist,
            wkslist,
            style=style,
            axis=axis,
            ylabel=ylabel,
            **kwargs,
        )


class PWUnfolder:
    """Reciprocal-coset planewave unfolding engine.

    Parameters
    ----------
    eigendata
        :class:`PWEigenData`, normally constructed by an input adapter.
    unfold_sc_mat
        Integer ``(3, 3)`` matrix with the project row convention:
        ``A_sc = M @ A_prim``.
    tol_k
        Periodic distance tolerance used when locating the stored SC
        k-point matching ``K = frac(k @ M.T)``.
    """

    def __init__(self, eigendata: PWEigenData, unfold_sc_mat, tol_k: float = 1e-8):
        if not isinstance(eigendata, PWEigenData):
            raise TypeError("eigendata must be a PWEigenData instance")
        matrix = _owned_array(unfold_sc_mat, int)
        if matrix.shape != (3, 3):
            raise ValueError("unfold_sc_mat must have shape (3, 3)")
        det = round(np.linalg.det(matrix))
        if det == 0:
            raise ValueError("unfold_sc_mat must be nonsingular")
        if tol_k <= 0:
            raise ValueError("tol_k must be positive")

        self.eigendata = eigendata
        self.unfold_sc_mat = matrix
        self.tol_k = float(tol_k)

    @staticmethod
    def fold_kpoints(unfold_sc_mat) -> np.ndarray:
        """Primitive fractional coset representatives of ``Z^3 / M^T Z^3``.

        The enumeration uses a bounded integer box and a stable rational
        grid key. In the row convention ``K = k @ M.T``, the fractional
        fold vectors solve ``M @ t = n``. Physical path-sector matching
        itself uses ``tol_k`` in :meth:`compute`.
        """
        matrix = np.asarray(unfold_sc_mat, dtype=int)
        if matrix.shape != (3, 3):
            raise ValueError("unfold_sc_mat must have shape (3, 3)")
        nfold = int(abs(round(np.linalg.det(matrix))))
        if nfold == 0:
            raise ValueError("unfold_sc_mat must be nonsingular")

        # Every quotient class has a short integer representative. Grow a
        # small positive box until all |det M| sectors have been found.
        reps = {}
        for bound in range(0, max(nfold, int(np.abs(matrix).max())) + 2):
            for n in product(range(bound + 1), repeat=3):
                frac = np.mod(np.linalg.solve(matrix, np.asarray(n, dtype=float)), 1.0)
                key = tuple(np.rint(frac * 1e12).astype(np.int64) % 10**12)
                reps.setdefault(key, frac)
            if len(reps) == nfold:
                break
        if len(reps) != nfold:
            raise RuntimeError("could not enumerate all reciprocal fold sectors")

        zero = (0, 0, 0)
        values = list(reps.values())
        values.sort(key=lambda value: (not np.allclose(value, 0.0), *value.tolist()))
        assert tuple(np.rint(values[0] * 1e12).astype(np.int64) % 10**12) == zero
        return np.asarray(values)

    @staticmethod
    def _periodic_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5

    def _stored_kpoint(self, kpoint: np.ndarray) -> int:
        target = np.mod(kpoint @ self.unfold_sc_mat.T, 1.0)
        delta = self._periodic_delta(self.eigendata.kpoints, target)
        distances = np.linalg.norm(delta, axis=1)
        ik = int(np.argmin(distances))
        if distances[ik] > self.tol_k:
            raise ValueError(
                "target SC k-point "
                f"{target.tolist()} has no stored match within tol_k={self.tol_k:g}; "
                f"nearest stored {self.eigendata.kpoints[ik].tolist()} "
                f"(periodic distance {distances[ik]:.3g})"
            )
        return ik

    def compute(self, kpoints, spin: int = 0, resolve_degenerate=None) -> PWWeights:
        """Compute weights for primitive fractional ``kpoints``.

        Each path point uses the matching stored SC k-point and sums the
        coefficients whose primitive fold label equals that path point.
        ``spin`` selects a collinear channel; the spinor axis is always
        traced, as required for SOC/noncollinear eigendata.

        ``resolve_degenerate`` (tolerance in the native eigenvalue unit,
        i.e. Hartree for WFK data) reassigns gauge-invariant weights inside
        each near-degenerate energy group. Degenerate bands may be stored
        in any unitary mixture of their fold sectors: the diagonal weights
        then split arbitrarily between band indices from one k-point to the
        next, which renders as dotted or fragmented weight-coded lines.
        Eigen-assigning the sector-projector Gram matrix within each group
        restores branch weights that are invariant under that mixing
        (for a pristine sector the group eigenvalues are again 0/1).
        """
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.ndim == 1:
            kpoints = kpoints[None, :]
        if kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (nk, 3)")
        if not 0 <= spin < self.eigendata.nspin:
            raise ValueError(f"spin must be in [0, {self.eigendata.nspin})")
        if resolve_degenerate is not None and resolve_degenerate < 0:
            raise ValueError("resolve_degenerate must be non-negative")

        nband = self.eigendata.eigenvalues[0].shape[1]
        if any(eig.shape[1] != nband for eig in self.eigendata.eigenvalues):
            raise ValueError("all stored k-points must have the same nband")

        eig = np.empty((len(kpoints), nband))
        weights = np.empty_like(eig)
        sc_kpoints = np.empty_like(kpoints)
        folds = np.mod(kpoints, 1.0)
        inv_transpose = np.linalg.inv(self.unfold_sc_mat.T)

        for i, (kpoint, fold) in enumerate(zip(kpoints, folds)):
            ik = self._stored_kpoint(kpoint)
            K = self.eigendata.kpoints[ik]
            gvec = self.eigendata.gvecs[ik]
            coeff = self.eigendata.coefficients[ik][spin]
            # Row convention A_sc = M @ A_prim gives primitive reciprocal
            # coordinates w = (K + n) @ M^-T (solve M @ w_col = f_col).
            wavevectors = (K + gvec) @ inv_transpose
            delta = self._periodic_delta(np.mod(wavevectors, 1.0), fold)
            sector = np.linalg.norm(delta, axis=1) <= self.tol_k
            eig[i] = self.eigendata.eigenvalues[ik][spin]
            projected = coeff[:, :, sector].reshape(coeff.shape[0], -1)
            gram = projected @ projected.conj().T
            weights[i] = gram.real.diagonal()
            if resolve_degenerate:
                order = np.argsort(eig[i], kind="stable")
                energies = eig[i][order]
                start = 0
                for stop in range(1, nband + 1):
                    if stop == nband or energies[stop] - energies[stop - 1] > resolve_degenerate:
                        idx = order[start:stop]
                        if stop - start > 1:
                            block_ev = np.linalg.eigvalsh(gram[np.ix_(idx, idx)])
                            weights[i][idx] = np.clip(np.sort(block_ev)[::-1], 0.0, 1.0)
                        start = stop
            sc_kpoints[i] = K

        return PWWeights(kpoints, eig, weights, folds, sc_kpoints)
