"""Shared degeneracy grouping and weight-assignment helpers (story 047).

Energy grouping, per-group weight assignment, and result-level
aggregation are distinct operations (ADR-019): the helpers here cover
the first two, which are algorithmically identical across engines;
each result class keeps its own public averaging/aggregation behavior.

Two assignment strategies exist:

- Hermitian-block eigenvalue assignment (phonon, PW, Wannier, PAW):
  inside a (near-)degenerate group the stored eigenvectors may be an
  arbitrary unitary mixture, so per-mode weights are gauge-dependent;
  the gauge-invariant group content is the spectrum of the weight
  operator restricted to the group.
- Group-total assignment (magnon BdG): the group total is the
  presentation-stable quantity; integer totals become (1, .., 1, 0,
  .., 0) on the energy-sorted group and fractional totals are split
  equally.

Sorting policy: helpers never sort the energies. Callers that group on
a sorted copy (phonon, PW, magnon) pass ``order`` (the argsort) so
group members map back to their original positions; callers whose
energies are already grouped in place (Wannier, PAW) pass
``order=None`` — which keeps the contiguous operator slice a view
(no np.ix_ copy).
"""
import numpy as np

__all__ = ["energy_groups", "assign_eigvalsh_groups", "assign_total_groups"]


def energy_groups(energies, tol, order=None):
    """(start, stop) boundaries of consecutive near-degenerate energies.

    A new group starts wherever ``energies[stop] - energies[stop-1] >
    tol``. With ``order`` given, the gaps are evaluated on
    ``energies[order]`` (sorted idiom) and the returned pairs index
    into ``order``; without it, the energies are grouped in the given
    order. Empty input yields no groups; a single energy yields one
    group.
    """
    e = np.asarray(energies, dtype=float)
    if order is not None:
        e = e[np.asarray(order)]
    if len(e) == 0:
        return []
    boundaries = np.r_[0, np.flatnonzero(np.diff(e) > tol) + 1, len(e)]
    return [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]


def assign_eigvalsh_groups(weights, operator, energies, tol, *,
                           order=None, clip=None, descending=True):
    """Assign Hermitian-block eigenvalue weights inside degenerate groups.

    ``weights`` is modified in place: every group with more than one
    member receives the eigenvalues of ``operator[group, group]``;
    singletons are left untouched (callers pre-fill diagonals or keep
    existing weights). ``descending=True`` (phonon/PW convention)
    assigns sorted-descending values; ``descending=False`` (PAW
    convention) assigns the eigvalsh output as returned (ascending).
    ``clip=(lo, hi)`` optionally bounds the assigned values. Returns
    ``weights``.
    """
    w = np.asarray(weights)
    for start, stop in energy_groups(energies, tol, order=order):
        idx = (np.arange(start, stop) if order is None
               else np.asarray(order)[start:stop])
        if len(idx) > 1:
            block = (operator[start:stop, start:stop] if order is None
                     else operator[np.ix_(idx, idx)])
            block_ev = np.linalg.eigvalsh(block)
            if descending:
                block_ev = np.sort(block_ev)[::-1]
            if clip is not None:
                block_ev = np.clip(block_ev, *clip)
            w[idx] = block_ev
    return w


def assign_total_groups(weights, energies, tol, *, order=None,
                        int_tol=1e-6):
    """Assign magnon group totals inside degenerate groups (in place).

    The invariant content at one fold is the group's total weight: a
    (numerically) integer total becomes ``k`` ones followed by zeros on
    the energy-sorted group; a fractional total (defect-hybridized
    degeneracies) is split equally. Singletons are left untouched.
    Returns ``weights``.
    """
    w = np.asarray(weights)
    for start, stop in energy_groups(energies, tol, order=order):
        idx = (np.arange(start, stop) if order is None
               else np.asarray(order)[start:stop])
        if len(idx) > 1:
            total = w[idx].sum()
            k = round(total)
            if abs(total - k) < int_tol:
                assign = np.zeros(len(idx))
                assign[:k] = 1.0
            else:
                assign = np.full(len(idx), total / len(idx))
            w[idx] = assign
    return w
