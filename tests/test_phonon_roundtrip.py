"""Pristine-supercell round-trip oracles for phonon unfolding.

For a pristine supercell every supercell eigenvector is a perfect Bloch-like
mode of the primitive cell folded with some reciprocal vector G, so:

1. (completeness)  sum over the full set of 3x3x3 unfolding sectors G of
   w(G) = 1 exactly, for every branch and every commensurate q-point.
2. (pristine sectors)  branches are gauge-mixed inside frequency-degenerate
   groups (LAPACK returns arbitrary rotations), so per-branch max_G w = 1 is
   NOT gauge invariant. The gauge-invariant pristine signature: for each
   frequency-degenerate group, the sector sums  sum_{branches in group} w(G)
   are integers, and their total equals the group size.
"""
import numpy as np
import pytest

# the 27 reciprocal-lattice triples {0,1,2}^3: the complete set of unfolding
# sectors for a 3x3x3 supercell at commensurate q-points
G_GRID_333 = np.array(
    [[a, b, c] for a in range(3) for b in range(3) for c in range(3)], dtype=float
)

ATOL = 1e-10


def test_completeness_sum_rule_all_branches(cu_pristine):
    """sum_G w(G) = 1 for every branch, on a subset of commensurate q-points."""
    uf = cu_pristine["unfolder"]
    qs = cu_pristine["qpoints"]
    evecs = cu_pristine["eigenvectors"]
    for iq in (0, 1, 4, 13):  # Gamma + generic grid points
        for branch in range(evecs.shape[1]):
            total = sum(
                uf.get_weight(evecs[iq, :, branch], qs[iq], G) for G in G_GRID_333
            )
            assert total == pytest.approx(1.0, abs=ATOL), (iq, branch, total)


def test_completeness_sum_rule_gamma_all_81(cu_pristine):
    uf = cu_pristine["unfolder"]
    evecs = cu_pristine["eigenvectors"]
    q0 = cu_pristine["qpoints"][0]
    totals = np.array(
        [
            sum(uf.get_weight(evecs[0, :, b], q0, G) for G in G_GRID_333)
            for b in range(evecs.shape[1])
        ]
    )
    assert np.allclose(totals, 1.0, atol=ATOL)


def test_pristine_sector_sums_are_integers_at_gamma(cu_pristine):
    """Degenerate-group sector sums are integers summing to the group size."""
    uf = cu_pristine["unfolder"]
    freqs = cu_pristine["frequencies"]
    evecs = cu_pristine["eigenvectors"]
    q0 = cu_pristine["qpoints"][0]

    order = np.argsort(freqs[0])
    groups, current = [], [order[0]]
    for idx in order[1:]:
        if freqs[0][idx] - freqs[0][current[-1]] < 1e-6:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    for group in groups:
        sector = np.array(
            [
                sum(uf.get_weight(evecs[0, :, b], q0, G) for b in group)
                for G in G_GRID_333
            ]
        )
        assert np.allclose(sector, np.round(sector), atol=ATOL), sector
        assert sector.sum() == pytest.approx(float(len(group)), abs=ATOL)
