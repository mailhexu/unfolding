"""Unit tests for the shared degeneracy helpers (story 047)."""
import numpy as np
import pytest

from unfolding.degeneracy import (assign_eigvalsh_groups, assign_total_groups,
                                  energy_groups)


def test_energy_groups_boundaries_and_empty():
    e = np.array([0.0, 0.0 + 1e-9, 0.5, 0.5, 1.0])
    groups = energy_groups(e, 1e-6)
    assert groups == [(0, 2), (2, 4), (4, 5)]
    assert energy_groups([], 1e-6) == []
    assert energy_groups([3.0], 1e-6) == [(0, 1)]
    # gap exactly at tolerance stays grouped; above splits
    assert energy_groups([0.0, 1e-6], 1e-6) == [(0, 2)]
    assert energy_groups([0.0, 2e-6], 1e-6) == [(0, 1), (1, 2)]


def test_energy_groups_sorted_order_maps_back():
    e = np.array([5.0, 0.0, 0.0 + 1e-9, 2.0])
    order = np.argsort(e, kind="stable")          # [1, 2, 3, 0]
    groups = energy_groups(e, 1e-6, order=order)
    # boundaries over sorted [0, 0, 2, 5]: (0,2), (2,3), (3,4)
    assert groups == [(0, 2), (2, 3), (3, 4)]


def test_assign_eigvalsh_rotation_invariant_and_sorted_desc():
    rng = np.random.default_rng(7)
    n = 6
    e = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0])   # triple + double + single
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = H @ H.conj().T
    w1 = np.real(np.diag(H)).copy()
    assign_eigvalsh_groups(w1, H, e, 1e-6)
    # random unitary within each degenerate group leaves group spectra fixed
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    Hr = H.copy()
    Hr[:3, :3] = q @ H[:3, :3] @ q.conj().T
    w2 = np.real(np.diag(Hr)).copy()
    assign_eigvalsh_groups(w2, Hr, e, 1e-6)
    assert np.allclose(w1[:3], w2[:3], atol=1e-12)
    assert np.allclose(np.diff(w1[:3]), -np.abs(np.diff(w1[:3])), atol=1e-15)  # desc
    # singleton untouched
    assert w1[5] == np.real(H[5, 5])
    # group spectra match direct eigvalsh
    assert np.allclose(np.sort(w1[:3])[::-1],
                       np.sort(np.linalg.eigvalsh(H[:3, :3]))[::-1], atol=1e-12)


def test_assign_eigvalsh_order_and_clip():
    rng = np.random.default_rng(3)
    e = np.array([1.0, 0.0, 0.0])                  # degenerate pair at the end
    order = np.argsort(e, kind="stable")           # [1, 2, 0]
    H = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    H = H @ H.conj().T
    w = np.zeros(3)
    assign_eigvalsh_groups(w, H, e, 1e-6, order=order, clip=(0.0, 1.0))
    ev = np.sort(np.linalg.eigvalsh(H[np.ix_([1, 2], [1, 2])]))[::-1]
    assert np.allclose(w[[1, 2]], np.clip(ev, 0.0, 1.0), atol=1e-12)
    assert w[0] == 0.0                             # singleton untouched


def test_assign_total_integer_and_fractional():
    n = 5
    e = np.zeros(n)                                # one fully degenerate group
    w = np.zeros(n)
    w[0] = 3.0
    assign_total_groups(w, e, 1e-6)
    assert w.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    wf = np.full(n, 2.5 / n)
    assign_total_groups(wf, e, 1e-6)
    assert np.allclose(wf, 0.5)
    # sorted-order idiom maps back through order
    e2 = np.array([1.0, 0.0, 0.0, 0.0])
    order = np.argsort(e2, kind="stable")
    w2 = np.array([0.0, 0.9, 0.0, 0.1])            # group total 1.0 over idx 1..3
    assign_total_groups(w2, e2, 1e-6, order=order)
    assert w2[0] == 0.0
    assert sorted(w2[1:]) == [0.0, 0.0, 1.0]


def test_phonon_block_weights_parity():
    """assign_eigvalsh_groups reproduces the phonon _block_weights idiom."""
    from unfolding.phonon_unfolder import _block_weights
    rng = np.random.default_rng(11)
    n = 20
    e = np.sort(rng.random(n))
    e[8:12] += 1e-9                                # plant a quadruple
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = H @ H.conj().T / n
    w_ref = _block_weights(H, e, 1e-6)
    order = np.argsort(e, kind="stable")
    w = np.real(np.diag(H)).copy()
    assign_eigvalsh_groups(w, H, e, 1e-6, order=order, clip=(0.0, 1.0))
    w = np.clip(w, 0.0, 1.0)
    assert np.allclose(w, w_ref, atol=1e-14)
