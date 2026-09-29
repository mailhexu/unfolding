"""Story-040: flexible plotting over parsed datasets."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from unfolding.dataset import Dataset


def _dataset(spin=1, fermi=None):
    rng = np.random.default_rng(3)
    nk, nb = 6, 4
    eig = np.sort(rng.normal(size=(nk, nb) if spin == 1
                             else (spin, nk, nb)), axis=-1)
    return Dataset(
        route="test", kpoints=rng.normal(size=(nk, 3)),
        eigenvalues=eig, weights=rng.random(eig.shape),
        spin_channels=spin, fermi_energy=fermi,
        energy_unit="eV")


def test_default_creates_axes_and_draws():
    ds = _dataset()
    ax = plt.gca()
    from unfolding import plot_dataset

    out = plot_dataset(ds)
    assert out is ax or out is not None
    assert len(out.collections) > 0
    assert out.get_ylabel() == "Energy (eV)"
    plt.close("all")


def test_user_axes_subplot_composition():
    from unfolding import plot_dataset

    ds = _dataset()
    fig, (a1, a2) = plt.subplots(1, 2)
    r1 = plot_dataset(ds, ax=a1)
    r2 = plot_dataset(ds, ax=a2, color="green")
    assert r1 is a1 and r2 is a2
    assert a1.collections and a2.collections
    plt.close(fig)


def test_spin_channel_selection():
    from unfolding import plot_dataset

    ds = _dataset(spin=2)
    both = plot_dataset(ds)
    one = plot_dataset(ds, spin=0)
    # both channels draw twice the collections of one channel
    assert len(both.collections) == 2 * len(one.collections)
    plt.close("all")


def test_fermi_at_zero_shifts():
    from unfolding import plot_dataset

    ds = _dataset(fermi=-0.5)
    ax = plot_dataset(ds, fermi_at_zero=True)
    # every band point appears as a (doubled) segment endpoint
    ys = np.concatenate([np.asarray(c.get_segments())[:, :, 1].ravel()
                         for c in ax.collections])
    np.testing.assert_allclose(
        np.unique(ys), np.unique((ds.eigenvalues + 0.5).ravel()), atol=1e-9)
    plt.close("all")


def test_fermi_at_zero_requires_stored_fermi():
    from unfolding import plot_dataset

    with pytest.raises(ValueError, match="fermi"):
        plot_dataset(_dataset(fermi=None), fermi_at_zero=True)


def test_overlay_requires_explicit_shift():
    from unfolding import plot_dataset

    ds = _dataset()
    x = np.arange(6, dtype=float)
    ref = ds.eigenvalues.T  # (nband, nk)
    with pytest.raises(ValueError, match="overlay_shift"):
        plot_dataset(ds, overlay=(x, ref))
    ax = plot_dataset(ds, overlay=(x, ref), overlay_shift=-0.25)
    lines = ax.get_lines()
    assert lines  # reference bands drawn as plain lines
    ref_lines = lines[-ref.shape[0]:]  # overlay lines are appended last
    np.testing.assert_allclose(
        np.asarray([ln.get_ydata() for ln in ref_lines]),
        ref - 0.25, atol=1e-12)
    plt.close("all")


def test_overlay_dataset_input():
    from unfolding import plot_dataset

    ds, ref = _dataset(), _dataset()
    ax = plot_dataset(ds, overlay=ref, overlay_shift=0.1)
    assert ax.get_lines()
    plt.close("all")


def test_documented_multipanel_recipe(tmp_path):
    """The documented <=15-line primitive-comparison recipe (story 040)."""
    import matplotlib.pyplot as plt

    from unfolding import plot_dataset

    ds = _dataset(fermi=-0.5)
    prim = _dataset()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    plot_dataset(ds, ax=ax1, fermi_at_zero=True)
    plot_dataset(ds, ax=ax2, fermi_at_zero=True,
                 overlay=prim, overlay_shift=-0.13)
    fig.savefig(tmp_path / "multipanel.png", dpi=80)
    assert ax1.collections and ax2.collections and ax2.get_lines()


def _ds(**kw):
    return _dataset(**kw)


def test_nan_weights_render_at_zero_weight():
    from unfolding import plot_dataset

    ds = _dataset()
    w = ds.weights.copy()
    w[0, 0] = np.nan
    object.__setattr__(ds, "weights", w)
    ax = plot_dataset(ds)  # must not raise on alpha=nan
    alphas = np.concatenate([
        np.asarray(c.get_colors())[:, 3] for c in ax.collections])
    assert not np.isnan(alphas).any()
    plt.close("all")


def test_electronic_ypad_default():
    from unfolding import plot_dataset

    ax = plot_dataset(_dataset())  # eV
    span = np.ptp(np.asarray(ax.get_ylim()))
    assert span < 30  # a phonon-scale 2*66 margin would dwarf these bands
    plt.close("all")


def test_square_dataset_overlay_transposed():
    """nk == nband overlay datasets must follow bands, not k-columns."""
    from unfolding import plot_dataset

    ds = _dataset()  # 6 k, 4 bands
    square = Dataset(
        route="ref", kpoints=np.zeros((4, 3)),
        eigenvalues=np.arange(16, dtype=float).reshape(4, 4),
        weights=np.ones((4, 4)))
    ax = plot_dataset(ds, overlay=square, overlay_shift=0.0)
    lines = np.asarray([ln.get_ydata() for ln in ax.get_lines()[-4:]])
    # eigenvalues[k, b] = 4k + b: band b over k is the column [b, b+4, ...]
    np.testing.assert_allclose(lines[0], [0.0, 4.0, 8.0, 12.0])
    np.testing.assert_allclose(lines[3], [3.0, 7.0, 11.0, 15.0])
    plt.close("all")


def test_spin_both_shares_vertical_range():
    from unfolding import plot_dataset

    rng = np.random.default_rng(5)
    nk, nb = 5, 3
    eig = np.stack([rng.normal(size=(nk, nb)),
                    rng.normal(size=(nk, nb)) + 10.0])  # disjoint channels
    ds = Dataset(route="t", kpoints=rng.normal(size=(nk, 3)),
                 eigenvalues=eig, weights=rng.random(eig.shape),
                 spin_channels=2)
    ax = plot_dataset(ds, ypad=0.5)
    lo, hi = ax.get_ylim()
    assert lo < eig.min() and hi > eig[1].max()  # both channels visible
    plt.close("all")



def test_fermi_auto_follows_energy_reference():
    from unfolding import plot_dataset

    rng = np.random.default_rng(11)
    nk, nb = 5, 3
    stored = rng.normal(size=(nk, nb))  # absolute energies
    fermi = -0.5
    kw = dict(route="t", kpoints=rng.normal(size=(nk, 3)),
              weights=rng.random((nk, nb)), fermi_energy=fermi)

    ax = plot_dataset(Dataset(eigenvalues=stored,
                              energy_reference="absolute", **kw))
    ys = np.unique(np.concatenate(
        [np.asarray(c.get_segments())[:, :, 1].ravel()
         for c in ax.collections]))
    np.testing.assert_allclose(ys, np.unique((stored - fermi).ravel()),
                               atol=1e-9)

    # "fermi" reference: no double shift even with fermi_energy present
    ax2 = plot_dataset(Dataset(eigenvalues=stored,
                               energy_reference="fermi", **kw))
    ys2 = np.unique(np.concatenate(
        [np.asarray(c.get_segments())[:, :, 1].ravel()
         for c in ax2.collections]))
    np.testing.assert_allclose(ys2, np.unique(stored.ravel()), atol=1e-9)
    plt.close("all")
