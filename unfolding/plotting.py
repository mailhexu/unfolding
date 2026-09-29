"""Flexible plotting over parsed unfolded-band datasets (story 040, ADR-014).

Reshapes a :class:`~unfolding.dataset.Dataset` into the argument lists of
:func:`unfolding.plotphon.plot_band_weight`; rendering stays in that one
place. Composability: pass your own ``ax`` for subplot layouts, and
``overlay`` to draw primitive reference bands next to the unfolded
weights.

    import matplotlib.pyplot as plt
    from unfolding import load_dataset, plot_dataset

    ds = load_dataset("bands.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plot_dataset(ds, ax=ax1)
    plot_dataset(ds, ax=ax2, overlay=(x_prim, e_prim), overlay_shift=-0.13)
"""
from __future__ import annotations

import numpy as np

__all__ = ["plot_dataset"]


def _channel_arrays(dataset, spin):
    """(energy, weight) column iterators for the selected channel(s)."""
    eig = dataset.eigenvalues
    w = dataset.weights
    if dataset.spin_channels == 2:
        channels = range(2) if spin == "both" else [int(spin)]
        for ch in channels:
            yield eig[ch], w[ch]
    else:
        if spin not in ("both", 0, None):
            raise ValueError(
                f"spin={spin!r}: this dataset has a single channel")
        yield eig, w


def plot_dataset(dataset, *, ax=None, style="alpha", color="blue",
                 ylabel=None, x=None, spin="both", fermi_at_zero=False,
                 overlay=None, overlay_shift=None, overlay_color="crimson",
                 overlay_lw=1.0, **kwargs):
    """Plot weight-coded unfolded bands from a parsed dataset.

    Parameters
    ----------
    dataset : unfolding.dataset.Dataset
        Parsed dataset (``load_dataset``).
    ax : matplotlib Axes, optional
        Draw into an existing axes (subplot composition); a new single
        axes is created when omitted.
    style, color, **kwargs
        Forwarded to :func:`unfolding.plotphon.plot_band_weight`.
    ylabel : str, optional
        Defaults to ``"Energy (<energy_unit>)"``.
    x : (nk,) array-like, optional
        Path coordinates; defaults to integer point indices.
    spin : {"both", 0, 1}
        Channel selection for two-channel datasets.
    fermi_at_zero : bool
        Subtract ``dataset.fermi_energy`` before plotting (requires a
        dataset that stores it; datasets whose eigenvalues are already
        E_F-referenced say so in ``energy_reference``).
    overlay : (x, energies) tuple or Dataset, optional
        Primitive reference bands drawn as thin lines. ``energies`` is
        (nband, nk) or (nk, nband); a Dataset uses its eigenvalues (and
        ``x`` defaults to its own point indices).
    overlay_shift : float, optional
        Rigid eV shift applied to the overlay energies. **Required**
        with ``overlay``: alignment between two runs is a physical
        decision (e.g. the median potential offset), never inferred.
    overlay_color, overlay_lw
        Styling of the reference lines.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from .plotphon import plot_band_weight

    if overlay is not None and overlay_shift is None:
        raise ValueError(
            "overlay needs an explicit overlay_shift (in the dataset's "
            "energy unit); aligning two runs is a physical decision")

    nk = len(dataset.kpoints)
    if x is None:
        x = np.arange(nk, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.shape != (nk,):
        raise ValueError("x must supply one coordinate per k-point")

    if fermi_at_zero:
        if dataset.fermi_energy is None:
            raise ValueError(
                "fermi_at_zero: dataset has no fermi_energy stored")
    if ylabel is None:
        ylabel = f"Energy ({dataset.energy_unit})"

    e_shift = -float(dataset.fermi_energy) if fermi_at_zero else 0.0
    for eig, w in _channel_arrays(dataset, spin):
        nb = eig.shape[1]
        eig = eig + e_shift
        ax = plot_band_weight(
            [x] * nb,
            [eig[:, ib] for ib in range(nb)],
            [np.clip(w[:, ib], 0.0, 1.0) for ib in range(nb)],
            axis=ax, style=style, color=color, ylabel=ylabel, **kwargs)


    if overlay is not None:
        if isinstance(overlay, tuple):
            ox, oe = overlay
            ox = np.asarray(ox, dtype=float)
            oe = np.asarray(oe, dtype=float)
        else:  # Dataset
            ox = np.arange(len(overlay.kpoints), dtype=float)
            oe = overlay.eigenvalues
        if oe.ndim == 2 and oe.shape[0] == len(ox) and oe.shape[1] != len(ox):
            oe = oe.T  # (nk, nband) -> (nband, nk) rows
        oshift = float(overlay_shift)
        for row in oe:
            ax.plot(ox, row + oshift, color=overlay_color, lw=overlay_lw)
    ax.set_xlim(x.min(), x.max())
    return ax
