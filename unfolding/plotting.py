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
                 ylabel=None, x=None, spin="both", fermi_at_zero=None,
                 overlay=None, overlay_shift=None, overlay_color="crimson",
                 overlay_lw=1.0, ypad=None, **kwargs):
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
        Channel selection for two-channel datasets; both channels share
        one vertical range.
    fermi_at_zero : bool or None
        ``None`` (default) follows the stored ``energy_reference``:
        absolute datasets that carry a Fermi level are shifted onto
        E_F = 0; datasets whose energies are already E_F-referenced are
        not shifted again. ``True`` forces the shift (requires a stored
        ``fermi_energy``); ``False`` never shifts.
    overlay : (x, energies) tuple or Dataset, optional
        Primitive reference bands drawn as thin lines. Tuple
        ``energies`` is (nband, nk) or (nk, nband); a Dataset is always
        (nk, nband) by schema and handled accordingly.
    overlay_shift : float, optional
        Rigid shift (in the dataset's energy unit) applied to the
        overlay energies. **Required** with ``overlay``: alignment
        between two runs is a physical decision (e.g. the median
        potential offset), never inferred.
    overlay_color, overlay_lw
        Styling of the reference lines.
    ypad : float, optional
        Vertical margin in energy units; defaults per ``energy_unit``
        (a few eV/meV for electronic units, the historical 66 for
        cm^-1 phonon plots).

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

    if fermi_at_zero is None:
        # follow the stored reference convention: shift only absolute
        # energies carrying a Fermi level; never double-shift
        fermi_at_zero = (
            dataset.energy_reference == "absolute"
            and dataset.fermi_energy is not None)
    if fermi_at_zero and dataset.fermi_energy is None:
        raise ValueError("fermi_at_zero: dataset has no fermi_energy stored")
    if ylabel is None:
        ylabel = f"Energy ({dataset.energy_unit})"
    if ypad is None:
        ypad = 66.0 if dataset.energy_unit == "cm^-1" else 1.5

    e_shift = -float(dataset.fermi_energy) if fermi_at_zero else 0.0
    channels = list(_channel_arrays(dataset, spin))
    energies = [np.asarray(eig, dtype=float) + e_shift for eig, _ in channels]
    weights = [np.where(np.isfinite(w), np.clip(w, 0.0, 1.0), 0.0)
               for _, w in channels]  # masked (NaN) bands render at weight 0
    default_yrange = (min(e.min() for e in energies) - ypad,
                      max(e.max() for e in energies) + ypad)
    yrange = kwargs.pop("yrange", default_yrange)
    for eig, w in zip(energies, weights):
        nb = eig.shape[1]
        ax = plot_band_weight(
            [x] * nb,
            [eig[:, ib] for ib in range(nb)],
            [w[:, ib] for ib in range(nb)],
            axis=ax, style=style, color=color, ylabel=ylabel,
            yrange=yrange, ypad=ypad, **kwargs)

    if overlay is not None:
        if isinstance(overlay, tuple):
            ox, oe = overlay
            ox = np.asarray(ox, dtype=float)
            oe = np.asarray(oe, dtype=float)
            if oe.ndim == 2 and oe.shape[0] == len(ox) \
                    and oe.shape[1] != len(ox):
                oe = oe.T  # (nk, nband) -> (nband, nk) rows
        else:  # Dataset: always (nk, nband) by schema
            if overlay.energy_unit != dataset.energy_unit:
                raise ValueError(
                    f"overlay energy unit {overlay.energy_unit!r} differs "
                    f"from dataset unit {dataset.energy_unit!r}")
            ox = np.arange(len(overlay.kpoints), dtype=float)
            oe = np.asarray(overlay.eigenvalues, dtype=float).T
        oshift = float(overlay_shift)
        for row in oe:
            ax.plot(ox, row + oshift, color=overlay_color, lw=overlay_lw)
    ax.set_xlim(x.min(), x.max())
    return ax
