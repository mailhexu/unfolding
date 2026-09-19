"""Headless smoke tests for the weighted-band plotting helper."""
import matplotlib


matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.collections import LineCollection

from unfolding.plotphon import plot_band_weight


def _synthetic_bands(nk=40, nbands=3):
    k = np.linspace(0.0, 1.0, nk)
    kslist = [k] * nbands
    ekslist = [np.sin(2 * np.pi * k + i) for i in range(nbands)]
    wkslist = [0.5 + 0.5 * np.cos(2 * np.pi * k + i) for i in range(nbands)]
    return kslist, ekslist, wkslist


@pytest.mark.parametrize("style", ["alpha", "width"])
def test_plot_returns_axes_with_lines(style):

    kslist, ekslist, wkslist = _synthetic_bands()
    ax = plot_band_weight(kslist, ekslist, wkslist, style=style, xticks=[["G", "X"], [0, 1]])
    collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(collections) == 3  # one per band


def test_plot_clips_out_of_range_weights():
    kslist, ekslist, wkslist = _synthetic_bands()
    bad = [w * 3.0 - 2.0 for w in wkslist]  # weights in [-2, 1] -> needs clipping
    ax = plot_band_weight(kslist, ekslist, bad, style="alpha")
    collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(collections) == 3
    for lc in collections:
        colors = np.array(lc.get_colors())
        assert colors.shape[0] > 0
        assert (colors[:, 3] >= 0.0).all() and (colors[:, 3] <= 1.0).all()
