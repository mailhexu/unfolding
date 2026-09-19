#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter

def negative_to_zero(x):
    # x is a numpy array, set negative values to zero
    x[x < 0] = 0
    x[x> 1] = 1
    return x

def plot_band_weight(kslist,
                     ekslist,
                     wkslist=None,
                     efermi=0,
                     yrange=None,
                     output=None,
                     style='alpha',
                     color='blue',
                     axis=None,
                     width=2,
                     xticks=None,
                     title=None,
                     ylabel='Frequency (cm$^{-1}$)',
                     ypad=66.0):
    """Draw weight-coded bands on axes `axis` (created when None).

    All styling is applied through the Axes object, so passing an embedded
    `axis` behaves the same as the freshly created one. `ypad` is the fixed
    margin added below/above the energy range (66 cm^-1 by default; pass a
    value in the units of `ekslist`, e.g. a few eV for electron bands), and
    `ylabel` labels the y axis.
    """
    if axis is None:
        fig, a = plt.subplots()
        plt.tight_layout(pad=2.19)
        plt.axis('tight')
        plt.gcf().subplots_adjust(left=0.17)
    else:
        a = axis
    if title is not None:
        a.set_title(title)

    xmax = max(kslist[0])
    if yrange is None:
        yrange = (np.array(ekslist).flatten().min() - ypad,
                  np.array(ekslist).flatten().max() + ypad)

    wkslist=negative_to_zero(np.array(wkslist))

    if wkslist is not None:
        for i in range(len(kslist)):
            x = kslist[i]
            y = ekslist[i]
            lwidths = np.array(wkslist[i]) * width
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if style == 'width':
                lc = LineCollection(segments, linewidths=lwidths, colors=color)
            elif style == 'alpha':
                alphas = [np.abs(lwidth / (width + 0.011))
                        for lwidth in lwidths
                    ]
                lc = LineCollection(
                    segments,
                    linewidths=[2] * len(x),
                    colors=[
                        colorConverter.to_rgba(
                            color, alpha=np.abs(lwidth / (width + 0.011)))
                        for lwidth in lwidths
                    ])

            a.add_collection(lc)
    a.set_ylabel(ylabel)
    for ks, eks in zip(kslist, ekslist):
        a.plot(ks, eks, color='gray', linewidth=0.001)
    a.set_xlim(0, xmax)
    a.set_ylim(yrange)
    if xticks is not None:
        a.set_xticks(xticks[1])
        a.set_xticklabels(xticks[0])
        for x in xticks[1]:
            a.axvline(x, color='gray', linewidth=0.5)
    if efermi is not None:
        a.axhline(linestyle='--', color='black')
    return a
