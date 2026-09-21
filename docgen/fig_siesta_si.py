#!/usr/bin/env python
"""Regenerate the SIESTA Si unfolded-band figure (docs/static/images/).

Unfolds the committed 8-atom conventional-cell Si supercell onto the
primitive high-symmetry path with the generic-k ideal weight, and
overlays the independently computed primitive-cell band structure.
"""
import os
import sys

import numpy as np
import si_siesta_common as common


def main(out_path):
    import matplotlib.pyplot as plt

    from unfolding.plotphon import plot_band_weight

    prim, sc = common.si_models()
    unf = common.make_unfolder(sc, prim)
    kpts, x, Xq, knames = common.band_path(prim.atoms)

    res = unf.compute(kpts, method="ideal")
    eprim = common.prim_bands(prim, kpts)

    nb = res.weights.shape[1]
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(res.eigenvalues[:, ib]) for ib in range(nb)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)]

    ax = plot_band_weight(
        kslist,
        ekslist,
        wkslist,
        xticks=[knames, Xq],
        ylabel="Energy (eV)",
        ypad=1.5,
    )
    lines = ax.plot(x, eprim, color="crimson", lw=1.0, alpha=0.9, zorder=5)
    lines[0].set_label("primitive-cell bands")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


if __name__ == "__main__":

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        common.ROOT, "docs", "static", "images", "si_unfolded.png")
    print(main(out))
