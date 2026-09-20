#!/usr/bin/env python
"""Regenerate the Si7P substitutional-dopant unfolded-band figure.

Same setup as fig_siesta_si.py but with the committed Si7P supercell
(one Si replaced by P, 2x2x2 k-grid SCF). The dopant site maps onto
the host site it replaces (match_species=False): the ideal weight then
measures resemblance to the ideal Si crystal, so host bands stay at
weight 1 while donor-derived states appear with reduced weight.
"""
import os
import sys

import si_siesta_common as common


def main(out_path):
    import matplotlib.pyplot as plt

    from unfolding.plotphon import plot_band_weight

    prim, sc = common.si_models(sc_fdf="si_sc_p.fdf")
    unf = common.make_unfolder(sc, prim, match_species=False)
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
    lines[0].set_label("Si primitive-cell bands")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


if __name__ == "__main__":
    import numpy as np

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        common.ROOT, "docs", "static", "images", "si_p_doped_unfolded.png")
    print(main(out))
