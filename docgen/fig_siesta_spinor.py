#!/usr/bin/env python
"""Regenerate the spinor (nspin=4) Si unfolded-band figure.

Same pristine Si supercell as fig_siesta_si.py but from Spin.Orbit
runs (nspin=4). The scalar Si pseudopotential has no SOC channels, so
the spinor bands equal the scalar bands with Kramers degeneracy - the
figure certifies the spinor machinery end to end (parse -> relabel ->
weights), not spinor physics. Spinor orbital counts are doubled:
8 per atom (4 PAO x 2 spin components).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

import si_siesta_common as common


def main(out_path):
    import matplotlib.pyplot as plt

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap
    from unfolding.plotphon import plot_band_weight

    data_dir = os.path.join(common.ROOT, "tests", "data", "si_example")
    sys.path.insert(0, os.path.join(common.ROOT, "tests"))
    from siesta_helpers import read_si_model

    prim = read_si_model(os.path.join(data_dir, "si_prim_pso.fdf"))
    sc = read_si_model(os.path.join(data_dir, "si_sc_pso.fdf"))

    rm = RelabelMap.from_atoms(
        sc.atoms,
        prim.atoms,
        common.B,
        orb_counts_sc=[8] * 8,   # 4 PAO x 2 spin components
        orb_counts_prim=[8, 8],
    )
    unf = LCAOUnfolder(HamiltonIOModel(sc), rm)

    kpts, x, Xq, knames = common.band_path(prim.atoms)
    res = unf.compute(kpts, method="ideal")

    pm = HamiltonIOModel(prim)
    from scipy.linalg import eigh

    eprim = np.array(
        [eigh(*pm.hs_and_eigen(k), eigvals_only=True) for k in kpts]
    )

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
    lines[0].set_label("spinor primitive-cell bands")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        common.ROOT, "docs", "static", "images", "si_spinor_unfolded.png")
    print(main(out))
