#!/usr/bin/env python
"""ABACUS Si example: unfold the 8-atom conventional cell (LCAO).

Uses the committed fixtures under tests/data/abacus_example (Si DZP
out_mat_hs2 runs) and the shared Si band-path helper. Writes
docs/static/images/abacus_si_unfolded.png (pristine) and
abacus_si_p_doped.png (one Si substituted by P).

Usage: python examples/abacus_si/unfold.py [docs/static/images]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docgen"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import si_siesta_common as common  # the shared Si band-path helper


def read_abacus_model(name):
    from HamiltonIO.abacus.abacus_wrapper import AbacusParser

    outpath = os.path.join(common.ROOT, "tests", "data", "abacus_example",
                           name, "OUT." + name)
    return AbacusParser(outpath=outpath).get_models()


def make_unfolder(sc_model, p_doped=False):
    import numpy as np

    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = read_abacus_model("si_prim")
    counts = [13] * 8  # DZP 2s2p1d
    rm = RelabelMap.from_atoms(
        sc_model.atoms, prim.atoms, common.B,
        orb_counts_sc=counts, orb_counts_prim=[13, 13],
        match_species=not p_doped,
    )
    return LCAOUnfolder(HamiltonIOModel(sc_model), rm), prim


def draw(unf, prim, efermi, out_path, title=None):
    import numpy as np

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from unfolding.plotphon import plot_band_weight

    kpts, x, Xq, knames = common.band_path(prim.atoms)
    res = unf.compute(kpts, method="ideal")
    eprim = common.prim_bands(read_abacus_model("si_prim"), kpts)

    # house style: energies in eV, zero at the Fermi level of the run
    # being unfolded; the same shift applies to the prim reference
    nb = res.weights.shape[1]
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(res.eigenvalues[:, ib] - efermi) for ib in range(nb)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)]

    ax = plot_band_weight(
        kslist, ekslist, wkslist,
        xticks=[knames, Xq],
        ylabel="Energy (eV)",
        ypad=1.5,
        efermi=0.0,
        yrange=(-13.0, 8.0),
    )
    lines = ax.plot(x, eprim - efermi, color="crimson", lw=1.0, alpha=0.9,
                    zorder=5)
    lines[0].set_label("primitive-cell bands")
    if title:
        ax.set_title(title)
    # plot_band_weight draws the weight-coded bands as blue lines with
    # alpha = w/(width + 0.011); the proxy matches the weight-1 appearance.
    unfolded = Line2D([0], [0], color="blue", alpha=0.5, lw=2,
                      label="unfolded spectral weight")
    ax.legend(handles=[unfolded, lines[0]], loc="upper right", fontsize=8,
              framealpha=0.85)
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


def main(images_dir):
    sc = read_abacus_model("si_conv")
    unf, prim = make_unfolder(sc)
    p1 = draw(unf, prim, sc.efermi,
              os.path.join(images_dir, "abacus_si_unfolded.png"))
    sc7p = read_abacus_model("si7p")
    unf7, _ = make_unfolder(sc7p, p_doped=True)
    p2 = draw(unf7, prim, sc7p.efermi,
              os.path.join(images_dir, "abacus_si_p_doped.png"))
    return p1, p2


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "docs", "static", "images")
    os.makedirs(out_dir, exist_ok=True)
    print(main(out_dir))
