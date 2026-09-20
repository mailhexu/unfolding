#!/usr/bin/env python
"""Regenerate the WFSX-driven Si unfolded-band figure.

Same path and physics as fig_siesta_si.py, but the eigenvalues and
coefficients come from the committed SIESTA WFSX path run instead of
diagonalizing H: the weight formula still uses the HSX overlap shells,
only the eigen-solve is replaced by SIESTA's own wavefunctions. The
overlay is the independently computed primitive-cell band structure
(raw energies); WFSX energies are Fermi-shifted, so the run's
EIG-header Fermi energy is subtracted here.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

import si_siesta_common as common


def main(out_path):
    import matplotlib.pyplot as plt

    from HamiltonIO.siesta.wfsx import SiestaWFSXParser

    from unfolding.plotphon import plot_band_weight
    from unfolding.wfsx_unfolder import WFSXUnfolder

    prim, sc = common.si_models()
    unf_hs = common.make_unfolder(sc, prim)

    data_dir = os.path.join(common.ROOT, "tests", "data", "si_example")
    cell = np.asarray(sc.atoms.cell)
    wfsx = SiestaWFSXParser(
        os.path.join(data_dir, "si_sc_path.selected.WFSX"), cell=cell
    ).read()

    with open(os.path.join(data_dir, "si_sc_path.EIG")) as fh:
        e_fermi = float(fh.readlines()[0])

    unf = WFSXUnfolder(wfsx, unf_hs._model, unf_hs._rm, common.B)
    kprim, _starts = common.path_grid(npts=150)
    res = unf.compute(kprim, method="ideal")

    bcart = 2 * np.pi * np.linalg.inv(np.asarray(prim.atoms.cell)).T
    x = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(kprim, axis=0) @ bcart, axis=1))]
    )
    _, seg_starts = common.path_grid(npts=150)
    Xq = [x[i] for i in seg_starts] + [x[-1]]
    knames = [{"G": "Γ"}.get(s, s) for s in common.PATH]
    eprim = common.prim_bands(prim, kprim)

    nb = res.weights.shape[1]
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(res.eigenvalues[:, ib] - e_fermi) for ib in range(nb)]
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
        common.ROOT, "docs", "static", "images", "si_wfsx_unfolded.png")
    print(main(out))
