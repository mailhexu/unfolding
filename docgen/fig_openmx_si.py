"""Regenerate the OpenMX Si / Si:P unfolded-band figures.

Same setup as fig_siesta_p_doped.py but with the committed OpenMX
fixtures (tests/data/si_example/openmx_*.scfout, Si7.0-s2p2d1 basis).
The pristine Si8 unfolding validates the OpenMX phase/unit conventions:
every ideal weight is exactly 0/1 and the weight-1 branches lie on the
independent primitive-cell bands (red overlay). For Si7P the dopant
site maps onto the host site (match_species=False), so donor-derived
states appear with reduced weight.
Energies are referenced to the unfolded run's own Fermi level (OpenMX
``ChemP``: pristine panel uses the Si8 ``ChemP``, doped panel the Si7P
``ChemP``); the primitive overlay uses the same shift as its panel, so
0 in every figure is E_F.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docgen"))

DATA = os.path.join(ROOT, "tests", "data", "si_example")
B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = B @ prim

import si_siesta_common as common  # noqa: E402  (band path helpers)


def _model(name):
    from HamiltonIO.openmx import OpenmxParser

    return OpenmxParser(os.path.join(DATA, name)).get_model()


def make_unfolder(sc_name, match_species=True):
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = _model("openmx_si_prim.scfout")
    sc = _model(sc_name)
    rm = RelabelMap.from_atoms(
        sc.atoms,
        prim.atoms,
        B,
        orb_counts_sc=[13] * 8,
        orb_counts_prim=[13, 13],
        match_species=match_species,
    )
    # Fermi level (eV) of the unfolded supercell run: each panel is
    # referenced to its own run's OpenMX ChemP (mirrors the SIESTA
    # figures, whose eigenvalues already carry E_F = 0). The primitive
    # overlay is shifted by the same value so weight-1 host branches
    # coincide with it.
    ef = sc.efermi
    return prim, LCAOUnfolder(HamiltonIOModel(sc), rm), ef


def main(sc_name, out_path, match_species=True, atol=1e-6):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from unfolding.plotphon import plot_band_weight

    prim, unf, ef = make_unfolder(sc_name, match_species)
    kpts, x, Xq, knames = common.band_path(prim.atoms)

    res = unf.compute(kpts, method="ideal", atol_orth=atol, atol_imag=atol)
    res.eigenvalues = res.eigenvalues - ef  # 0 in the figure is E_F
    eprim = common.prim_bands(prim, kpts) - ef

    nb = res.weights.shape[1]
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(res.eigenvalues[:, ib]) for ib in range(nb)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)]

    ax = plot_band_weight(
        kslist,
        ekslist,
        wkslist,
        efermi=ef,
        yrange=(-13.0, 8.0),  # same window as the SIESTA figures
        xticks=[knames, Xq],
        ylabel="Energy (eV)",
        ypad=1.5,
    )
    lines = ax.plot(x, eprim, color="crimson", lw=1.0, alpha=0.9, zorder=5)
    lines[0].set_label("OpenMX primitive-cell bands")
    # plot_band_weight draws the weight-coded bands as blue lines with
    # alpha = w/(width + 0.011); the proxy matches the weight-1 appearance.
    unfolded = Line2D([0], [0], color="blue", alpha=0.5, lw=2,
                      label="unfolded spectral weight")
    ax.legend(handles=[unfolded, lines[0]], loc="upper right", fontsize=8,
              framealpha=0.85)
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        common.ROOT, "docs", "static", "images")
    print(main(
        "openmx_si_sc.scfout",
        os.path.join(out, "openmx_si_unfolded.png"),
    ))
    print(main(
        "openmx_si_sc_p.scfout",
        os.path.join(out, "openmx_si_p_doped.png"),
        match_species=False,
    ))
