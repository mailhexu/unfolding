#!/usr/bin/env python
"""Reproduce the ABACUS LCAO Si unfolded-band figures (abacus_si_unfolded.png).

Docs example: docs/content/examples/abacus-si.md in the unfolding repo.
Unfolds the committed 8-atom conventional-cubic Si supercell (LCAO run
with `out_mat_hs2 1`, i.e. data-HR-sparse_SPIN0.csr / data-SR-sparse_SPIN0.csr)
onto the 2-atom fcc primitive cell along Gamma-X-W-Gamma-L-W-X, overlaying
the independently computed primitive-cell bands in crimson. Every
weight-1 unfolded branch lies on a primitive band.

Run from the unpacked bundle directory (this file's directory):

    python reproduce.py                    # both figures, few seconds each

Requires the ``unfolding`` package importable (pip install -e <unfolding
repo checkout>, or PYTHONPATH=<unfolding repo checkout>) plus numpy,
matplotlib and HamiltonIO (the ABACUS parser). No ABACUS run is needed:
the committed OUT directories in data/ carry the sparse H and S tables,
and refs/ the pseudopotentials and numerical orbitals.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = B @ prim
PATH = "GXWGLWX"
NPTS = 300

# fcc high-symmetry points in primitive reciprocal fractional coordinates
# (Setyawan-Curtarolo values)
SPECIAL = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75),
    "L": (0.5, 0.5, 0.5),
}


def read_abacus_model(name):
    from HamiltonIO.abacus.abacus_wrapper import AbacusParser

    outpath = os.path.join(DATA, name, "OUT." + name)
    return AbacusParser(outpath=outpath).get_models()


def make_unfolder(sc_model, p_doped=False):
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = read_abacus_model("si_prim")
    counts = [13] * 8  # DZP 2s2p1d numerical orbitals
    rm = RelabelMap.from_atoms(
        sc_model.atoms, prim.atoms, B,
        orb_counts_sc=counts, orb_counts_prim=[13, 13],
        match_species=not p_doped,
    )
    return LCAOUnfolder(HamiltonIOModel(sc_model), rm), prim


def band_path(prim_atoms):
    """Primitive high-symmetry path: (kpts, x, Xq, knames)."""
    cell = np.asarray(prim_atoms.cell)
    bcart = 2 * np.pi * np.linalg.inv(cell).T
    segs = list(zip(PATH, PATH[1:]))
    lengths = [
        np.linalg.norm((np.array(SPECIAL[b]) - np.array(SPECIAL[a])) @ bcart)
        for a, b in segs
    ]
    total = sum(lengths)
    kpts, seg_starts = [], []
    for i, (a, b) in enumerate(segs):
        pa, pb = np.array(SPECIAL[a]), np.array(SPECIAL[b])
        n = max(2, round(NPTS * lengths[i] / total) + 1)
        last = i == len(segs) - 1
        t = np.linspace(0.0, 1.0, n, endpoint=last)
        seg_starts.append(len(kpts))
        kpts.extend(pa + (pb - pa) * t[:, None])
    kpts = np.array(kpts)
    d = np.linalg.norm(np.diff(kpts, axis=0) @ bcart, axis=1)
    x = np.concatenate([[0.0], np.cumsum(d)])
    Xq = [x[i] for i in seg_starts] + [x[-1]]
    knames = [{"G": r"$\Gamma$"}.get(s, s) for s in PATH]
    return kpts, x, np.array(Xq), knames


def prim_bands(prim, kfrac):
    """Primitive-cell eigenvalues along the path, shape (nk, n_orb_prim)."""
    from scipy.linalg import eigh

    from unfolding.lcao_unfolder import HamiltonIOModel

    pm = HamiltonIOModel(prim)
    return np.array([eigh(*pm.hs_and_eigen(k), eigvals_only=True) for k in kfrac])


def draw(unf, prim, efermi, out_path, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from unfolding.plotphon import plot_band_weight

    kpts, x, Xq, knames = band_path(prim.atoms)
    res = unf.compute(kpts, method="ideal")
    eprim = prim_bands(prim, kpts)

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


def main():
    sc = read_abacus_model("si_conv")
    unf, prim = make_unfolder(sc)
    p1 = draw(unf, prim, sc.efermi,
              os.path.join(HERE, "abacus_si_unfolded.png"),
              title="ABACUS LCAO Si unfolded (8-atom conventional cell)")
    print("wrote", p1)

    sc7p = read_abacus_model("si7p")
    unf7, _ = make_unfolder(sc7p, p_doped=True)
    p2 = draw(unf7, prim, sc7p.efermi,
              os.path.join(HERE, "abacus_si_p_doped.png"),
              title="ABACUS LCAO Si:P unfolded (dopant on host site)")
    print("wrote", p2)
    return p1, p2


if __name__ == "__main__":
    main()
