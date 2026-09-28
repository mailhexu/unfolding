#!/usr/bin/env python
"""Reproduce the GPAW LCAO Si figures (gpaw_si_unfolded.png).

Docs example: docs/content/examples/gpaw-si.md in the unfolding repo.
Unfolds the 8-atom conventional-cubic Si supercell LCAO restart onto the
2-atom fcc primitive cell along Gamma-X-W-Gamma-L-W-X (pristine figure,
with the independently computed primitive-cell bands overlaid in
crimson) and the Si7P supercell (doped figure, dopant mapped onto the
host site).

Run from the unpacked bundle directory (this file's directory):

    python reproduce.py                # both figures

The .gpw restarts are NOT shipped (about 30 MB each, over the bundle
cap). Generate them with GPAW (about 2-3 h serial):

    python generate_fixtures.py

then re-run this script. Requires the ``unfolding`` package importable
(pip install -e <unfolding repo checkout>, or PYTHONPATH=<unfolding repo
checkout>) plus numpy, matplotlib, scipy and GPAW >= 25.

GPAW's LCAO matrices already carry every PAW contribution -- the overlap
is the projector-augmented one and the Hamiltonian includes the dH
projector terms -- so the unfolding consumes them exactly like any other
atomic-orbital table: no PAW correction is applied on top.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = B @ prim
A = 5.43
PRIM_CELL = A * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
SPECIAL = {
    "G": (0.0, 0.0, 0.0), "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75), "L": (0.5, 0.5, 0.5),
}
PATH = "GXWGLWX"
NPTS = 300

NEEDED = ["si_prim_lcao.gpw", "si_sc_lcao.gpw", "si7p_lcao.gpw"]

FIXTURES_MISSING = """
The LCAO restarts are NOT shipped (about 30 MB each). Generate them with
GPAW (PBE, h=0.17, gamma-centered grids; ~2-3 h serial):

    python generate_fixtures.py

then re-run this script.
"""


def make_unfolder(sc_gpw, match_species=True):
    from HamiltonIO.gpaw import GpawLcaoModel
    from unfolding.lcao_unfolder import LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = GpawLcaoModel.from_file(os.path.join(DATA, "si_prim_lcao.gpw"))
    sc = GpawLcaoModel.from_file(os.path.join(DATA, sc_gpw))
    # GPAW szp basis: 4 atomic orbitals per atom (Si and P alike)
    rm = RelabelMap.from_atoms(
        sc.atoms, prim.atoms, B,
        orb_counts_sc=[4] * 8, orb_counts_prim=[4, 4],
        match_species=match_species,
    )
    return prim, LCAOUnfolder(sc, rm)


def band_path():
    """Primitive-path k-points, accumulated distance, tick positions, names."""
    bcart = 2 * np.pi * np.linalg.inv(PRIM_CELL).T
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
        t = np.linspace(0.0, 1.0, n, endpoint=i == len(segs) - 1)
        seg_starts.append(len(kpts))
        kpts.extend(pa + (pb - pa) * t[:, None])
    kpts = np.array(kpts)
    d = np.linalg.norm(np.diff(kpts, axis=0) @ bcart, axis=1)
    x = np.concatenate([[0.0], np.cumsum(d)])
    Xq = [x[i] for i in seg_starts] + [x[-1]]
    knames = [{"G": r"$\Gamma$"}.get(s, s) for s in PATH]
    return kpts, x, np.array(Xq), knames


def primitive_bands(prim, kpts):
    """Primitive-cell eigenvalues (eV) along the path."""
    from scipy.linalg import eigh

    return np.array([eigh(*prim.hs_and_eigen(k), eigvals_only=True) for k in kpts])


def spectral_figure(res, eprim, x, Xq, knames, title, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from unfolding.plotphon import plot_band_weight

    nb = res.weights.shape[1]
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(res.eigenvalues[:, ib]) for ib in range(nb)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)]
    ax = plot_band_weight(
        kslist, ekslist, wkslist,
        xticks=[knames, Xq],
        ylabel="Energy (eV)",
        ypad=1.5,
    )
    ax.set_title(title, fontsize=10)
    lines = ax.plot(x, eprim, color="crimson", lw=1.0, alpha=0.9, zorder=5)
    lines[0].set_label("GPAW primitive-cell bands")
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
    missing = [n for n in NEEDED if not os.path.isfile(os.path.join(DATA, n))]
    if missing:
        print("missing fixtures: " + ", ".join(missing))
        print(FIXTURES_MISSING)
        raise SystemExit(1)

    kpts, x, Xq, knames = band_path()

    prim, unf = make_unfolder("si_sc_lcao.gpw")
    res = unf.compute(kpts, method="ideal", atol_imag=0.05, atol_orth=0.05)
    eprim = primitive_bands(prim, kpts)
    # one common zero per panel: the unfolded system's own Fermi level
    efermi = unf._model.efermi
    res.eigenvalues -= efermi
    eprim_pristine = eprim - efermi
    print("pristine: max |w(1-w)| =", np.abs(res.weights * (1 - res.weights)).max())
    dev = [
        np.abs(eprim_pristine[ik] - res.eigenvalues[ik, ib]).min()
        for ik in range(len(kpts))
        for ib in np.where(res.weights[ik] > 0.99)[0]
    ]
    print("pristine: max |E_unfolded - E_prim| over weight-1 branches =",
          f"{np.max(np.abs(dev)) * 1e3:.1f} meV")
    p = spectral_figure(
        res, eprim_pristine, x, Xq, knames,
        "GPAW LCAO Si unfolded (8-atom conventional cell)",
        os.path.join(HERE, "gpaw_si_unfolded.png"),
    )
    print("wrote", p)

    _, unfd = make_unfolder("si7p_lcao.gpw", match_species=False)
    resd = unfd.compute(kpts, method="ideal")
    efermi_d = unfd._model.efermi
    resd.eigenvalues -= efermi_d
    p = spectral_figure(
        resd, eprim - efermi_d, x, Xq, knames,
        "GPAW LCAO Si:P unfolded (substitutional P)",
        os.path.join(HERE, "gpaw_si_p_doped.png"),
    )
    print("wrote", p)


if __name__ == "__main__":
    main()
