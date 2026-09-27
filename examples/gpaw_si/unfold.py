"""Unfolding example: pristine Si and Si:P from a GPAW LCAO calculation.

Reads committed GPAW restarts (``tests/data/gpaw_example/``), converts
them to the real-space model interface of
:class:`unfolding.lcao_unfolder.LCAOUnfolder` through
``HamiltonIO.gpaw.GpawLcaoModel``, and evaluates the ideal unfolding
weight along the primitive fcc path Gamma-X-W-Gamma-L-W-X.

The supercell is the 8-atom conventional cubic cell written in units of
the primitive fcc lattice vectors (rows convention
``A_sc = B @ A_prim``)::

    B = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]

The supercell momentum of a primitive k is ``K = k @ B.T`` (B symmetric).
For Si:P the dopant atom maps onto the host site it replaces
(``match_species=False``): the ideal weight then measures resemblance to
the ideal Si crystal, so host bands stay at weight 1 while
impurity-hybridized states carry fractional weight.

Run headless:  python unfold.py [outdir]
Outputs:       gpaw_si_unfolded.png, gpaw_si_p_doped.png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, "tests", "data", "gpaw_example")

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
A = 5.43
PRIM_CELL = A * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
SPECIAL = {
    "G": (0.0, 0.0, 0.0), "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75), "L": (0.5, 0.5, 0.5),
}
PATH = "GXWGLWX"
NPTS = 300


def make_unfolder(sc_gpw, match_species=True):
    """LCAOUnfolder for a committed GPAW supercell restart."""
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
    """Primitive-cell eigenvalues (eV) along the path from the GPAW model."""
    from scipy.linalg import eigh

    return np.array([eigh(*prim.hs_and_eigen(k), eigvals_only=True) for k in kpts])


def spectral_figure(res, eprim, x, Xq, knames, title, out_path):
    import matplotlib.pyplot as plt

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
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


def main(outdir):
    kpts, x, Xq, knames = band_path()

    prim, unf = make_unfolder("si_sc_lcao.gpw")
    # the coarse-grid GPAW tables carry a ~1e-3 position-gauge
    # artifact (see GpawLcaoModel); the weights stay within ~0.02
    # of the binary ideal
    res = unf.compute(kpts, method="ideal", atol_imag=0.05, atol_orth=0.05)
    print("pristine: max |w(1-w)| =", np.abs(res.weights * (1 - res.weights)).max())
    p = spectral_figure(
        res, primitive_bands(prim, kpts), x, Xq, knames,
        "GPAW LCAO Si unfolded (8-atom conventional cell)",
        os.path.join(outdir, "gpaw_si_unfolded.png"),
    )
    print("wrote", p)

    _, unfd = make_unfolder("si7p_lcao.gpw", match_species=False)
    resd = unfd.compute(kpts, method="ideal")
    p = spectral_figure(
        resd, primitive_bands(prim, kpts), x, Xq, knames,
        "GPAW LCAO Si:P unfolded (substitutional P)",
        os.path.join(outdir, "gpaw_si_p_doped.png"),
    )
    print("wrote", p)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REPO, "docs", "static", "images")
    os.makedirs(outdir, exist_ok=True)
    main(outdir)
