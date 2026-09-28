"""Unfolding example: pristine Si planewave bands from a GPAW PW restart.

Reads the committed planewave restart ``si8_pw.gpw`` (8-atom conventional
cell, 340 eV cutoff, PBE, 300-point primitive path sampled in supercell
coordinates) through ``HamiltonIO.gpaw.GpawPWParser``, feeds the
eigendata to the backend-free :class:`unfolding.pw_unfolder.PWUnfolder`,
and plots the weight-coded unfolded bands with the independently
computed primitive-cell planewave bands overlaid.

The planewave weight of a state at primitive k is the norm of its
coefficients on the reciprocal coset ``(K + n) @ B^-T = k (mod 1)``: the
basis is orthonormal, so pristine weights are exactly 0/1 (GPAW's
pseudo-wavefunction normalization is renormalized to sum |c|^2 = 1 per
band, which leaves coset fractions unchanged).

Run headless:  python unfold.py [outdir]
Outputs:       gpaw_si_pw_unfolded.png
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
DEGEN_EV = 1e-3  # resolve degenerate branch weights when plotting


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


def load_pw_unfolder():
    """(GpawPWData, PWUnfolder) for the committed planewave restart."""
    from HamiltonIO.gpaw import GpawPWParser
    from unfolding.pw_unfolder import PWEigenData, PWUnfolder

    data = GpawPWParser(os.path.join(DATA, "si8_pw.gpw")).read()
    eigendata = PWEigenData(
        kpoints=data.kpoints,
        gvecs=data.gvecs,
        # PWEigenData layout: (nspin, nband, nspinor, npw)
        coefficients=[c[None, :, None, :] for c in data.coefficients],
        eigenvalues=data.eigenvalues[:, None, :],
    )
    return data, PWUnfolder(eigendata, B)


def main(outdir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from unfolding.plotphon import plot_band_weight

    kpts, x, Xq, knames = band_path()
    data, unf = load_pw_unfolder()
    # the four highest stored bands converge loosely in the path SCF
    # and are dropped from the figure
    res = unf.compute(kpts, resolve_degenerate=DEGEN_EV)
    print("pw pristine: max |w(1-w)| (lowest 20 bands) =",
          np.abs(res.weights[:, :20] * (1 - res.weights[:, :20])).max())

    overlay = None
    bands_npz = os.path.join(DATA, "si_prim_pw_path_bands.npz")
    if os.path.exists(bands_npz):
        ref = np.load(bands_npz)
        eprim = ref["eigenvalues"] - ref["efermi"]
        esc = res.eigenvalues - data.efermi
        hi = res.weights > 0.9
        devs = []
        for ik in range(len(res.eigenvalues)):
            u = np.sort(esc[ik][hi[ik]])
            p = np.sort(eprim[ik])
            n = min(len(u), len(p))
            if n:
                devs.append(u[:n] - p[:n])
        shift = float(np.median(np.concatenate(devs)))
        overlay = eprim + shift
        print("primitive-overlay potential shift:", round(shift, 4), "eV")

    # plot the 20 lowest bands: the four highest stored states converge
    # loosely in the path SCF and are not meaningful to show
    nb = 20
    esc = res.eigenvalues - data.efermi
    kslist = [list(x) for _ in range(nb)]
    ekslist = [list(esc[:, ib]) for ib in range(nb)]
    wkslist = [list(np.clip(res.weights[:, ib], 0.0, 1.0)) for ib in range(nb)]
    ax = plot_band_weight(
        kslist, ekslist, wkslist,
        xticks=[knames, Xq],
        ylabel="Energy (eV)",
        ypad=1.5,
    )
    ax.set_title("GPAW planewave Si unfolded (8-atom conventional cell)",
                 fontsize=10)
    # plot_band_weight draws the weight-coded bands as blue lines with
    # alpha = w/(width + 0.011); the proxy matches the weight-1 appearance.
    handles = [Line2D([0], [0], color="blue", alpha=0.5, lw=2,
                      label="unfolded spectral weight")]
    if overlay is not None:
        lines = ax.plot(x, overlay, color="crimson", lw=1.0, alpha=0.9,
                        zorder=5)
        lines[0].set_label("primitive-cell PW bands")
        handles.append(lines[0])
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.85)
    out = os.path.join(outdir, "gpaw_si_pw_unfolded.png")
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print("wrote", out)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REPO, "docs", "static", "images")
    os.makedirs(outdir, exist_ok=True)
    main(outdir)
