#!/usr/bin/env python
"""Reproduce the pristine GPAW plane-wave Si path figure.

Docs example: docs/content/examples/gpaw-si-pw.md in the unfolding repo.

The pristine path figure requires data/si8_pw.gpw (~290 MB, not shipped);
generate it with your own GPAW as described in README.txt.

Requires the ``unfolding`` package importable (pip install -e <unfolding
repo checkout>, or PYTHONPATH=<unfolding repo checkout>) plus numpy,
matplotlib and a recent GPAW (the .gpw reader). The plane-wave weight of
a state at primitive k is the norm of its coefficients on the reciprocal
coset (K + n) @ B^-T = k (mod 1): the basis is orthonormal, so pristine
weights are exactly 0/1 (GPAW's PAW-metric normalization is renormalized
to sum |c|^2 = 1 per band, which leaves coset fractions unchanged -- the
weights describe the *pseudo* wavefunctions).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = B @ prim
A = 5.43
PRIM_CELL = A * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
SPECIAL = {
    "G": (0.0, 0.0, 0.0), "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75), "L": (0.5, 0.5, 0.5),
}
PATH = "GXWGLWX"
NPTS = 300
DEGEN_EV = 1e-3  # resolve degenerate branch weights when plotting

PRISTINE_MISSING = """
data/si8_pw.gpw is NOT shipped (about 290 MB, over the bundle cap).
Generate it with `python generate_restart.py`; the script runs an 8-atom
conventional-cell PBE/PW calculation (340 eV, 24 bands, symmetry off) on
the exact 305-point primitive GXWGLWX path mapped to supercell coordinates
with K_sc = k_prim @ B.T. The restart is written to data/si8_pw.gpw.
"""


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


def _eigendata(data):
    from unfolding.pw_unfolder import PWEigenData

    return PWEigenData(
        kpoints=data.kpoints,
        gvecs=data.gvecs,
        # PWEigenData layout: (nspin, nband, nspinor, npw)
        coefficients=[c[None, :, None, :] for c in data.coefficients],
        eigenvalues=data.eigenvalues[:, None, :],
    )


def pristine():
    """Path figure from the generated pristine path restart."""
    src = DATA / "si8_pw.gpw"
    if not src.is_file():
        raise FileNotFoundError(
            f"missing {src}; run `python generate_restart.py` first\n{PRISTINE_MISSING}"
        )

    from unfolding.plotphon import plot_band_weight
    from unfolding.pw_unfolder import PWUnfolder
    from HamiltonIO.gpaw import GpawPWParser

    data = GpawPWParser(src).read()
    unf = PWUnfolder(_eigendata(data), B)

    kpts, x, Xq, knames = band_path()
    res = unf.compute(kpts, resolve_degenerate=DEGEN_EV)
    print("pw pristine: max |w(1-w)| (lowest 20 bands) =",
          np.abs(res.weights[:, :20] * (1 - res.weights[:, :20])).max())

    overlay = None
    bands_npz = DATA / "si_prim_pw_path_bands.npz"
    if bands_npz.is_file():
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
    handles = [Line2D([0], [0], color="blue", alpha=0.5, lw=2,
                      label="unfolded spectral weight")]
    if overlay is not None:
        lines = ax.plot(x, overlay, color="crimson", lw=1.0, alpha=0.9,
                        zorder=5)
        lines[0].set_label("primitive-cell PW bands")
        handles.append(lines[0])
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.85)
    out = HERE / "gpaw_si_pw_unfolded.png"
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print("wrote", out)
    return out


if __name__ == "__main__":
    pristine()
