#!/usr/bin/env python
"""Reproduce the ABINIT Si / Si:P unfolded-band figures of the
"ABINIT WFK Si and Si:P" docs example.

Run from the unpacked bundle directory (the directory containing this
script); the ``unfolding`` package must be importable (pip install
unfolding, or put the repo root on PYTHONPATH).  Writes two PNGs into
the bundle directory:

  si8_abinit_unfolded.png    pristine Si 8-atom supercell unfolded onto
                             Gamma-X-W-Gamma-L-W-X, with the independent
                             primitive-cell bands overlaid when the
                             primitive reference WFK is available
  si7p_abinit_unfolded.png   Si:P (one P substituting Si) spectral weight

No WFK ships with this bundle (they are large binary data): data/
holds the WFKs you regenerate with ABINIT (>= 9, netCDF support).  The
corner decks in inputs/ yield small corner WFKs -- with only those, the
figures are rendered from the corner points along the same path axis.
For the full 305-point published figures, run the dense dataset-2
non-SCF path inputs as printed below; this script picks the dense WFK
up automatically whenever it is present.
"""
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parent
DATA = BUNDLE / "data"

MATRIX = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int)
ACELL = 10.26  # bohr, conventional cubic cell
NPTS = 300
DEGEN_TOL_EV = 1e-3
SIGMA_EV = 0.045

# fcc high-symmetry points in primitive reciprocal fractional coordinates
# (Setyawan-Curtarolo; identical to the SIESTA example's path).
SPECIAL = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75),
    "L": (0.5, 0.5, 0.5),
}
PATH = "GXWGLWX"
KNAMES = [r"$\Gamma$", "X", "W", r"$\Gamma$", "L", "W", "X"]

REGENERATE = """\
Dense path WFK(s) not found in data/.  For the full 305-point published
figures, regenerate them with ABINIT >= 9 (built with netCDF, iomode 3).
From this bundle directory run:

  # dense pristine-Si8 supercell path WFK (~787 MB, ecut 25; dataset 1 is
  # the Gamma SCF, dataset 2 the frozen-density non-SCF 305-point path)
  mkdir -p run_si8 && cd run_si8
  cp ../inputs/si8_gxwglwx.abi ../pseudos/14-Si.nlcc.fhi .
  abinit si8_gxwglwx.abi > si8_gxwglwx.abo 2>&1
  cp si8_gxwglwxo_DS2_WFK.nc ../data/ && cd ..

  # dense Si7P supercell path WFK (~787 MB, ecut 25, dataset-2 non-SCF path)
  mkdir -p run_si7p && cd run_si7p
  cp ../inputs/si7p_gamma_x_path.abi ../pseudos/14-Si.nlcc.fhi ../pseudos/15-P.LDA.fhi .
  abinit si7p_gamma_x_path.abi > si7p_gamma_x_path.abo 2>&1
  cp si7p_gamma_x_patho_DS2_WFK.nc ../data/ && cd ..

  # (optional, for the crimson primitive-cell overlay on the Si8 figure)
  # 2-atom primitive cell, 8x8x8 SCF + same path, primitive coordinates
  mkdir -p run_prim && cd run_prim
  cp ../inputs/si_prim_path.abi ../pseudos/14-Si.nlcc.fhi .
  abinit si_prim_path.abi > si_prim_path.abo 2>&1
  cp si_prim_patho_DS2_WFK.nc ../data/ && cd ..

Each run is a two-dataset calculation (SCF at Gamma, then iscf -2 /
getden 2 non-SCF path sampling with kptopt 0); the *_DS2_WFK.nc files
feed this script.  Rendering below continues with the included corner
fixtures meanwhile.
"""


def siesta_style_path():
    """(kpts, xqpts, Xqpts) on the fcc GXWGLWX path, points distributed
    proportionally to Cartesian segment length."""
    prim = np.linalg.inv(MATRIX.astype(float)) @ (np.eye(3) * ACELL)
    bcart = 2 * np.pi * np.linalg.inv(prim).T
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
        t = np.linspace(0.0, 1.0, n, endpoint=(i == len(segs) - 1))
        seg_starts.append(len(kpts))
        kpts.extend(pa + (pb - pa) * t[:, None])
    kpts = np.asarray(kpts)
    d = np.linalg.norm(np.diff(kpts, axis=0) @ bcart, axis=1)
    xqpts = np.concatenate([[0.0], np.cumsum(d)])
    Xqpts = [xqpts[s] for s in seg_starts] + [xqpts[-1]]
    return kpts, xqpts, Xqpts


def match_path_subset(wfk_path, matrix=MATRIX):
    """(data, kpts, xqpts) for the stored WFK in path order.

    A dense regenerated WFK stores the full grid; the included corner
    fixtures store a subset, plotted on the same path axis.  ``matrix``
    selects the frame of the stored k-points (supercell by default;
    identity for a primitive-cell WFK already on the path).
    """
    from unfolding.abinit_unfold import read_wfk

    kpts, xqpts, _ = siesta_style_path()
    data = read_wfk(wfk_path)
    stored = np.mod(data.kpoints @ np.linalg.inv(matrix.T), 1.0)
    delta = ((stored[:, None, :] - np.mod(kpts, 1.0)[None, :, :] + 0.5) % 1.0) - 0.5
    idx = np.linalg.norm(delta, axis=2).argmin(axis=1)
    if np.linalg.norm(delta[np.arange(len(idx)), idx], axis=1).max() > 1e-6:
        raise ValueError(
            f"{wfk_path.name}: k-points are not on the GXWGLWX path"
        )
    sel = np.sort(idx)
    return data, kpts[sel], xqpts[sel]


def pw_data(data):
    """Transfer HamiltonIO WFK arrays to the backend-free unfolding core."""
    from unfolding.pw_unfolder import PWEigenData

    return PWEigenData(data.kpoints, data.gvecs, data.coefficients, data.eigenvalues)


def spectral_weight_map(res, data):
    """(x, egrid, A, E) Gaussian-smeared effective-band-structure map."""
    from unfolding.abinit_unfold import HARTREE_TO_EV

    E = res.eigenvalues * HARTREE_TO_EV - data.fermi_energy * HARTREE_TO_EV
    W = np.clip(res.weights, 0.0, 1.0)
    egrid = np.linspace(E.min() - 0.8, E.max() + 0.8, 1100)
    pre = 1.0 / (SIGMA_EV * np.sqrt(2 * np.pi))
    A = np.zeros((len(E), len(egrid)))
    for ik in range(len(E)):
        for eb, wb in zip(E[ik], W[ik]):
            A[ik] += wb * pre * np.exp(-0.5 * ((egrid - eb) / SIGMA_EV) ** 2)
    return E, W, egrid, A


def draw_map(x, egrid, A, Xqpts, title, out_path, overlay=None):
    """Save the EBS heatmap; optional (x, energies) overlay curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.pcolormesh(
        x, egrid, A.T, cmap="Blues", vmin=0.0, vmax=2.0,
        shading="auto", rasterized=True,
    )
    if overlay is not None:
        ox, oe = overlay
        ax.plot(ox, oe, color="crimson", lw=1.0, alpha=0.9, zorder=5,
                label="primitive-cell bands")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    for xt in Xqpts[1:-1]:
        ax.axvline(xt, color="gray", lw=0.5)
    ax.axhline(0.0, ls="--", color="k", lw=0.7)
    ax.set_xticks(Xqpts)
    ax.set_xticklabels(KNAMES)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(egrid[0], egrid[-1])
    ax.set_ylabel(r"Energy relative to $E_F$ (eV)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def si8_figure(out_path):
    """Pristine Si8: unfold onto the primitive path, overlay primitive
    bands when the primitive reference WFK is available."""
    from unfolding.abinit_unfold import HARTREE_TO_EV
    from unfolding.pw_unfolder import PWUnfolder

    dense = DATA / "si8_gxwglwxo_DS2_WFK.nc"
    sparse = DATA / "si8_gxwglx_cornerso_DS2_WFK.nc"
    wfk = dense if dense.is_file() else sparse
    if not wfk.is_file():
        raise FileNotFoundError(
            f"no Si8 path WFK found (tried {dense.name}, {sparse.name})"
        )

    data, kpts, x = match_path_subset(wfk)
    res = PWUnfolder(pw_data(data), MATRIX).compute(
        kpts, resolve_degenerate=DEGEN_TOL_EV / HARTREE_TO_EV
    )
    E, W, egrid, A = spectral_weight_map(res, data)

    overlay = None
    prim_wfk = DATA / "si_prim_patho_DS2_WFK.nc"
    if prim_wfk.is_file():
        pdata, pkpts, px = match_path_subset(prim_wfk, np.eye(3, dtype=int))
        pres = PWUnfolder(pw_data(pdata), np.eye(3, dtype=int)).compute(pkpts)
        pe = pres.eigenvalues * HARTREE_TO_EV - pdata.fermi_energy * HARTREE_TO_EV
        # Align the potential reference by the median high-weight offset.
        hi = W > 0.9
        devs = []
        for ik in range(len(E)):
            unfolded = np.sort(E[ik][hi[ik]])
            primitive = np.sort(pe[ik])
            n = min(len(unfolded), len(primitive))
            if n:
                devs.append(unfolded[:n] - primitive[:n])
        shift = np.median(np.concatenate(devs))
        overlay = (px, pe + shift)

    _, _, Xqpts = siesta_style_path()
    return draw_map(
        x, egrid, A, Xqpts,
        "ABINIT Si unfolded: supercell vs primitive bands", out_path,
        overlay=overlay,
    )


def si7p_figure(out_path):
    """Si:P: spectral-weight map; defect-hybridized states are dimmer.

    Coverage note: the corner deck (si7p_gxwglx_corners.abi) stores only
    four supercell momenta -- (0,0,0), (0,1,0), (0.5,1,0), (0.5,0.5,0.5),
    i.e. the path corners Gamma/X/W/L via K = k_prim @ M.T -- so with the
    corner fallback only 7 of the 305 requested path folds (the corner
    ticks themselves) have a stored match and the map shows weight only
    there. The dense deck (si7p_gamma_x_path.abi) restores the full map.
    """
    from unfolding.abinit_unfold import HARTREE_TO_EV
    from unfolding.pw_unfolder import PWUnfolder

    dense = DATA / "si7p_gamma_x_patho_DS2_WFK.nc"
    sparse = DATA / "si7p_gxwglx_cornerso_DS2_WFK.nc"
    wfk = dense if dense.is_file() else sparse
    if not wfk.is_file():
        raise FileNotFoundError(
            f"no Si7P path WFK found (tried {dense.name}, {sparse.name})"
        )

    data, kpts, x = match_path_subset(wfk)
    res = PWUnfolder(pw_data(data), MATRIX).compute(
        kpts, resolve_degenerate=DEGEN_TOL_EV / HARTREE_TO_EV
    )
    _, _, egrid, A = spectral_weight_map(res, data)
    _, _, Xqpts = siesta_style_path()
    return draw_map(
        x, egrid, A, Xqpts, "ABINIT Si$_7$P unfolded spectral weight", out_path
    )


def main():
    import matplotlib

    matplotlib.use("Agg")

    dense = [
        name
        for name in (
            "si8_gxwglwxo_DS2_WFK.nc",
            "si7p_gamma_x_patho_DS2_WFK.nc",
            "si_prim_patho_DS2_WFK.nc",
        )
        if not (DATA / name).is_file()
    ]
    corners = [
        name
        for name in ("si8_gxwglx_cornerso_DS2_WFK.nc", "si7p_gxwglx_cornerso_DS2_WFK.nc")
        if not (DATA / name).is_file()
    ]
    if corners:
        # nothing renderable at all: no corner WFKs either
        print("Missing WFK fixtures: " + ", ".join(dense + corners))
        print("No WFK files ship with this bundle (they are large binary data).")
        print(REGENERATE)
        raise SystemExit(1)
    if dense:
        # the corner WFKs still render coarse figures on the same axis
        print("Missing dense WFKs: " + ", ".join(dense))
        print(REGENERATE)

    print(si8_figure(BUNDLE / "si8_abinit_unfolded.png"))
    print(si7p_figure(BUNDLE / "si7p_abinit_unfolded.png"))


if __name__ == "__main__":
    main()
