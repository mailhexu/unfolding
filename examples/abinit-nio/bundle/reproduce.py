#!/usr/bin/env python
"""Reproduce the AFM NiO unfolded-band figure of the
"ABINIT WFK AFM NiO" docs example.

Run from the unpacked bundle directory (the directory containing this
script); the ``unfolding`` package must be importable (pip install
unfolding, or put the repo root on PYTHONPATH).  Writes
nio_afm_unfolded.png into the bundle directory.

Type-II antiferromagnetic NiO in the 4-atom magnetic primitive cell
(A_afm = M @ A_prim with M = [[1, 0, 1], [0, 1, 1], [1, 1, 0]], i.e. the
2-atom rocksalt primitive cell doubled along [111]) is unfolded back
onto the 2-atom primitive cell along Gamma-X-W-Gamma-L-W-X.  The
collinear WFK (nsppol 2) stores two spin channels; because the AFM keeps
inversion (time reversal x sublattice translation maps the channels onto
each other) the spin-up and spin-down unfolded band structures coincide
band-for-band, so a single spin-up panel is plotted.  Pass spin=1 to
compute the identical partner.

No WFK ships with this bundle (the dense nsppol-2 path WFK is ~1.1 GB):
data/ holds the WFK you regenerate with ABINIT (ecut 40, nsppol 2,
dataset-2 non-SCF path run of inputs/nio_afm.abi) as printed below; the
script picks the dense WFK up automatically whenever it is present (the
corner deck nio_afm_corners.abi yields a small WFK that is enough for a
coarse figure).
"""
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parent
DATA = BUNDLE / "data"

# AFM magnetic supercell = 2x the 2-atom rocksalt primitive cell.
M_AFM = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int)
SIGMA_EV = 0.05

# fcc high-symmetry points in primitive reciprocal fractional coordinates
# (same path convention as the Si example).
SPECIAL = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75),
    "L": (0.5, 0.5, 0.5),
}
PATH = "GXWGLWX"
KNAMES = [r"$\Gamma$", "X", "W", r"$\Gamma$", "L", "W", "X"]
NPTS = 300
ACELL = 10.26  # axis scaling only; the path shape is fcc-independent

REGENERATE = """\
Dense path WFK data/nio_afm_patho_DS2_WFK.nc not found.  For the full
305-point published figure, regenerate it with ABINIT >= 9 (netCDF,
iomode 3).  From this bundle directory run:

  mkdir -p run_nio && cd run_nio
  cp ../inputs/nio_afm.abi ../pseudos/Ni.psp8 ../pseudos/O.psp8 .
  abinit nio_afm.abi > nio_afm.abo 2>&1
  cp nio_afmo_DS2_WFK.nc ../data/ && cd ..

The input is a two-dataset calculation at ecut 40 with nsppol 2 / nspden 2
(Spinat-given type-II AFM moments): dataset 1 is the AFM SCF on a 4x4x4
mesh, dataset 2 a frozen-density (iscf -2, getden 2) non-SCF run over the
305-point supercell path with kptopt 0 and istwfk 1.  Expect on the order
of hours and a ~1.1 GB WFK.
"""


def gxwglwx_path():
    """(kpts, xqpts, Xqpts) on the fcc GXWGLWX path, points distributed
    proportionally to Cartesian segment length.

    Axis construction only: uses the fcc frame of the Si example (the
    rocksalt primitive cell is fcc too, so the special points agree).
    """
    prim = np.linalg.inv(M_AFM.astype(float)) @ (np.eye(3) * ACELL)
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


def match_path_subset(wfk_path):
    """(data, kpts, xqpts) for the stored WFK in path order.

    A dense regenerated WFK stores the full grid; the included corner
    fixture stores a subset, plotted on the same path axis.  Stored
    k-points are supercell-frame momenta.
    """
    from unfolding.abinit_unfold import read_wfk

    kpts, xqpts, _ = gxwglwx_path()
    data = read_wfk(wfk_path)
    stored = np.mod(data.kpoints @ np.linalg.inv(M_AFM.T), 1.0)
    delta = ((stored[:, None, :] - np.mod(kpts, 1.0)[None, :, :] + 0.5) % 1.0) - 0.5
    idx = np.linalg.norm(delta, axis=2).argmin(axis=1)
    if np.linalg.norm(delta[np.arange(len(idx)), idx], axis=1).max() > 1e-6:
        raise ValueError(
            f"{wfk_path.name}: k-points are not on the GXWGLWX path"
        )
    sel = np.sort(idx)
    return data, kpts[sel], xqpts[sel]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from unfolding.abinit_unfold import HARTREE_TO_EV
    from unfolding.pw_unfolder import PWUnfolder

    dense = DATA / "nio_afm_patho_DS2_WFK.nc"
    sparse = DATA / "nio_afm_cornerso_DS2_WFK.nc"
    wfk = dense if dense.is_file() else sparse
    if not wfk.is_file():
        print("No WFK files ship with this bundle (they are large binary data).")
        print(REGENERATE)
        raise SystemExit(1)
    if not dense.is_file():
        print(REGENERATE)

    data, kpts, x = match_path_subset(wfk)
    _, _, Xqpts = gxwglwx_path()

    res = PWUnfolder(data, M_AFM).compute(
        kpts, spin=0, resolve_degenerate=1e-3 / HARTREE_TO_EV
    )
    E = res.eigenvalues * HARTREE_TO_EV - data.fermi_energy * HARTREE_TO_EV
    W = np.clip(res.weights, 0.0, 1.0)
    # Window covers O-2p/Ni-3d valence and low conduction; the Ni-3s
    # semicore multiplet (zion=18) sits near -60 eV and is out of scope.
    egrid = np.linspace(-16.0, 8.0, 900)
    pre = 1.0 / (SIGMA_EV * np.sqrt(2 * np.pi))
    A = np.zeros((len(E), len(egrid)))
    for ik in range(len(E)):
        for eb, wb in zip(E[ik], W[ik]):
            A[ik] += wb * pre * np.exp(-0.5 * ((egrid - eb) / SIGMA_EV) ** 2)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.pcolormesh(
        x, egrid, A.T, cmap="Blues", vmin=0.0, vmax=2.0,
        shading="auto", rasterized=True,
    )
    for xt in Xqpts[1:-1]:
        ax.axvline(xt, color="gray", lw=0.5)
    ax.axhline(0.0, ls="--", color="k", lw=0.7)
    ax.set_xticks(Xqpts)
    ax.set_xticklabels(KNAMES)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(egrid[0], egrid[-1])
    ax.set_ylabel(r"Energy relative to $E_F$ (eV)")
    ax.set_title("AFM NiO unfolded (spin up; spin down identical by symmetry)")
    fig.tight_layout()
    out_path = BUNDLE / "nio_afm_unfolded.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
