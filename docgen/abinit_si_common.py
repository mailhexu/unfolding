#!/usr/bin/env python
"""Shared helpers for the ABINIT Si/Si7P unfolded-band docgen figures.

The band path is identical to the SIESTA Si example: fcc special points
``GXWGLWX`` (Setyawan-Curtarolo values in the primitive reciprocal basis),
points distributed proportionally to Cartesian segment length. Supercell
momenta are ``K = k_prim @ M.T`` for the WFK path runs.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "abinit_si"
MATRIX = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int)
ACELL = 10.26  # bohr, conventional cubic cell
NPTS = 300
DEGEN_TOL_EV = 1e-3
SIGMA_EV = 0.045

# fcc high-symmetry points in primitive reciprocal fractional coordinates
# (Setyawan-Curtarolo; identical to the SIESTA example's SPECIAL table).
SPECIAL = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "W": (0.5, 0.25, 0.75),
    "L": (0.5, 0.5, 0.5),
}
PATH = "GXWGLWX"
KNAMES = [r"$\Gamma$", "X", "W", r"$\Gamma$", "L", "W", "X"]


def siesta_style_path():
    """(kpts, xqpts, Xqpts) replicating the SIESTA example's path grid."""
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

    The dense nic6 regeneration stores the full grid; a sparse committed
    fixture stores a subset, plotted on the same path axis. ``matrix``
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
            f"{wfk_path.name}: k-points are not on the GXWGLWX path; "
            "regenerate with tests/data/abinit_si/regenerate_on_nic6.sh"
        )
    sel = np.sort(idx)
    return data, kpts[sel], xqpts[sel]


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
