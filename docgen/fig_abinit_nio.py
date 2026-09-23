#!/usr/bin/env python
"""Render the AFM NiO unfolded-band figure (single spin channel).

Type-II antiferromagnetic NiO in the 4-atom magnetic primitive cell
(``A_afm = M @ A_prim`` with ``M = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]``,
i.e. the 2-atom rocksalt primitive cell doubled along [111]) is unfolded
back onto the 2-atom primitive cell. The collinear WFK stores two spin
channels; because the AFM keeps inversion (time reversal x sublattice
translation maps the channels onto each other), the spin-up and spin-down
unfolded band structures coincide band-for-band, so a single spin-up
panel is plotted. Pass ``spin=1`` to compute the identical partner.
"""
import sys

import numpy as np

from abinit_si_common import DATA, KNAMES, ROOT, match_path_subset
from unfolding.abinit_unfold import HARTREE_TO_EV
from unfolding.pw_unfolder import PWUnfolder

# AFM magnetic supercell = 2x the 2-atom rocksalt primitive cell.
M_AFM = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int)
SIGMA_EV = 0.05


def main(out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from abinit_si_common import siesta_style_path

    dense = DATA / "nio_afm_patho_DS2_WFK.nc"
    sparse = DATA / "nio_afm_cornerso_DS2_WFK.nc"
    wfk = dense if dense.is_file() else sparse
    if not wfk.is_file():
        raise FileNotFoundError(
            f"no AFM NiO path WFK found (tried {dense.name}, {sparse.name})"
        )

    data, kpts, x = match_path_subset(wfk, M_AFM)
    _, _, Xqpts = siesta_style_path()

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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    output = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ROOT / "docs" / "static" / "images" / "nio_afm_unfolded.png"
    )
    print(main(output))
