#!/usr/bin/env python
"""Render the G-AFM SrMnO3 downfolded-magnon figure (TB2J example).

Source: the committed TB2J pickle from the SrMnO3 VASP-Wannier example
(sqrt2 x sqrt2 x sqrt2 G-AFM cell, 10 atoms, 2 Mn at +/-2.81 mu_B). The
downfold target is the 5-atom pseudo-cubic primitive cell: the TB2J cell
axes are (0,1,1), (1,0,1), (1,1,0) in pseudo-cubic units, so
M = [[0,1,1],[1,0,1],[1,1,0]] and A_prim = M^-1 A_sc. The G-AFM
dispersion is Q-periodic, so both folds of each stored momentum are
degenerate; the group-total presentation renders binary weights.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "tb2j_srmmo3" / "TB2J_results"
M_AFM = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


def main(out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.dft.kpoints import bandpath, get_special_points

    from unfolding.tb2j_unfold import magnon_eigendata_from_tb2j

    from TB2J.magnon.magnon3 import Magnon

    magnon = Magnon.from_TB2J_results(path=str(DATA))
    magnon.set_reference(
        Q=(0, 0, 0),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([1.0, 0.0, 0.0]),
    )

    prim = np.linalg.solve(M_AFM.astype(float), np.asarray(magnon.cell, float))
    points = get_special_points(prim, eps=0.01)
    letters = "GXMGR"
    path = bandpath([points[c] for c in letters], prim, 200)
    kpts, (x, X, labels) = path.kpts, path.get_linear_kpoint_axis()

    from unfolding import MagnonUnfolder

    eigendata = magnon_eigendata_from_tb2j(magnon, np.mod(kpts @ M_AFM.T, 1.0))
    unf = MagnonUnfolder(eigendata, M_AFM)
    res = unf.compute(kpts, resolve_degenerate=1e-5)

    from unfolding.plotphon import plot_band_weight

    energies = res.energies * 1000.0  # meV
    nmode = energies.shape[1]
    ax = plot_band_weight(
        [x for _ in range(nmode)],
        [energies[:, i] for i in range(nmode)],
        [res.weights[:, i] for i in range(nmode)],
        xticks=(labels, X),
        ylabel="Energy (meV)",
        ypad=2.0,
    )
    ax.set_title("G-AFM SrMnO$_3$ downfolded magnons (TB2J)")
    ax.figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)

    # seals: Goldstone at Gamma, binary weights along the path
    gamma = int(np.argmin(np.abs(x)))
    assert energies[gamma].min() < 0.05, energies[gamma].min()
    binary = np.minimum(np.abs(res.weights), np.abs(res.weights - 1.0)).max()
    assert binary < 1e-8, binary
    return out_path


if __name__ == "__main__":
    output = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ROOT / "docs" / "static" / "images" / "srmmo3_magnon_unfolded.png"
    )
    print(main(output))
