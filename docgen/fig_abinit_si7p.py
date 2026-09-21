#!/usr/bin/env python
"""Render the ABINIT Si7P Gamma--X unfolded spectral-weight figure.

The committed checkout uses the small WFK fixture. A dense 300-point WFK
regenerated on nic6 is supported automatically when supplied at the same path.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "abinit_si"
WFK = DATA / "si7p_gamma_x_patho_DS2_WFK.nc"
MATRIX = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int)


def main(out_path):
    """Render the available Si7P WFK path to ``out_path``."""
    import matplotlib.pyplot as plt

    from unfolding.abinit_unfold import HARTREE_TO_EV, read_wfk, unfold_abinit
    from unfolding.pw_unfolder import PWUnfolder

    if not WFK.is_file():
        raise FileNotFoundError(f"ABINIT Si7P path WFK not found: {WFK}")

    data = read_wfk(WFK)
    # Use every stored path point. The dense nic6 regeneration has 300 points;
    # the small committed fixture has four unique points plus the periodic X
    # endpoint, and remains a valid CI/docgen fallback.
    kpts = np.mod(data.kpoints @ np.linalg.inv(MATRIX.T), 1.0)
    xqpts = np.linspace(0.0, 1.0, len(kpts))
    result = PWUnfolder(data, MATRIX).compute(kpts)
    ax = unfold_abinit(
        data=data,
        unfold_sc_mat=MATRIX,
        kpts=kpts,
        knames=[r"$\Gamma$", "X"],
        xqpts=xqpts,
        Xqpts=[0.0, 1.0],
        ylabel=r"Energy relative to $E_F$ (eV)",
        ypad=1.5,
        color="navy",
    )

    energies = result.eigenvalues * HARTREE_TO_EV - data.fermi_energy * HARTREE_TO_EV
    xmesh = np.broadcast_to(xqpts[:, None], energies.shape)
    donor_window = (np.abs(energies) < 0.5) & (result.weights < 0.7)
    if np.any(donor_window):
        ax.scatter(
            xmesh[donor_window],
            energies[donor_window],
            s=14,
            c="crimson",
            marker="o",
            linewidths=0,
            zorder=6,
            label="fractional donor-window weight",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title("ABINIT Si$_7$P unfolded spectral weight")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "static" / "images" / "si7p_abinit_unfolded.png"
    print(main(output))
