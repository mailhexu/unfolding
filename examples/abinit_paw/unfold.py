"""Render PAW-corrected and pseudo-only Si7P Gamma-fold spectral weights.

Run from the unfolding repository: python examples/abinit_paw/unfold.py
The JTH XML and real ABINIT WFK fixtures live in tests/data/abinit_paw.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from HamiltonIO.abinit import HARTREE_TO_EV, read_paw_wfk
from unfolding.abinit_paw import unfold_abinit_paw

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/abinit_paw"
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
XML = {symbol: DATA / f"{symbol}.xml" for symbol in ("Si", "P")}


def main(output=None):
    output = Path(output) if output else ROOT / "docs/static/images/abinit_paw_si7p.png"
    primitive = DATA / "si_primitive_foldso_WFK.nc"
    defect = read_paw_wfk(DATA / "si7p_gammao_WFK.nc", XML)
    result = unfold_abinit_paw(defect, primitive, XML, M)
    fermi = defect.wavefunctions.fermi_energy
    energies = (result.eigenvalues - fermi) * HARTREE_TO_EV
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    labels = ["Γ", "(0,½,½)", "(½,0,½)", "(½,½,0)"]
    for ax, weights, title in zip(axes,
                                  (result.weights, result.pseudo_weights),
                                  ("PAW S-metric reference", "Pseudo-only coset")):
        for i, (row, e) in enumerate(zip(weights, energies)):
            ax.scatter(np.full(len(row), i), e, s=130 * np.clip(row, 0, 1),
                       c="navy", alpha=np.clip(row, 0.04, 1))
        ax.set_xticks(range(4), labels, rotation=30)
        ax.set_xlim(-.5, 3.5)
        ax.set_title(title)
    axes[0].set_ylabel("Energy relative to Fermi level (eV)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print("maximum PAW norm error:", np.max(np.abs(result.norm_residuals)))
    print("max PAW-pseudo weight difference:",
          np.max(np.abs(result.weights - result.pseudo_weights)))
    print("wrote", output)
    return result


if __name__ == "__main__":
    main()
