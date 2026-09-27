"""Unfold real ABACUS Si7P plane-wave Gamma states into four primitive folds.

Run from the unfolding repository:
python examples/abacus_si_pw/unfold_doped.py
The four primitive momenta all map to supercell Gamma. We plot each fold
separately rather than implying that a Gamma-only calculation sampled a path.
Weights are pseudo-wavefunction coset fractions, not PAW all-electron weights.
Energies are in eV relative to the run's Fermi level (the donor level
sits at E = 0 in this high-concentration 8-atom cell), in the house
[-13, 8] eV window.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from HamiltonIO.abacus.pw_wfc import AbacusPWParser
from unfolding.pw_unfolder import PWEigenData, PWUnfolder

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/abacus_example/si7p_pw/OUT.si7p_pw"
OUTPUT = ROOT / "docs/static/images/abacus_si_p_pw_unfolded.png"
MATRIX = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
FOLDS = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def main(output=OUTPUT):
    data = AbacusPWParser(DATA).read()
    eigen = PWEigenData(
        data.kpoints, data.gvecs,
        tuple(c[None, :, None, :] for c in data.coefficients),
        data.eigenvalues[:, None, :],
    )
    result = PWUnfolder(eigen, MATRIX).compute(FOLDS)
    energies = result.eigenvalues - data.efermi
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (energies_i, weights) in enumerate(zip(energies, result.weights)):
        ax.scatter(np.full(len(energies_i), i), energies_i,
                   s=140 * weights, c="navy", alpha=np.clip(weights, 0.03, 1.0))
    ax.axhline(0.0, linestyle="--", color="black", linewidth=0.8)
    ax.set_xticks(range(4), ["Γ", "(0,½,½)", "(½,0,½)", "(½,½,0)"])
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(-13.0, 8.0)
    ax.set_ylabel("Energy (eV)")
    ax.set_title("ABACUS plane-wave Si:P: four primitive folds of supercell Γ"
                 "\n(E$_F$ = 0)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print("E_Fermi (eV):", data.efermi)
    print("maximum four-fold sum-rule error:", np.max(np.abs(result.weights.sum(axis=0) - 1)))
    print("donor-band (at E_F) fold weights:", result.weights[:, 16])
    print("wrote", output)
    return result


if __name__ == "__main__":
    main()
