#!/usr/bin/env python
"""Reproduce the ABINIT PAW Gamma-fold figure (abinit_paw_si7p_gamma.png).

Docs example: docs/content/examples/abinit-paw.md in the unfolding repo.
Unfolds the committed Si7P Gamma-point PAW WFK onto the four primitive
folds of supercell Gamma -- G, (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0) --
with the PAW S-metric overlap and, side by side, the pseudo-only coset
fractions. The two panels agree to ~1e-3 here (light-atom Si:P at these
energies); donor-derived states carry the ~1/4 fold weight.

Run from the unpacked bundle directory (this file's directory):

    python reproduce.py

Requires the ``unfolding`` package importable (pip install -e <unfolding
repo checkout>, or PYTHONPATH=<unfolding repo checkout>) plus numpy,
matplotlib, netCDF4 and pypao (the JTH PAW XML reader; HamiltonIO pulls
it in). No ABINIT run is needed: data/ carries the committed netCDF WFKs
and the JTH XML datasets.

The dense GXWGLWX path figure is NOT reproducible from this bundle: its
WFKs are ~250 MB each. Produce them from inputs/*_paw_path.abi with your
own ABINIT (see regenerate_path.sh and README.txt), place the
*_DS2_WFK.nc files in data/, and use the unfold-path recipe documented
on the docs page.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = M @ prim
XML = {symbol: DATA / f"{symbol}.xml" for symbol in ("Si", "P")}
# Degenerate supercell states are stored by ABINIT in an arbitrary gauge;
# resolve eigenspaces (1e-3 eV, as in the NC ABINIT route) so the plotted
# per-band weights are gauge-invariant reference-band populations.
DEGEN_TOL_EV = 1e-3


def main(output=None):
    from HamiltonIO.abinit import HARTREE_TO_EV, read_paw_wfk
    from unfolding.abinit_paw import unfold_abinit_paw

    output = Path(output) if output else HERE / "abinit_paw_si7p_gamma.png"
    primitive = DATA / "si_primitive_foldso_WFK.nc"
    defect = read_paw_wfk(DATA / "si7p_gammao_WFK.nc", XML)
    result = unfold_abinit_paw(defect, primitive, XML, M,
                               resolve_degenerate=DEGEN_TOL_EV / HARTREE_TO_EV)
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
