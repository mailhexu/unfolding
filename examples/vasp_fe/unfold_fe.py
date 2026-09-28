#!/usr/bin/env python
"""Unfold the bcc-Fe 2x2x2 conventional supercell onto the primitive path.

Runs the PAW-metric unfolding of a real VASP supercell NSCF (ICHARG=11)
along the dense primitive bcc path Gamma-H-N-Gamma-P-H (250 points,
K_sc = k_prim @ M.T with M = 2*[[0,1,1],[1,0,1],[1,1,0]], det 16) and
draws the spin-resolved weighted band map, overlaying the independently
computed primitive-cell bands (same POTCAR/ENCUT, ICHARG from its own
SCF). Also reports the numeric consistency checks: PAW norm residuals,
sector binarity of the unfolded weights, and the maximum supercell-minus-
primitive branch energy deviation on weight-1 branches after E_F alignment.

Private licensed inputs (never redistributed):

- UNFOLDING_VASP_FE_SEED: directory holding the PAW_PBE Fe POTCAR.
- UNFOLDING_VASP_FE_RUNS: directory with the sc16_scf, sc16_nscf,
  prim_scf and prim_nscf run directories from make_inputs.py + vasp
  (default /tmp/unfolding_vasp_fe).

Run from the repository root:
    UNFOLDING_VASP_FE_SEED=... python examples/vasp_fe/unfold_fe.py
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_inputs import MATRIX, PATH, primitive_path

DEGEN_TOL_EV = 1e-3  # gauge-invariant degenerate-group weights, as in the
# Si examples; wider tolerances over-group sector crossings and mis-assign
SIGMA_EV = 0.045  # house Gaussian map width (docgen/abinit_si_common.py)
WINDOW_EV = (-8.0, 8.0)  # metallic Fe: d bands + sp states around E_F


def read_efermi(outcar):
    match = re.search(r"E-fermi\s*:\s*([-\d.]+)", Path(outcar).read_text())
    if not match:
        raise ValueError(f"no E-fermi in {outcar}")
    return float(match.group(1))


def spectral_map(energies, weights, egrid):
    pre = 1.0 / (SIGMA_EV * np.sqrt(2 * np.pi))
    A = np.zeros((len(energies), len(egrid)))
    for ik in range(len(energies)):
        for eb, wb in zip(energies[ik], weights[ik]):
            A[ik] += wb * pre * np.exp(-0.5 * ((egrid - eb) / SIGMA_EV) ** 2)
    return A


def consistency(res, prim_e, ef_sc, ef_prim):
    """Binarity, completeness and unfolded-vs-primitive branch deviations."""
    weights = np.clip(res.weights, 0.0, 1.0)
    energies = res.eigenvalues - ef_sc
    inside = (energies > WINDOW_EV[0]) & (energies < WINDOW_EV[1])
    sector_gap = np.minimum(weights, np.abs(1 - weights))
    off_sector = sector_gap[inside & (sector_gap > 1e-8)]
    # unfolded-minus-primitive deviation on weight-1 branches: pair each
    # weight-1 supercell state with its closest primitive branch (the
    # 16-fold bank is index-shifted), primitive shifted by the E_F
    # difference as in the figure.  The statistic stays inside the plot
    # window: the top of the finite SC bank is Davidson-unconverged and
    # its branch energies are meaningless there.
    hi = weights > 0.9
    prim_rel = prim_e - ef_prim  # same E_F frame as `energies`
    dev = []
    for ik in range(len(energies)):
        for ib in np.flatnonzero(hi[ik]):
            if not WINDOW_EV[0] < energies[ik, ib] < WINDOW_EV[1]:
                continue
            nearest = np.argmin(np.abs(prim_rel[ik] - energies[ik, ib]))
            dev.append(energies[ik, ib] - prim_rel[ik, nearest])
    dev = np.asarray(dev)
    shift = float(np.median(dev))
    aligned = np.abs(dev - shift)
    return {
        "max_norm_residual": float(np.max(np.abs(res.norm_residuals))),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "off_sector_max": float(off_sector.max()) if off_sector.size else 0.0,
        "off_sector_fraction": float(
            np.mean(sector_gap[inside] > 0.05)),
        "ef_gap_supercell_minus_primitive": float(ef_sc - ef_prim),
        "median_branch_shift": shift,
        "max_abs_dev_aligned_weight1_window": float(aligned.max()),
        "rms_dev_aligned_weight1_window": float(np.sqrt(np.mean(aligned**2))),
    }


def main(figure=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from unfolding.vasp_paw import unfold_vasp_paw

    from read_wavecar_ordered import read_wavecar_ordered

    seed = Path(os.environ["UNFOLDING_VASP_FE_SEED"])
    runs = Path(os.environ.get("UNFOLDING_VASP_FE_RUNS",
                               "/tmp/unfolding_vasp_fe"))
    potcar = seed / "POTCAR"
    sc = runs / "sc16_nscf"
    prim = runs / "prim_nscf"
    for need in (potcar, sc / "WAVECAR", prim / "WAVECAR",
                 runs / "sc16_scf/OUTCAR", runs / "prim_scf/OUTCAR"):
        if not need.is_file():
            raise FileNotFoundError(f"missing private fixture: {need}")

    ef_sc = read_efermi(runs / "sc16_scf/OUTCAR")
    ef_prim = read_efermi(runs / "prim_scf/OUTCAR")
    sc_data = read_wavecar_ordered(sc / "WAVECAR", sc / "POSCAR")
    prim_data = read_wavecar_ordered(prim / "WAVECAR", prim / "POSCAR")

    kprim, x, X, labels = primitive_path()
    if not np.allclose(np.mod(prim_data.kpoints, 1.0), np.mod(kprim, 1.0),
                       atol=1e-7):
        raise ValueError("primitive WAVECAR k-points differ from the "
                         f"{PATH} path of make_inputs.py")

    results, prim_curves = {}, {}
    stats = {}
    prim_eigs = np.array(prim_data.eigenvalues)  # (nk, nspin, nb)
    for spin in (0, 1):
        res = unfold_vasp_paw(sc_data, prim_data, potcar, MATRIX, spin=spin,
                              resolve_degenerate=DEGEN_TOL_EV)
        results[spin] = res
        prim_curves[spin] = prim_eigs[:, spin, :]
        stats[spin] = consistency(res, prim_curves[spin], ef_sc, ef_prim)

    figure = Path(figure) if figure else \
        ROOT / "docs/static/images/vasp_fe_path.png"
    egrid = np.linspace(WINDOW_EV[0] - 0.8, WINDOW_EV[1] + 0.8, 1100)
    names = [r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, spin in zip(axes, (0, 1)):
        res = results[spin]
        energies = res.eigenvalues - ef_sc
        weights = np.clip(res.weights, 0.0, 1.0)
        A = spectral_map(energies, weights, egrid)
        ax.pcolormesh(x, egrid, A.T, cmap="Blues", vmin=0.0, vmax=2.0,
                      shading="auto", rasterized=True)
        ax.plot(x, prim_curves[spin] - ef_prim, color="crimson",
                lw=1.0, alpha=0.9, zorder=5, label="primitive-cell bands")
        for xt in X[1:-1]:
            ax.axvline(xt, color="gray", lw=0.5)
        ax.axhline(0.0, ls="--", color="k", lw=0.7)
        ax.set_xticks(list(X))
        ax.set_xticklabels(names)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(*WINDOW_EV)
        ax.set_title(r"$2{\times}2{\times}2$ bcc Fe unfolded, "
                     rf"spin {'up' if spin == 0 else 'down'}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    axes[0].set_ylabel(r"Energy relative to $E_F$ (eV)")
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)

    np.savez(Path(__file__).resolve().parent / "vasp_fe_path_result.npz",
             kprim=kprim, x=x, X=X, labels=np.array(labels),
             eigenvalues=np.stack([results[s].eigenvalues for s in (0, 1)]),
             weights=np.stack([results[s].weights for s in (0, 1)]),
             pseudo_weights=np.stack(
                 [results[s].pseudo_weights for s in (0, 1)]),
             norm_residuals=np.stack(
                 [results[s].norm_residuals for s in (0, 1)]),
             prim_eigenvalues=np.stack([prim_curves[s] for s in (0, 1)]),
             ef_sc=ef_sc, ef_prim=ef_prim,
             stats=np.array([stats[0], stats[1]], dtype=object),
             allow_pickle=True)

    print(f"E_F supercell {ef_sc:.4f} eV, primitive {ef_prim:.4f} eV, "
          f"gap {ef_sc - ef_prim:+.4f} eV")
    for spin in (0, 1):
        s = stats[spin]
        print(f"spin {spin}: PAW norm residual max {s['max_norm_residual']:.2e}; "
              f"weights in [{s['min_weight']:.2e}, {1 - s['max_weight']:.2e} "
              f"from 0/1]; off-sector max {s['off_sector_max']:.2e}; "
              f"interstitial fraction {s['off_sector_fraction']:.2e}")
        print(f"spin {spin}: weight-1 branch deviation after shift "
              f"{s['median_branch_shift']:+.4f} eV: max "
              f"{s['max_abs_dev_aligned_weight1_window']:.4f} eV, rms "
              f"{s['rms_dev_aligned_weight1_window']:.4f} eV "
              f"(window {list(WINDOW_EV)} eV)")
    print("wrote", figure)
    return stats


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
