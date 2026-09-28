"""Unfold the PAW dense-path WFKs onto the house-style GXWGLWX figure.

Run from the unfolding repository: python examples/abinit_paw/unfold_path.py
The fixtures are produced by tests/data/abinit_paw/regenerate_path_on_nic6.sh:
a pristine Si8 and a Si7P supercell NSCF over the 305-point path (dataset 2
k list identical to the validated NC deck), plus a Si2 primitive reference
WFK on the same path that provides the embedded reference banks. The SC
k-list is K = k_prim @ M.T, so the NSCF k-list IS the path: each stored
supercell state contributes one primitive momentum sector per path point.
The DS2 WFKs are large and stay external (gitignored).
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
DEGEN_TOL_EV = 1e-3
DEGEN_HA = DEGEN_TOL_EV / HARTREE_TO_EV
ACELL = 10.26  # bohr, conventional cubic cell
PATH = "GXWGLWX"
KNAMES = [r"$\Gamma$", "X", "W", r"$\Gamma$", "L", "W", "X"]
# fcc special points in primitive reciprocal fractional coordinates
# (Setyawan-Curtarolo, as in the SIESTA and ABINIT NC examples).
SPECIAL = {"G": (0.0, 0.0, 0.0), "X": (0.5, 0.0, 0.5),
           "W": (0.5, 0.25, 0.75), "L": (0.5, 0.5, 0.5)}
YRANGE = (-13.0, 8.0)
# The top of the 32-band manifold (E above ~7 eV) sits at the tail of the
# NSCF Davidson subspace: its states are sector-pure but not fully
# converged, so per-band weights go fuzzy there. Weight statistics and the
# figure's data therefore trust energies below CONVERGED_EV; the plotted
# window keeps the house-style [-13, 8] eV frame (the NC route's 24 bands
# reach only ~4.6 eV).
CONVERGED_EV = 6.0
NC_DENSE = ROOT / "tests/data/abinit_si/si8_gxwglwxo_DS2_WFK.nc"


def _bcart():
    """Primitive reciprocal Cartesian basis (cycles/bohr, no 2*pi in k)."""
    prim = np.linalg.inv(M.astype(float)) @ (np.eye(3) * ACELL)
    return 2 * np.pi * np.linalg.inv(prim).T


def path_positions(prim_kpoints):
    """Cumulative Cartesian distance along the stored primitive path."""
    cart = np.asarray(prim_kpoints) @ _bcart()
    return np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(cart, axis=0), axis=1))])


def path_ticks(prim_kpoints, x):
    """X positions of the Γ, X, W, Γ, L, W, X labels on the stored path."""
    ticks, start = [], 0
    for name in PATH:
        target = np.mod(SPECIAL[name], 1.0)
        for i in range(start, len(prim_kpoints)):
            delta = np.abs((np.mod(prim_kpoints[i], 1.0) - target + .5) % 1.0 - .5).max()
            if delta < 1e-6:
                ticks.append(x[i])
                start = i + 1
                break
        else:
            raise ValueError(f"special point {name} not found on the stored path")
    return ticks


def unfold_system(sc_wfk, primitive):
    """(PawWeights, energies in eV relative to the supercell Fermi level)."""
    supercell = read_paw_wfk(sc_wfk, XML)
    result = unfold_abinit_paw(supercell, primitive, XML, M,
                               resolve_degenerate=DEGEN_HA)
    energies = (result.eigenvalues - supercell.wavefunctions.fermi_energy) * HARTREE_TO_EV
    return result, energies


def _nondegenerate_bands(result):
    """Yield (ik, band) for SC bands outside degenerate eigenspaces."""
    for ik in range(len(result.eigenvalues)):
        e = result.eigenvalues[ik]
        boundaries = np.r_[0, np.flatnonzero(np.diff(e) > DEGEN_HA) + 1, len(e)]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if b - a == 1:
                yield ik, a


def binary_error(result, energies, window=YRANGE):
    """max |w(1-w)| over non-degenerate SC bands with energy in a window.

    ``energies`` are eV relative to E_F (same shape as the weights).
    Pristine sectors never mix, so this stays ~1e-6 below the Davidson
    tail; in the doped system sector states anticross at band crossings
    (the impurity couples sectors), so isolated weights go fractional
    there while whole valence regions stay binary.
    """
    worst = 0.0
    for ik, ib in _nondegenerate_bands(result):
        if not window[0] < energies[ik, ib] < window[1]:
            continue
        w = np.clip(result.weights[ik, ib], 0.0, 1.0)
        worst = max(worst, abs(w * (1.0 - w)))
    return worst


def binary_fraction(result, energies, window=YRANGE, tol=1e-3):
    """Fraction of non-degenerate band points with binary weight in window."""
    total = binary = 0
    for ik, ib in _nondegenerate_bands(result):
        if not window[0] < energies[ik, ib] < window[1]:
            continue
        w = np.clip(result.weights[ik, ib], 0.0, 1.0)
        total += 1
        binary += abs(w * (1.0 - w)) < tol
    return binary / max(total, 1)


def fractional_weights(result, energies, window=YRANGE):
    """(energies, weights) of fractional non-degenerate bands in a window.

    For pristine Si every non-degenerate band is binary; a substitutional
    donor leaves host-like bands binary and donor-derived bands carrying
    the ~1/4 sector weight at generic k.
    """
    e_out, w_out = [], []
    for ik, ib in _nondegenerate_bands(result):
        e, w = energies[ik, ib], np.clip(result.weights[ik, ib], 0.0, 1.0)
        if window[0] < e < window[1] and 0.05 < w < 0.95:
            e_out.append(e)
            w_out.append(w)
    return np.asarray(e_out), np.asarray(w_out)


def compare_norm_conserving(paw_result, paw_energies, nc_wfk=NC_DENSE):
    """PAW vs NC unfolded band positions on the shared path.

    Both routes sample the same 305-point path; sectors with spectral
    weight above 0.9 (binary bands) inside the plot window are matched by
    sorted energy at each path point. Returns (max |dE| after the global
    median shift, the shift, median |dE| of matched pairs). Sorting can
    mispair neighbouring conduction bands at crossings, which shows up in
    the max but not the median.
    """
    from HamiltonIO.abinit import read_wfk
    from unfolding.pw_unfolder import PWEigenData, PWUnfolder

    data = read_wfk(nc_wfk)
    stored_prim = np.mod(data.kpoints @ np.linalg.inv(M.T), 1.0)
    paw_prim = np.mod(paw_result.kpoints, 1.0)
    delta = np.abs((stored_prim[:, None, :] - paw_prim[None, :, :] + .5) % 1.0 - .5)
    dist = delta.max(axis=2)  # (nc point, paw point) sup-norm distance
    if dist.min(axis=1).max() > 1e-6:
        raise ValueError("NC and PAW path WFKs do not share the k list")
    order = dist.argmin(axis=1)  # PAW path index for each NC stored point

    nc = PWUnfolder(
        PWEigenData(data.kpoints, data.gvecs, data.coefficients, data.eigenvalues),
        M,
    ).compute(stored_prim, resolve_degenerate=DEGEN_HA)
    nc_e = nc.eigenvalues * HARTREE_TO_EV - data.fermi_energy * HARTREE_TO_EV
    nc_w = np.clip(nc.weights, 0.0, 1.0)
    paw_w = np.clip(paw_result.weights, 0.0, 1.0)
    devs = []
    for j in range(len(stored_prim)):
        i = order[j]  # PAW path point stored at the same momenta as NC point j
        en = nc_e[j][(nc_w[j] > 0.9) & (YRANGE[0] < nc_e[j]) & (nc_e[j] < YRANGE[1])]
        ep = paw_energies[i][(paw_w[i] > 0.9) & (YRANGE[0] < paw_energies[i])
                             & (paw_energies[i] < YRANGE[1])]
        n = min(len(en), len(ep))
        if n:
            devs.append(np.sort(en)[:n] - np.sort(ep)[:n])
    if not devs:
        raise ValueError("no high-weight bands matched between PAW and NC")
    devs = np.concatenate(devs)
    shift = float(np.median(devs))
    return float(np.max(np.abs(devs - shift))), shift, float(np.median(np.abs(devs - shift)))


def _draw(ax, x, energies, weights, ticks, title, ylabel=""):
    from unfolding.plotphon import plot_band_weight

    nband = energies.shape[1]
    plot_band_weight(
        [x] * nband,
        [energies[:, ib] for ib in range(nband)],
        [np.clip(weights[:, ib], 0.0, 1.0) for ib in range(nband)],
        efermi=0, yrange=YRANGE, axis=ax, title=title,
        xticks=[KNAMES, ticks], ylabel=ylabel,
    )


def main(output=None):
    output = Path(output) if output else ROOT / "docs/static/images/abinit_paw_si_path.png"
    primitive = read_paw_wfk(DATA / "si_prim_paw_patho_DS2_WFK.nc", XML)
    prim_k = primitive.wavefunctions.kpoints
    x = path_positions(prim_k)
    ticks = path_ticks(prim_k, x)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    systems = [
        ("si8_paw_patho_DS2_WFK.nc", "Si$_8$ pristine (PAW)"),
        ("si7p_paw_patho_DS2_WFK.nc", "Si$_7$P doped (PAW)"),
    ]
    for ax, (name, title), ylabel in zip(
            axes, systems,
            ("Energy relative to $E_F$ (eV)", "")):
        result, energies = unfold_system(DATA / name, primitive)
        _draw(ax, x, energies, result.weights, ticks, title, ylabel=ylabel)
        if ax is axes[0]:
            from matplotlib.lines import Line2D

            ax.legend(
                handles=[Line2D([0], [0], color="blue", lw=2,
                                label="unfolded spectral weight")],
                loc="upper right", fontsize=8, framealpha=0.85,
            )
        window = (YRANGE[0], CONVERGED_EV)
        print(f"{name}: max |<psi|S|psi>-1| = "
              f"{np.max(np.abs(result.norm_residuals)):.2e}")
        print(f"{name}: max non-degenerate |w(1-w)| below {CONVERGED_EV} eV = "
              f"{binary_error(result, energies, window=window):.2e}")
        print(f"{name}: binary fraction below {CONVERGED_EV} eV = "
              f"{binary_fraction(result, energies, window=window):.4f}")
        frac_e, frac_w = fractional_weights(result, energies, window=window)
        if frac_w.size:
            print(f"{name}: {frac_w.size} fractional bands, weights in "
                  f"[{frac_w.min():.3f}, {frac_w.max():.3f}]")
        else:
            print(f"{name}: no fractional bands")
        if "si8" in name and NC_DENSE.is_file():
            try:
                max_dE, shift, med_dE = compare_norm_conserving(result, energies)
                print(f"PAW vs NC unfolded bands: median shift {shift:+.3f} eV, "
                      f"max |dE| {max_dE:.3f} eV, median |dE| {med_dE:.3f} eV")
            except (ValueError, OSError) as exc:
                print(f"PAW vs NC comparison skipped: {exc}")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print("wrote", output)
    return output


if __name__ == "__main__":
    main()
