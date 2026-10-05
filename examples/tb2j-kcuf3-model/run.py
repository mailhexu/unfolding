"""KCuF3-inspired Heisenberg model magnons unfolded to the pseudo-cubic BZ.

Construct the four-Cu type-A-AFM magnetic cell directly as a TB2J Magnon
from Lake et al. (2005) exchanges. This is a model, not a DFT or a full
orbital-order calculation. The flavor-asymmetric variant is illustrative.

Run from a scratch directory: python /path/to/examples/tb2j-kcuf3-model/run.py
"""
import json

import numpy as np

# --- model constants (Lake, Tennant, Nagler, cond-mat/0503128, T=10 K) -----
AP, CP = 4.126, 3.914           # pseudo-cubic parent cell (Angstrom)
# Scale the pair-exchange model to the TB2J JR convention. With Snorm=1/2,
# JR=J_pair/8 makes the isolated AFM chain obey 2*S*J_pair*|sin(pi*k)|;
# the SymPy and numeric check is in derive_chain.py.
JR_SCALE = 0.5**2 / 2
JC_PAIR = -0.034                 # chain AFM pair exchange (eV, TB2J sign)
JP_PAIR = +0.0016                # in-plane FM pair exchange (eV)
JC_EV = JC_PAIR * JR_SCALE       # JR values fed to TB2J
JP_EV = JP_PAIR * JR_SCALE
M_R_INS = 0.011                # INS in-plane zone-boundary scale M at R (eV)
M = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]])   # parent -> I4/mcm (row conv)
FLAVOR_DELTA = 0.15             # illustrative A-C vs B-D chain anisotropy

def build_magnon(ncu=4, flavor_delta=0.0, jc=JC_EV, jp=JP_EV):
    """Construct TB2J Magnon for the KCuF3 cell.

    ncu=4: I4/mcm cell, Cu at tetragonal (0,0,0),(1/2,1/2,0),(0,0,1/2),
    (1/2,1/2,1/2); type-A moments (+,+,-,-) along z.
    ncu=2: minimal magnetic cell 1x1x2 of the parent, same physics.
    """
    from TB2J.magnon.magnon3 import Magnon

    if ncu == 4:
        cell = M @ np.diag([AP, AP, CP])
        taus = np.array([[0, 0, 0], [0.5, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]])
        signs = np.array([1.0, 1.0, -1.0, -1.0])
    elif ncu == 2:
        cell = np.diag([AP, AP, 2 * CP])
        taus = np.array([[0, 0, 0], [0, 0, 0.5]])
        signs = np.array([1.0, -1.0])
    else:
        raise ValueError("ncu must be 2 or 4")
    n = len(taus)
    magmom = np.stack([np.array([0.0, 0.0, s]) for s in signs])  # |m|=1 -> S=1/2


    # bond displacements (fractional, cell frame): pseudo-cubic axes for the
    # 2-Cu cell, tetragonal axes for the I4/mcm cell
    if ncu == 4:
        inplane = [np.array([dx, dy, 0.0]) for dx in (0.5, -0.5) for dy in (0.5, -0.5)]
    else:
        inplane = [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
                   np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])]
    bonds = [(d, jp) for d in inplane]
    bonds += [(np.array([0.0, 0.0, dz]), jc) for dz in (0.5, -0.5)]

    # enumerate ordered pairs (i, j, R): tau_j + R = tau_i + disp
    Rset, entries = set(), []
    for i in range(n):
        for disp, jval in bonds:
            target = taus[i] + disp
            for j in range(n):
                R = target - taus[j]
                if np.allclose(R, np.round(R), atol=1e-8):
                    val = jval
                    if flavor_delta and disp[2] != 0.0 and ncu == 4:
                        # flavor caricature: A-C bonds keep jc, B-D reduced
                        if i in (1, 3) or j in (1, 3):
                            val = jc * (1.0 - flavor_delta)
                    Rset.add(tuple(np.round(R).astype(int)))
                    entries.append((i, j, tuple(np.round(R).astype(int)), val))
                    break
            else:
                raise ValueError(f"unmapped bond from site {i} by {disp}")
    Rlist = np.array(sorted(Rset), dtype=float)
    JR = np.zeros((len(Rlist), n, n, 3, 3))
    ridx = {tuple(r): k for k, r in enumerate(Rlist.astype(int))}
    for i, j, R, val in entries:
        JR[ridx[R], i, j] += val * np.eye(3)

    magnon = Magnon(
        nspin=n, magmom=magmom, Rlist=Rlist, JR=JR, cell=cell,
        _Q=np.zeros(3), _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([1.0, 0.0, 0.0]), pbc=(True, True, True),
        positions=taus @ cell, symbols=["Cu"] * n,
    )
    magnon.set_reference([0, 0, 0], [[0, 0, 1]], [1, 0, 0], magmom)
    return magnon


def bands(magnon, kpts):
    es = magnon.get_magnon_eigenstates(np.asarray(kpts, float))
    return es.energies


def check_1d_limit():
    """In-plane J=0: analytic AFM-chain LSWT w = 2 S |Jc| |sin(pi kz)|."""
    m2 = build_magnon(2, jc=JC_EV, jp=0.0)
    kz = np.linspace(0, 0.5, 7)
    kpts = np.stack([np.zeros_like(kz), np.zeros_like(kz), kz], axis=1)
    got = bands(m2, kpts)[:, -1]
    want = 2 * 0.5 * abs(JC_PAIR) * np.abs(np.sin(np.pi * kz))
    err = np.max(np.abs(got[1:] - want[1:]))
    print(f"[1D] non-Gamma max |BdG - analytic| = {err:.3e} eV; "
          f"Gamma regularization offset = {got[0]:.3e} eV")
    assert err < 5e-6, (got, want)
    assert got[0] < 3e-4, got[0]
    return err


def check_folding():
    """Uniform model: 4-Cu I4/mcm bands at K(q) == 2-Cu bands at both
    in-plane folds of q (the cell is sqrt2 x sqrt2 x 1 of the 2-Cu cell)."""
    m4, m2 = build_magnon(4), build_magnon(2)
    rng = np.random.default_rng(0)
    q = rng.random((12, 3)) * 0.5
    K4 = (q @ M.T) % 1.0
    # 2-Cu cell is diag(1,1,2) of the parent: k2 = q @ diag(1,1,2), mod 1;
    # its in-plane fold partner differs by (1/2,1/2,0) in parent coords
    K2 = np.diag([1.0, 1.0, 2.0])
    q2 = (q @ K2) % 1.0
    q2f = ((q + np.array([0.5, 0.5, 0.0])) @ K2) % 1.0
    e4 = np.sort(bands(m4, K4), axis=1)
    e2 = np.sort(np.concatenate([bands(m2, q2), bands(m2, q2f)], axis=1), axis=1)
    err = np.max(np.abs(e4 - e2))
    print(f"[fold] max |E_4Cu - E_2Cu(2 folds)| = {err:.3e} eV")
    assert err < 1e-10, (e4, e2)
    return err


def qpath():
    pts = [
        (r"$\Gamma$", [0, 0, 0]), ("X", [0.5, 0, 0]), ("M", [0.5, 0.5, 0]),
        (r"$\Gamma$", [0, 0, 0]), ("Z", [0, 0, 0.5]), ("R", [0.5, 0, 0.5]),
        ("A", [0.5, 0.5, 0.5]),
    ]
    q, names, seg = [], [], []
    for i, (nm, p) in enumerate(pts):
        if i:
            a, b = np.array(pts[i - 1][1]), np.array(p)
            for t in np.linspace(0, 1, 40, endpoint=False):
                q.append(a + t * (b - a))
        q.append(np.array(p, float))
        names.append(nm)
        seg.append(len(q) - 1)
    return np.array(q), names, seg


def unfold(flavor_delta):
    from unfolding.tb2j_unfold import magnon_eigendata_from_tb2j
    from unfolding.magnon_unfolder import MagnonUnfolder

    magnon = build_magnon(4, flavor_delta=flavor_delta)
    q, names, seg = qpath()
    ksc = (q @ M.T) % 1.0
    eig = magnon_eigendata_from_tb2j(magnon, ksc)
    unf = MagnonUnfolder(eig, M)
    w = unf.compute(q, resolve_degenerate=1e-5)
    # The sum rule is per mode *across folds*, not across bands at one q.
    folds = unf.fold_kpoints(M)
    wfold = np.stack([unf.compute((q + fold) % 1.0).weights for fold in folds])
    sums = wfold.sum(axis=0)
    offbinary = np.abs(w.weights - np.round(w.weights))
    nonbinary = int(np.count_nonzero(offbinary > 0.05))
    qR = np.array([[0.5, 0.0, 0.5]])
    r = unf.compute(qR, resolve_degenerate=1e-5)
    energy_R = r.energies[0, np.argmax(r.weights[0])]
    qZ = np.array([[0.0, 0.0, 0.5]])
    z = unf.compute(qZ, resolve_degenerate=1e-5)
    energy_Z = z.energies[0, np.argmax(z.weights[0])]
    return dict(q=q, names=names, seg=seg, weights=w, sums=sums,
                nonbinary=nonbinary, energy_R=energy_R, energy_Z=energy_Z)


def main(output_path="kcuf3_unfolded.png", results_path="kcuf3_results.json"):
    results = {}
    results["1d"] = check_1d_limit()
    results["folding"] = check_folding()

    uni = unfold(0.0)
    fla = unfold(FLAVOR_DELTA)
    for label, res in (("uniform", uni), ("flavor", fla)):
        print(f"[{label}] fold-sum deviation = {np.max(np.abs(res['sums']-1.0)):.3e}; "
              f"nonbinary entries = {res['nonbinary']}")
    print(f"AFM Goldstone at parent Z: {uni['energy_Z']*1000:.3f} meV "
          f"(TB2J Cholesky regularization)")
    print(f"R-point LSWT energy: {uni['energy_R']*1000:.3f} meV; "
          f"INS in-plane zone-boundary scale M = {M_R_INS*1000:.1f} meV")
    assert np.max(np.abs(uni['sums']-1.0)) < 1e-10
    assert np.max(np.abs(fla['sums']-1.0)) < 1e-10
    # In the clean model, group-resolved weights are binary away from
    # TB2J's near-Goldstone Cholesky regularization (<1 meV).
    wuni = uni["weights"]
    off = np.abs(wuni.weights - np.round(wuni.weights))
    assert np.max(off[wuni.energies > 0.001]) < 1e-5
    assert fla["nonbinary"] > 0, "flavor variant must mix parent folds"

    # figure: two panels
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, res, title in (
        (axes[0], uni, "KCuF$_3$ uniform Lake model (exact folding)"),
        (axes[1], fla, f"illustrative flavor-asymmetric $J_c$ ($\\delta$={FLAVOR_DELTA})"),
    ):
        w = res["weights"]
        x = np.arange(len(w.kpoints), dtype=float)
        nk, nm = w.energies.shape
        x = np.repeat(x, nm)
        y = (w.energies * 1000).ravel()
        weight = w.weights.ravel()
        visible = weight > 1e-3
        order = np.argsort(weight[visible], kind="stable")  # bright bands on top
        ax.scatter(x[visible][order], y[visible][order],
                   c=weight[visible][order], s=8.0, cmap="viridis",
                   vmin=0, vmax=1)
        for s in res["seg"]:
            ax.axvline(s, color="k", lw=0.5)
        ax.set_xticks(res["seg"])
        ax.set_xticklabels(res["names"])
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, nk - 1)
    axes[0].set_ylabel("Energy (meV)")
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, label="unfolded weight", pad=0.015)
    fig.suptitle("Model magnons only — not a first-principles KCuF$_3$ spectrum")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"figure: {output_path}")

    with open(results_path, "w") as fp:
        json.dump(
            {
                "check_1d_limit_eV": results["1d"],
                "check_folding_eV": results["folding"],
                "uniform_fold_sum_max_dev": float(np.max(np.abs(uni["sums"] - 1.0))),
                "flavor_fold_sum_max_dev": float(np.max(np.abs(fla["sums"] - 1.0))),
                "uniform_nonbinary_entries": uni["nonbinary"],
                "flavor_nonbinary_entries": fla["nonbinary"],
                "goldstone_Z_meV": uni["energy_Z"] * 1000,
                "energy_R_meV": uni["energy_R"] * 1000,
                "M_R_INS_meV": M_R_INS * 1000,
            }, fp, indent=1,
        )
    print("all checks passed")


if __name__ == "__main__":
    main()
