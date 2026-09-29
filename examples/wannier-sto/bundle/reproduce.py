#!/usr/bin/env python
"""Reproduce the 'Wannier90 SrTiO3' example page.

Real-data route (the page's headline figures): run the bundled Wannier90
datasets (data/pristine, data/ti_vacancy) through the package's built-in
Wannier90 reader and WannierUnfolder:

    python reproduce.py --real pristine        # -> sto_unfolded.png
    python reproduce.py --real ti_vacancy      # -> sto_ti_vacancy.png

Synthetic smoke route: out of the box the bundled data/example_hr.dat is
a small SYNTHETIC SrTiO3-like tight-binding model (Ti t2g conduction
bands: 3 orbitals on a simple-cubic Ti sublattice, 2x2x2 supercell; see
README.txt) so the script is demonstrably runnable without DFT: it then
parses the hr file with a minimal built-in Wannier90 reader and drives
the documented WannierUnfolder low-level route, which needs only numpy.

    python reproduce.py                      # pristine -> sto_unfolded.png
    python reproduce.py --defect             # impurity  -> sto_defect.png
    python reproduce.py --hr PATH            # your own wannier90_hr.dat
    python reproduce.py --engine minimulti   # force the MyTB route

Output: a weight-coded unfolded band figure in the working directory.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: figure is saved, never shown

import numpy as np

# Synthetic model geometry (the bundled data/example_hr.dat)
A0 = 1.0                      # cubic lattice constant (arbitrary units)
SCMAT = np.diag([2, 2, 2])    # supercell the hr file describes
L_SC = 2                      # supercell edge in primitive cells
T_HOP = 1.0                   # nearest-neighbour t2g hopping (eV)
DEFECT_POTENTIAL = 6.0        # on-site potential of the impurity site (eV)
DEFECT_COUPLING = 0.3         # surviving fraction of the impurity hoppings

KVECTORS = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
KNAMES = [r"$\Gamma$", "X", "M", r"$\Gamma$", "R"]
LABELS = ["d_xy", "d_yz", "d_zx"] * (L_SC ** 3)   # one per supercell orbital


# ---------------------------------------------------------------------------
# Synthetic SrTiO3-like t2g model -> data/example_hr.dat
# ---------------------------------------------------------------------------
def build_t2g_hr(L=L_SC, t=T_HOP):
    """Real-space t2g Hamiltonian of a simple-cubic Ti lattice.

    Primitive cell: one Ti site with d_xy / d_yz / d_zx orbitals (eps=0).
    Nearest-neighbour hopping t along the two axes each orbital bonds in
    (d_xy -> x,y; d_yz -> y,z; d_zx -> x,z): the standard cubic STO
    conduction-band model.  The model cell is an L^3 supercell, so each
    supercell-lattice translation R has a (3 L^3) x (3 L^3) block; only
    R in {-1,0,1}^3 carry weight.
    """
    sites = [(ix, iy, iz) for ix in range(L) for iy in range(L) for iz in range(L)]
    index = {s: i for i, s in enumerate(sites)}
    norb = 3 * len(sites)
    orbitals = [("d_xy", (0, 1)), ("d_yz", (1, 2)), ("d_zx", (0, 2))]

    def orb(site, o):
        return 3 * index[site] + o

    blocks = {}

    def add(R, i, j, val):
        blocks.setdefault(tuple(R), np.zeros((norb, norb)))[i, j] += val

    steps = [np.array(v, dtype=int) for v in
             ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    for o, (_name, axes) in enumerate(orbitals):
        for a in axes:
            for d in steps:
                if a not in np.flatnonzero(d):
                    continue
                for s in sites:
                    n = np.array(s) + d              # neighbour, may stick out
                    nw = tuple(n % L)                # wrapped into the cell
                    R = tuple((n - np.array(nw)) // L)
                    add(R, orb(nw, o), orb(s, o), t)
    return blocks, norb


def write_hr_dat(path, blocks, norb):
    """Write a canonical Wannier90-format wannier90_hr.dat."""
    Rs = sorted(blocks)
    with open(path, "w") as fh:
        fh.write("%d\n" % norb)
        deg = ["    1.00000000000000E+000"] * len(Rs)
        for start in range(0, len(deg), 15):
            fh.write("".join("%28s" % v for v in deg[start:start + 15]) + "\n")
        for R in Rs:
            h = blocks[R]
            for m in range(norb):
                for n in range(norb):
                    fh.write("%4d%4d%4d%4d%4d %16.10f %16.10f\n"
                             % (R[0], R[1], R[2], m + 1, n + 1, h[m, n], 0.0))
    print("wrote %s (%d orbitals, %d R vectors)" % (path, norb, len(Rs)))


def orbital_positions(L=L_SC):
    """Supercell-fractional orbital positions; order matches the model
    (orbital index = 3 * site + o, all three orbitals at the site)."""
    pos = []
    for (ix, iy, iz) in [(ix, iy, iz) for ix in range(L) for iy in range(L)
                         for iz in range(L)]:
        p = (ix / L, iy / L, iz / L)
        pos.extend([p, p, p])
    return np.array(pos)


# ---------------------------------------------------------------------------
# Minimal Wannier90 _hr.dat reader + the documented model interface
# ---------------------------------------------------------------------------
def read_hr_dat(path):
    """Parse wannier90_hr.dat -> (R array (NR,3) float, H array (NR,n,n))."""
    def is_int(tok):
        return tok.lstrip("+-").isdigit()

    with open(path) as fh:
        lines = fh.readlines()
    norb = int(lines[0].split()[0])
    blocks = {}
    ndata = 0
    for line in lines[1:]:
        tok = line.split()
        if len(tok) != 7 or not all(is_int(t) for t in tok[:5]):
            continue              # degeneracy block / wrapped header / blank
        rx, ry, rz, m, n = (int(t) for t in tok[:5])
        blocks.setdefault((rx, ry, rz), np.zeros((norb, norb), dtype=complex)
                          )[m - 1, n - 1] = float(tok[5]) + 1j * float(tok[6])
        ndata += 1
    if not blocks:
        raise ValueError("no R-matrix entries found in %s" % path)
    nR = ndata // (norb * norb)
    if nR * norb * norb != ndata:
        raise ValueError("truncated hr file: %d data lines for %d orbitals"
                         % (ndata, norb))
    Rs = np.array(sorted(blocks), dtype=float)
    if len(blocks) != nR:
        raise ValueError("inconsistent hr file: %d blocks for %d R vectors"
                         % (len(blocks), nR))
    H = np.stack([blocks[tuple(R)] for R in Rs])
    return Rs, H


class SimpleTB(object):
    """Tight-binding model exposing the WannierUnfolder interface:
    `.atoms` (ASE Atoms of the primitive cell), `._orb` (supercell-
    fractional orbital positions) and `.solve_all(k_list=, eig_vectors=)`
    returning (evals [nband, nk], evecs [nband, nk, norb]).  Same
    convention as minimulti's MyTB.

    This minimal implementation is hardcoded to the bundled synthetic
    geometry (3 orbitals per site, cubic L^3 supercell of one-site cells).
    For real multi-orbital Wannier90 directories use the minimulti engine.
    """

    def __init__(self, hr_path, sc_matrix=SCMAT, cell=A0 * np.eye(3), L=L_SC):
        from ase import Atoms
        self.Rs, self.H = read_hr_dat(hr_path)
        self.norb = self.H.shape[1]
        self.scmatrix = np.array(sc_matrix, dtype=float)
        self.atoms = Atoms("Ti", cell=cell, pbc=True, positions=[(0, 0, 0)])
        self._orb = orbital_positions(L)
        self.iR0 = int(np.argmin(np.abs(self.Rs).sum(axis=1)))
        if self._orb.shape[0] != self.norb:
            raise ValueError("hr file has %d orbitals, synthetic geometry "
                             "expects %d" % (self.norb, self._orb.shape[0]))

    def solve_all(self, k_list, eig_vectors=False):
        k = np.atleast_2d(np.asarray(k_list, dtype=float))
        phase = np.exp(2j * np.pi * k @ self.Rs.T)          # (nk, NR)
        hk = np.einsum("kr,rmn->kmn", phase, self.H)        # (nk, n, n)
        hk = 0.5 * (hk + np.conj(np.transpose(hk, (0, 2, 1))))
        evals, evecs = np.linalg.eigh(hk)
        if not eig_vectors:
            return evals.T
        return evals.T, np.transpose(evecs, (1, 0, 2))      # (nband, nk, norb)


def apply_defect(model, potential=DEFECT_POTENTIAL, coupling=DEFECT_COUPLING,
                 site=0):
    """Weaken one site into an impurity (in place).

    The hoppings to `site`'s orbitals are reduced to `coupling` of their
    value and the on-site energy is raised, so partly-localized impurity
    levels split off inside the t2g band region.  The orbital lattice
    (positions/labels) is unchanged - exactly how a Wannier model of a
    defective supercell keeps all its basis functions - so the unfolding
    shows reduced, fractional weights on the impurity-derived states while
    the host bands stay at weight 1.
    """
    bad = [3 * site + o for o in range(3)]
    mask = np.zeros(model.norb, dtype=bool)
    mask[bad] = True
    edge = mask[:, None] ^ mask[None, :]          # exactly one end at the site
    for h in model.H:
        h[edge] *= coupling
    for b in bad:
        model.H[model.iR0][b, b] = potential
    print("defect: site %d orbitals weakened (coupling %.2f), on-site "
          "potential %.2f eV" % (site, coupling, potential))


# ---------------------------------------------------------------------------
# The two engines
# ---------------------------------------------------------------------------
def run_minimulti(hr_dir, prefix, out_png, npoints):
    """The page's documented convenience driver (requires minimulti)."""
    from unfolding.wannier_unfold import run

    return run(
        path=hr_dir, prefix=prefix,
        labels=LABELS, scmat=SCMAT,
        output_figure=out_png,
        kvectors=KVECTORS, knames=KNAMES, npoints=npoints,
    )


# ---------------------------------------------------------------------------
# Real bundled Wannier90 datasets (the page's headline figures)
# ---------------------------------------------------------------------------
REAL_SCMAT = [[1, -1, 0], [1, 1, 0], [0, 0, 2]]   # sqrt(2)xsqrt(2)x2 of cubic a=3.9 A
REAL_LABELS = ["pz", "px", "py"] * 12 + ["dz2", "dxy", "dyz", "dx2", "dxz"] * 4
REAL_DIRS = {"pristine": ("data", "pristine", "sto_unfolded.png"),
             "ti_vacancy": ("data", "ti_vacancy", "sto_ti_vacancy.png")}


def run_real(which, out_png, npoints):
    """Unfold a bundled real Wannier90 dataset with the built-in reader."""
    from unfolding.wannier_unfold import Wannier90Model, WannierUnfolder

    parent, dirname, default_out = REAL_DIRS[which]
    model = Wannier90Model(os.path.join(parent, dirname), "wannier90",
                           scmat=REAL_SCMAT)
    u = WannierUnfolder(model, labels=REAL_LABELS, sc_matrix=REAL_SCMAT)
    ax = u.plot_unfolded_band(kvectors=KVECTORS, knames=KNAMES, npoints=npoints)
    ax.figure.savefig(out_png, dpi=200, bbox_inches="tight")
    return ax


def sector_weights(u, kpts_sc, scmat):
    """Sector-resolved unfolding weights for the built-in engine.

    At each path point k_sc the supercell modes fold onto |det scmat|
    primitive momenta q_m = frac(k_sc @ scmat^-1 + m @ scmat^-1).  Each
    mode is projected onto the primitive Bloch sums

        |mu, q_m> = (1/sqrt(N)) sum_n exp(-2 pi i q_m.x_n) |n, mu>,

    built on the N = |det scmat| internal primitive translations (x_n:
    site positions in primitive-fractional units); A_m collects the
    projections for sector q_m and its nonzero eigenvalues within each
    (near-)degenerate energy group are the gauge-invariant branch weights
    for that sector.  A pristine branch scores exactly 1 in its own sector
    (and each branch of an exact crossing scores 1), while an incoherent
    defect state scores fractionally in every sector.

    Returns w[nk, nband].
    """
    scmat = np.asarray(scmat, dtype=float)
    inv = np.linalg.inv(scmat)
    nsc = int(round(abs(np.linalg.det(scmat))))
    ns = int(round(nsc ** (1.0 / 3.0)))
    sites = [(ix / ns, iy / ns, iz / ns) for ix in range(ns)
             for iy in range(ns) for iz in range(ns)]
    nsite = len(sites)
    # internal translation phases need INTEGER primitive steps (L * x_sc);
    # with fractional supercell coordinates the sector characters are wrong
    x_int = np.rint(ns * np.asarray(sites, dtype=float))
    ms = np.array(np.meshgrid(*[range(ns)] * 3, indexing="ij")).reshape(3, -1).T
    q_all = ((np.atleast_2d(kpts_sc) @ inv) % 1.0)[:, None, :] \
        + (ms @ inv)[None, :, :]                                # [nk, nsec, 3]
    evecs = np.transpose(u.evecs, (1, 0, 2))                    # [nk, ndof, nband]
    freqs = u.evals.T                                           # [nk, nband]
    nband = evecs.shape[2]
    norb = 3

    w = np.zeros((len(q_all), nband))
    for iq in range(len(q_all)):
        # group branches by (near-)degenerate energy
        order = np.argsort(freqs[iq])
        grp = np.zeros(nband, dtype=int)
        g = 0
        for i in range(1, nband):
            if freqs[iq][order[i]] - freqs[iq][order[i - 1]] > 1e-6:
                g += 1
            grp[i] = g
        for gg in np.unique(grp):
            sel = order[grp == gg]
            ng = len(sel)
            C = evecs[iq][:, sel].reshape(nsite, norb, ng)      # (site, orb, mode)
            if ng == 1:
                # isolated mode: its maximal sector content
                # W_m = sum_mu |<mu, q_m|c>|^2 is 1 for a pristine branch
                # (pure in one sector) and < 1 for an incoherent defect state
                best = 0.0
                for qm in q_all[iq]:
                    ph = np.exp(-2j * np.pi * (x_int @ qm))     # (nsite,)
                    A = np.einsum("n,nmb->mb", ph, C) / np.sqrt(nsite)
                    best = max(best, float(np.sum(np.abs(A) ** 2)))
                w[iq, sel] = best
            else:
                # degenerate group: eigh may return any unitary mixture of
                # the sector states, so project onto the complete set of
                # sector Bloch sums; the group jointly carries full weight
                # in each present sector -> every branch scores 1
                blocks = []
                for qm in q_all[iq]:
                    ph = np.exp(-2j * np.pi * (x_int @ qm))     # (nsite,)
                    blocks.append(np.einsum("n,nmb->mb", ph, C) / np.sqrt(nsite))
                A = np.concatenate(blocks, axis=0)              # (nsec*norb, ng)
                ev = np.linalg.eigvalsh(A.conj().T @ A)
                w[iq, sel] = np.sort(np.clip(ev, 0.0, 1.0))[::-1]
    return w


def run_builtin(hr_path, out_png, npoints, defect):
    """numpy-only route through the documented WannierUnfolder API."""
    from ase.dft.kpoints import bandpath

    from unfolding.plotphon import plot_band_weight
    from unfolding.wannier_unfold import WannierUnfolder

    model = SimpleTB(hr_path)
    if defect:
        apply_defect(model)
    u = WannierUnfolder(model, labels=LABELS, sc_matrix=SCMAT)

    kvectors_sc = [np.dot(k, SCMAT) for k in KVECTORS]
    path = bandpath(kvectors_sc, SCMAT @ model.atoms.get_cell(), npoints)
    x = np.atleast_1d(path.get_linear_kpoint_axis()[0])
    ticks, j0 = [], 0
    for v in kvectors_sc:
        j = j0 + int(np.argmin(np.linalg.norm(path.kpts[j0:] - v, axis=1)))
        ticks.append(x[j])
        j0 = j

    u.unfold(path.kpts)                                 # fills u.evals/u.evecs
    weights = sector_weights(u, path.kpts, SCMAT)       # [nk, nband]
    ax = plot_band_weight(
        [list(x)] * len(model._orb),
        u.evals,                                        # [nband, nk]
        weights.T * 0.98 + 0.01,
        efermi=None, yrange=None, output=None, style='alpha',
        ylabel='Energy (eV)',
        ypad=float(np.ptp(u.evals) * 0.05 + 1e-3),
        xticks=[KNAMES, ticks])
    ax.figure.savefig(out_png, dpi=300, bbox_inches="tight")
    return ax, weights


def report_weights(weights):
    w = np.asarray(weights).ravel()
    mid = np.mean((w > 0.01) & (w < 0.99))
    print("weights: %.1f%% folded onto one primitive k (w<=0.01 or w>=0.99), "
          "%.1f%% fractional (defect / degeneracy states)"
          % (100 * (1 - mid), 100 * mid))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Unfold a Wannier90 tight-binding model of SrTiO3 "
        "(bundled real Wannier90 datasets or the synthetic t2g model).")
    parser.add_argument("--real", choices=sorted(REAL_DIRS), default=None,
                        help="run a bundled real Wannier90 dataset "
                             "(data/pristine or data/ti_vacancy) through the "
                             "built-in reader instead of the synthetic model")
    parser.add_argument("--hr", default=os.path.join("data", "example_hr.dat"),
                        help="wannier90_hr.dat path (default %(default)s, "
                             "synthetic route only)")
    parser.add_argument("--prefix", default="example",
                        help="file prefix for the minimulti engine (default %(default)s)")
    parser.add_argument("--engine", choices=("auto", "builtin", "minimulti"),
                        default="auto",
                        help="minimulti MyTB route or the built-in numpy reader "
                             "(default: minimulti if importable, else builtin)")
    parser.add_argument("--defect", action="store_true",
                        help="apply an isolated-site impurity to the synthetic "
                             "model (mimics the oxygen-vacancy figure)")
    parser.add_argument("--npoints", type=int, default=200,
                        help="k-points along the path (default %(default)s)")
    parser.add_argument("--output", default=None,
                        help="output figure (default sto_unfolded.png, or "
                             "sto_defect.png with --defect)")
    parser.add_argument("--regenerate-hr", action="store_true",
                        help="rewrite the bundled synthetic data/example_hr.dat")
    args = parser.parse_args(argv)

    if args.real:
        parent, dirname, default_out = REAL_DIRS[args.real]
        out_png = args.output or default_out
        print("engine: built-in Wannier90 reader + WannierUnfolder "
              "(real dataset: %s)" % os.path.join(parent, dirname))
        run_real(args.real, out_png, args.npoints)
        print("wrote %s" % out_png)
        return 0

    if args.regenerate_hr or not os.path.exists(args.hr):
        blocks, norb = build_t2g_hr()
        write_hr_dat(args.hr, blocks, norb)

    out_png = args.output or ("sto_defect.png" if args.defect else "sto_unfolded.png")

    engine = args.engine
    if engine == "auto":
        try:
            import minimulti  # noqa: F401
            engine = "minimulti"
        except ImportError:
            engine = "builtin"

    if engine == "minimulti" and not args.defect:
        print("engine: minimulti MyTB (the page's documented route)")
        try:
            run_minimulti(os.path.dirname(args.hr) or ".", args.prefix,
                          out_png, args.npoints)
            print("wrote %s" % out_png)
            return 0
        except Exception as exc:  # bundled file is hr-only: fall back
            print("minimulti route failed (%s); using the built-in reader" % exc)

    print("engine: built-in numpy reader + WannierUnfolder")
    _ax, weights = run_builtin(args.hr, out_png, args.npoints, args.defect)
    report_weights(weights)
    print("wrote %s" % out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
