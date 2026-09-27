"""Generate the GPAW Si fixtures (LCAO + planewave) for unfolding story 017.

Runs serial PBE calculations with gpaw (>= 26, new calculators) and
writes ``mode='all'`` .gpw restarts into both consuming repos:

- HamiltonIO/tests/data/gpaw/      reader tests (si_prim_lcao, si_sc_lcao, si8_pw)
- unfolding/tests/data/gpaw_example/  adapter tests + example figures (all files)

Fixtures
--------
si_prim_lcao.gpw   2-atom fcc primitive cell (a=5.43 A), LCAO (default szp
                   basis, 4 AOs/atom), PBE, Gamma-centered 16x16x16 grid,
                   symmetry off. Band-structure oracle for the examples.
                   The grid has two jobs: (1) accurate off-grid
                   interpolation — a uniform grid cannot disentangle the
                   +N/2 and -N/2 real-space shells, and H[T] at |T| = 2
                   primitive translations still carries ~1e-1 eV, so the
                   interpolated bands drift by ~eV between grid points
                   of a 4x4x4 file (at |T| = 8 the tables are ~1e-14 eV);
                   (2) a k-sampling converged enough that the
                   primitive-cell reference matches the supercell
                   eigenvalues to ~1 meV.
si_sc_lcao.gpw     8-atom conventional cell, same settings, Gamma-centered
                   8x8x8 grid. The Gamma-centered grid (not GPAW's default
                   half-shifted one) is required for the real-space
                   transform; the 8x8x8 SCF sampling keeps the supercell
                   density as well converged as the primitive-cell
                   reference (eigenvalue references agree to ~1 meV),
                   and the table range (in supercell translations the
                   Nyquist shell = 4 supercell steps = 8 primitive
                   steps) is resolved so the folded-band interpolation
                   is exact off the grid.
si7p_lcao.gpw      One Si replaced by P in the 8-atom cell, 8x8x8.
                   Substitutional dopant example (P carries 4 AOs/atom
                   like Si, so RelabelMap works with match_species=False).
si8_pw.gpw         8-atom conventional cell, planewave mode (340 eV
                   cutoff), SCF on a Gamma-centered 2x2x2 grid, then a
                   fixed-density non-SCF run storing 24 bands on the
                   300-point primitive path (Gamma-X-W-Gamma-L-W-X mapped
                   to supercell coordinates). Planewave unfolding input.
si_prim_pw_path_bands.npz
                   Primitive-cell planewave bands along the same path
                   (kpoints, eigenvalues, efermi) - the crimson overlay
                   of the planewave example. Eigenvalues only, so this
                   stays a few kilobytes.

Usage (mydev env):  python generate_fixtures.py [--workdir DIR]
"""
import argparse
import os
import shutil
import sys

import numpy as np

A = 5.43  # lattice constant, Angstrom
PRIM_CELL = A * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
CONV_CELL = A * np.eye(3)
SC_SCALED = [
    (0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
    (0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75),
    (0.25, 0.75, 0.75),
]
# fcc special points, primitive reciprocal fractional (Setyawan-Curtarolo)
SPECIAL = {
    "G": (0.0, 0.0, 0.0), "X": (0.5, 0.0, 0.5), "W": (0.5, 0.25, 0.75),
    "L": (0.5, 0.5, 0.5),
}
PATH = "GXWGLWX"
NPTS = 300
# supercell (conventional cubic) matrix in primitive units, row convention
B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])


def primitive_atoms():
    from ase import Atoms

    return Atoms(
        "Si2",
        scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
        cell=PRIM_CELL,
        pbc=True,
    )


def sc_atoms(symbols="Si8"):
    from ase import Atoms
    from ase.symbols import string2symbols

    if len(string2symbols(symbols)) != 8:
        raise ValueError(f"{symbols} does not name 8 atoms")
    return Atoms(symbols, scaled_positions=SC_SCALED, cell=CONV_CELL, pbc=True)


def path_kpoints():
    """Primitive-path k-points (NPTS) and their supercell coordinates."""
    bcart = 2.0 * np.pi * np.linalg.inv(PRIM_CELL).T
    segs = list(zip(PATH, PATH[1:]))
    lengths = [
        np.linalg.norm((np.array(SPECIAL[b]) - np.array(SPECIAL[a])) @ bcart)
        for a, b in segs
    ]
    total = sum(lengths)
    kpts, seg_starts = [], []
    for i, (a, b) in enumerate(segs):
        pa, pb = np.array(SPECIAL[a]), np.array(SPECIAL[b])
        n = max(2, round(NPTS * lengths[i] / total) + 1)
        t = np.linspace(0.0, 1.0, n, endpoint=i == len(segs) - 1)
        seg_starts.append(len(kpts))
        kpts.extend(pa + (pb - pa) * t[:, None])
    kpts = np.array(kpts)
    return kpts, np.array(seg_starts), kpts @ B.T


def run_lcao(atoms, kpts, txt):
    from gpaw import GPAW

    atoms.calc = GPAW(
        mode="lcao",
        h=0.17,
        xc="PBE",
        kpts={"size": kpts, "gamma": True},
        symmetry="off",
        txt=txt,
    )
    energy = atoms.get_potential_energy()
    return atoms.calc, energy


def run_pw_scf(atoms, kpts, txt):
    from gpaw import GPAW, PW

    atoms.calc = GPAW(
        mode=PW(340.0),
        xc="PBE",
        kpts={"size": kpts, "gamma": True},
        nbands=24,
        symmetry="off",
        txt=txt,
    )
    return atoms.calc, atoms.get_potential_energy()


def run_pw_path(calc, atoms, path_kc, nbands, txt):
    """SCF directly on the path k-set (eigenvalues converge with density).

    gpaw >= 25 new calculators dropped the old ``fixdensity`` non-SCF
    flow; a converged SCF on the fixed path sampling yields exactly the
    path band structure we unfold.
    """
    from gpaw import GPAW, PW

    atoms.calc = GPAW(
        mode=PW(340.0),
        xc="PBE",
        # a plain list of tuples is stored verbatim; a numpy array gets
        # silently resampled by the new GPAW path machinery
        kpts=[tuple(v) for v in path_kc],
        nbands=nbands,
        symmetry="off",
        txt=txt,
    )
    return atoms.get_potential_energy()


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    unfolding_data = os.path.join(repo_root, "tests", "data", "gpaw_example")
    ham_root = os.environ.get(
        "HAMILTONIO_ROOT",
        os.path.join(
            os.path.dirname(os.path.dirname(repo_root)), "siesta_dev", "HamiltonIO"
        ),
    )
    ham_data = os.path.join(ham_root, "tests", "data", "gpaw")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=".gpaw_fixtures_tmp")
    parser.add_argument(
        "--hamiltonio-data", default=ham_data,
        help="HamiltonIO tests/data/gpaw target (empty string skips copy)")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(unfolding_data, exist_ok=True)

    kprim, seg_starts, ksc = path_kpoints()

    # --- LCAO runs ---------------------------------------------------------
    # Grid choices (root cause of story-017 band/overlay defects): the
    # Fourier interpolation H(k) = sum_T e^{2i pi k.T} H[T] of a grid
    # with N points per axis loses the Nyquist shell |T| = N/2 (the grid
    # cannot distinguish +N/2 from -N/2), so the primitive reference
    # needs N large enough that H[+-N/2] is negligible.  For the 2-atom
    # cell, H[T] at |T| = 4 primitive translations is ~1e-11 eV while
    # |T| = 2 still carries ~1e-1 eV: the primitive cell must use
    # (8, 8, 8).  The supercell runs use (4, 4, 4): their table range is
    # measured in supercell translations (2 supercell steps = 4
    # primitive steps, tail ~1e-6), and the folded 4^3 supercell mesh
    # coincides exactly with the 8^3 primitive mesh, so the SCF density
    # - and hence the eigenvalue reference - is identical for the
    # supercell and the primitive-cell reference bands.
    lcao_jobs = [
        ("si_prim_lcao", "primitive 2-atom LCAO 16x16x16",
         primitive_atoms(), (16, 16, 16)),
        ("si_sc_lcao", "8-atom conventional LCAO 8x8x8",
         sc_atoms(), (8, 8, 8)),
        ("si7p_lcao", "Si7P LCAO 8x8x8", sc_atoms("Si7P"), (8, 8, 8)),
    ]
    for name, label, atoms, kpts in lcao_jobs:
        out = os.path.join(args.workdir, f"{name}.gpw")
        if os.path.exists(out):
            print(f"[skip] {name}")
            continue
        print(f"[run ] {name}: {label}", flush=True)
        calc, energy = run_lcao(atoms, kpts, os.path.join(args.workdir, f"{name}.txt"))
        calc.write(out, mode="all")
        print(f"[done] {name}: E={energy:.6f} eV, efermi={calc.get_fermi_level():.6f}",
              flush=True)

    # --- planewave SCF + path run ------------------------------------------
    pw_out = os.path.join(args.workdir, "si8_pw.gpw")
    if not os.path.exists(pw_out):
        print(f"[run ] si8_pw: 8-atom PW 340 eV on the path ({len(ksc)} kpts)",
              flush=True)
        sc = sc_atoms()
        energy = run_pw_path(None, sc, ksc, 24,
                             os.path.join(args.workdir, "si8_pw.txt"))
        calc = sc.calc
        calc.write(pw_out, mode="all")
        print(f"[done] si8_pw path: E={energy:.6f} eV, "
              f"efermi={calc.get_fermi_level():.6f}", flush=True)

        # primitive-cell PW bands along the same path (overlay, eigenvalues only)
        print("[run ] si_prim_pw: path bands", flush=True)
        prim = primitive_atoms()
        penergy = run_pw_path(None, prim, kprim, 16,
                              os.path.join(args.workdir, "si_prim_pw.txt"))
        pcalc = prim.calc
        bands = np.array([pcalc.get_eigenvalues(kpt=i) for i in range(len(kprim))])
        np.savez(
            os.path.join(args.workdir, "si_prim_pw_path_bands.npz"),
            kpoints=kprim,
            eigenvalues=bands,
            efermi=float(pcalc.get_fermi_level()),
            seg_starts=seg_starts,
        )
        print(f"[done] si_prim_pw: E={penergy:.6f} eV", flush=True)

    # A Gamma-only Si:P PW run suffices for the four primitive folds of
    # supercell Gamma; it does not pretend to sample a continuous path.
    doped_pw = os.path.join(args.workdir, "si7p_pw.gpw")
    if not os.path.exists(doped_pw):
        print("[run ] si7p_pw: 8-atom PW 340 eV at Gamma", flush=True)
        doped = sc_atoms("Si7P")
        _, energy = run_pw_scf(doped, (1, 1, 1),
                               os.path.join(args.workdir, "si7p_pw.txt"))
        doped.calc.write(doped_pw, mode="all")
        print(f"[done] si7p_pw: E={energy:.6f} eV", flush=True)

    targets = [unfolding_data]
    if args.hamiltonio_data:
        os.makedirs(args.hamiltonio_data, exist_ok=True)
        targets.append(args.hamiltonio_data)
    names = ["si_prim_lcao.gpw", "si_sc_lcao.gpw", "si7p_lcao.gpw",
             "si8_pw.gpw", "si7p_pw.gpw", "si_prim_pw_path_bands.npz"]
    for name in names:
        src = os.path.join(args.workdir, name)
        for target in targets:
            # HamiltonIO tests do not need the dopant fixture or the overlay
            if target == unfolding_data or name in (
                "si_prim_lcao.gpw", "si_sc_lcao.gpw", "si8_pw.gpw"
            ):
                shutil.copy2(src, os.path.join(target, name))
    print(f"fixtures written to: {targets}")


if __name__ == "__main__":
    sys.exit(main())
