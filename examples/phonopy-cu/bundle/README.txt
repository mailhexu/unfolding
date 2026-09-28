====================================================================
Phonopy Cu phonons -- reproduction bundle (phonopy-cu)
====================================================================

Regenerates the figure of the "Phonopy Cu phonons" example
(https://mailhexu.github.io/unfolding/examples/phonopy-cu/): the phonon
band structure of a 3x3x3 fcc Cu supercell (FORCE_CONSTANTS + SPOSCAR
from a phonopy run, both shipped) unfolded onto the primitive fcc
Brillouin zone along Gamma-X-W-Gamma-L.

What the example shows
----------------------
For a pristine crystal every supercell phonon folds from exactly one
primitive momentum, so the unfolding weights are binary: three bold
branches trace the primitive Cu dispersion while the other 24 folded
copies of the 27-atom supercell stay invisible. In a defective or
distorted supercell the same plot would show fractional weights.

BUNDLE LAYOUT
  unfold.toml       config for the `unfolding` command (see CLI below)
  reproduce.py      runs the same unfolding and saves the figure
  FORCE_CONSTANTS   second-order force constants (27x27, from the
                    committed phonopy run on the 3x3x3 supercell)
  SPOSCAR           the 27-atom 3x3x3 supercell of primitive fcc Cu
                    (a = 3.61 Ang)

PREREQUISITES
  - Python 3.11+ with numpy, matplotlib, ase, spglib, phonopy
  - the unfolding package: pip install "unfolding[phonopy]"

STRUCTURE AND PATH
  - supercell matrix M = diag(3,3,3), row convention A_sc = M @ A_prim;
    the path cell is derived as inv(M) @ SPOSCAR cell (the primitive fcc
    cell). The phonopy reading matrix sc_mat is the identity: the
    SPOSCAR itself is the unfolded cell.
  - path Gamma-X-W-Gamma-L, 300 q-points in primitive reciprocal
    fractional coordinates (Setyawan-Curtarolo special points via ase)
  - frequencies are read from phonopy in THz and plotted in cm^-1

HOW TO RUN
  From the unpacked bundle directory, either the CLI:

    unfolding --config unfold.toml       # writes unfolded_band_structure.png
    # equivalent explicit flags:
    unfolding phonopy --force-constants FORCE_CONSTANTS --sposcar SPOSCAR \
        --unfold-mat 3 0 0 0 3 0 0 0 3 --special-points GXWGL \
        --npts 300 --output unfolded_band_structure.png

  or the Python reproduction script:

    python reproduce.py                  # same figure, same defaults
    python reproduce.py --output any.png --npoints 400

  or the documented Python API from the bundle root:

    from ase.build import bulk
    from ase.dft.kpoints import bandpath, get_special_points
    from unfolding.phonopy_unfolder import phonopy_unfold

    atoms = bulk("Cu", "fcc", a=3.61)
    points = get_special_points("fcc", atoms.cell, eps=0.01)
    kpts, x, X = bandpath([points[k] for k in "GXWGL"], atoms.cell, 300)
    ax = phonopy_unfold(
        sc_mat=np.diag([1, 1, 1]), unfold_sc_mat=np.diag([3, 3, 3]),
        force_constants="FORCE_CONSTANTS", sposcar="SPOSCAR",
        qpts=kpts, xqpts=x, Xqpts=X,
        qnames=[r"$\Gamma$", "X", "W", r"$\Gamma$", "L"],
    )
    ax.figure.savefig("unfolded_band_structure.png", dpi=300)

EXPECTED OUTPUT
  unfolded_band_structure.png: weight-coded unfolded phonon branches
  (blue intensity = unfolding weight) along Gamma-X-W-Gamma-L,
  frequencies in cm^-1. Runtime: a few seconds on a laptop; no DFT is
  involved (the force constants ship with the bundle).

  To unfold your own system, point --force-constants/--sposcar (or the
  TOML [input] section) at your phonopy run's FORCE_CONSTANTS/SPOSCAR
  and set the supercell matrix you used.
