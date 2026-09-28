====================================================================
ABINIT WFK AFM NiO -- reproduction bundle (abinit-nio)
====================================================================

Companion bundle of the "ABINIT WFK AFM NiO" example
(https://mailhexu.github.io/unfolding/examples/abinit-nio/). Unfolds
type-II antiferromagnetic NiO from the 4-atom magnetic primitive cell
onto the 2-atom rocksalt primitive cell along Gamma-X-W-Gamma-L-W-X,
spin-up channel.

What the example shows
----------------------
The magnetic cell doubles the rocksalt primitive cell along [111] and
carries two Ni moments (+m and -m along z) plus two O:
A_afm = M @ A_prim with M = [[1,0,1],[0,1,1],[1,1,0]] (acell 3*7.8817
bohr = 4.171 Ang, fcc rprim). Because the AFM keeps inversion (time
reversal x sublattice translation maps the channels onto each other),
the spin-up and spin-down unfolded band structures coincide
band-for-band; a single spin-up panel is plotted (spin=1 computes the
identical partner). The unfolding weight of a state is computed within
its own spin channel; no spin mixing is involved.

BUNDLE LAYOUT
  unfold.toml  config for the `unfolding` command -- runnable once the
               dense WFK has been generated into data/ (see below)
  inputs/      nio_afm.abi           dense 305-point path deck (ecut 40,
                                     nsppol 2, dataset-1 AFM SCF on a
                                     4x4x4 mesh, dataset-2 frozen-density
                                     non-SCF path run, kptopt 0, istwfk 1)
               nio_afm_corners.abi   Gamma/X/W/L corners only (small WFK,
                                     enough for a coarse figure)
  pseudos/     Ni.psp8, O.psp8 (18-valence-electron Ni; the Ni-3s
               semicore multiplet near -60 eV is present in the WFK but
               outside the plotted -16...8 eV window)
  data/        WFK output directory; no WFK ships with this bundle
               (the dense nsppol-2 path WFK is ~1.1 GB) -- run the
               ABINIT commands printed by reproduce.py first
  reproduce.py renders nio_afm_unfolded.png; run from this directory

PREREQUISITES
  - Python >= 3.9 with numpy and matplotlib
  - the unfolding package: pip install unfolding
  - only for regenerating the WFK: ABINIT >= 9 built with netCDF
    support (iomode 3)

STRUCTURE AND PATH
  - M = [[1,0,1],[0,1,1],[1,1,0]], magnetic cell = M @ primitive
  - path Gamma-X-W-Gamma-L-W-X, exactly 305 primitive reciprocal-
    fractional points. data/path_kpoints.txt, path_xcoords.txt and
    path_xticks.txt contain the exact grid/axis used in examples; the
    AFM WFK stores K = k_prim @ M.T.
  - ABINIT settings the WFK reader requires: iomode 3, istwfk 1,
    prtwf 1, chkprim 0; nsppol 2 / nspden 2 with spinat 0 0 2.0 /
    0 0 -2.0 on the two Ni

HOW TO RUN
  1. Regenerate the WFK from the unpacked bundle directory (reproduce.py
     prints this exact command when data/ is empty):

       mkdir -p run_nio && cd run_nio
       cp ../inputs/nio_afm.abi ../pseudos/Ni.psp8 ../pseudos/O.psp8 .
       abinit nio_afm.abi > nio_afm.abo 2>&1
       cp nio_afmo_DS2_WFK.nc ../data/ && cd ..

     Expect on the order of hours and a ~1.1 GB WFK. (nio_afm_corners.abi
     gives a small WFK in minutes; render a coarse figure from it by
     pointing reproduce.py at the corner file, or regenerate the dense
     deck for the published map.)

  2. Render the figure, either the CLI:

       unfolding --config unfold.toml      # writes nio_afm_unfolded_cli.png

     or the reproduction script:

       python reproduce.py                 # writes nio_afm_unfolded.png

EXPECTED OUTPUT
  Weight-coded unfolded bands (blue intensity = spectral weight),
  energies relative to the WFK Fermi level (eV).
