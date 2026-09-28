====================================================================
ABINIT DDB phonons -- reproduction bundle (abinit-ddb)
====================================================================

Companion bundle of the "ABINIT DDB phonons" example
(https://mailhexu.github.io/unfolding/examples/abinit-ddb/). Unfolds
phonons from an ABINIT DDB (derivatives database) onto a primitive
cell: the route runs ABINIT's `anaddb` through abipy to obtain the
supercell eigenvectors along your path, then computes the unfolding
weights (gauge-robust Bloch-sum projectors with degenerate-group
resolution, so eigenvector storage gauges do not change the weights).
Two systems ship in data/:

  data/out_DDB   fcc Cu in the conventional cubic cell (natom 4);
                 unfolded onto the fcc primitive cell along
                 Gamma-X-W-Gamma-L
  data/out.DDB   CaTiO3, Pnma ground-state cell (20 atoms,
                 a~b~sqrt(2)*a_pc, c~2*a_pc, four formula units of the
                 5-atom pseudo-cubic perovskite); unfolded onto the
                 pseudo-cubic cell along Gamma-X-M-Gamma-R with
                 dipdip=0

BUNDLE LAYOUT
  unfold.toml            config for the `unfolding` command (Cu)
  unfold-catio3.toml     config for the CaTiO3 case
  reproduce.py           renders both figures (--system cu|catio3)
  data/out_DDB, data/out.DDB   the bundled DDBs behind the published
                         Cu and CaTiO3 figures
  inputs/                the ABINIT decks that produced them
                         (cu_phonon_scf.in + cu_phonon_rf.in,
                          catio3_phonon_scf.in + catio3_phonon_rf.in)
  pseudos/               Ca.psp8, O.psp8, Ti-sp.psp8

PREREQUISITES
  - Python 3.11+ with numpy, matplotlib
  - pip install "unfolding[abipy]" (abipy)
  - a working `anaddb` on your PATH (or configured in abipy's
    manager.yml). Without anaddb, reproduce.py still parses the DDB and
    writes a summary; only the unfolding step needs it.

STRUCTURE AND PATH (the part that bites)
  Cu: supercell matrix sc_mat = [[-1,1,1],[1,-1,1],[1,1,-1]] with the
  DDB cell = sc_mat @ primitive cell. Mind the k-path frame: ase's
  get_special_points returns fcc points in PRIMITIVE-frame fractional
  coordinates, while abipy reads path vertices in the fractional frame
  of the cell stored in the DDB (for a conventional-cubic-cell DDB:
  X=(0,1,0), W=(1/2,1,0), L=(1/2,1/2,1/2)). The configs therefore pass
  explicit vertices in the DDB frame; anaddb interpolates between
  consecutive vertices (no point density needed). CaTiO3: sc_mat rows
  are the Pnma axes in pseudo-cubic units, (1,-1,0), (1,1,0), (0,0,2);
  vertices are the pseudo-cubic Gamma-X-M-Gamma-R points in the DDB
  frame.

HOW TO RUN
  From the unpacked bundle directory, either the CLI:

    unfolding --config unfold.toml            # Cu   -> cu_fcc_unfolded_cli.png
    unfolding --config unfold-catio3.toml     #      -> catio3_unfolded_cli.png
    # equivalent explicit flags (Cu):
    unfolding abinit-ddb --ddb data/out_DDB \
        --sc-mat -1 1 1 1 -1 1 1 1 -1 \
        --kpoints 0 0 0 0 1 0 0.5 1 0 0 0 0 0.5 0.5 0.5 \
        --names G X W G L --output cu_fcc_unfolded_cli.png

  or the reproduction script:

    python reproduce.py                       # Cu   -> cu_unfolded.png
    python reproduce.py --system catio3       #      -> catio3_unfolded.png

EXPECTED OUTPUT
  Spectral-weight maps on the phonon branches: Cu (three bold acoustic
  branches on the primitive path; LO-TO splitting handled by dipdip=1)
  and CaTiO3 pseudo-cubic branches without the dipole-dipole term
  (dipdip=0, as published). Runtime: seconds per system once anaddb is
  available.

PRODUCING YOUR OWN DDB
  reproduce.py prints a how-to (also in its docstring): a ground-state
  SCF run, then a DFPT run with optdriver 1, rfphon 1, rfatpol 1 natom,
  rfdir 1 1 1, one q-point per run (qpt 0 0 0 ...), prtddb 1; merge
  partial DDBs with mrgddb. The bundled inputs/ decks are worked
  examples for both systems.
