====================================================================
OpenMX Si unfolded bands -- reproduction bundle (openmx-si)
====================================================================
Docs example: docs/content/examples/openmx-si.md in the unfolding
repository. Regenerates the figure openmx_si_unfolded.png: the committed
8-atom conventional-cubic Si supercell (binary .scfout from an OpenMX
run with HS.fileout on) unfolded onto the 2-atom fcc primitive cell
along Gamma-X-W-Gamma-L-W-X, with the independently diagonalized OpenMX
primitive-cell bands overlaid in red. Every weight-1 unfolded branch
lies on a primitive band, which is the practical validation of the
OpenMX real-space-Hamiltonian conventions (image translations atv_ijk,
Hartree->eV and Bohr->Angstrom conversion, ChemP referencing).
Optionally (with --doped and a user-supplied Si7P scfout) regenerates
openmx_si_p_doped.png.

BUNDLE LAYOUT
  reproduce.py   self-contained script; regenerates the figure
  unfold.toml    config for the `unfolding` CLI (same computation):
                   unfolding --config unfold.toml
  data/          committed binary fixtures (the OpenMX outputs)
                   openmx_si_prim.scfout  primitive cell (2 atoms, 4x4x4 k-grid)
                   openmx_si_sc.scfout    supercell   (8 atoms, 2x2x2 k-grid)
  inputs/        the OpenMX inputs (.dat) of those runs, plus the Si7P
                 input for regenerating the doped fixture yourself

PREREQUISITES
  - Python 3.11 with numpy, scipy, matplotlib, ase and HamiltonIO
    (the OpenMX .scfout parser), and the unfolding package importable
    (pip install -e <unfolding repo checkout>, or PYTHONPATH=<unfolding
    repo checkout>). Tested with unfolding 0.1.0 and HamiltonIO 0.3.x.
  - No OpenMX installation is needed to reproduce the pristine figure:
    the committed .scfout already carries the real-space Hamiltonian and
    overlap. OpenMX is only needed to produce NEW scfouts (e.g. the
    Si7P run, see DOPED RUN below).

STRUCTURES
  Primitive cell: 2-atom fcc, a = 5.43 A
      cell = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]]
      Si at (0,0,0) and (1/4,1/4,1/4) (fractional)
  Supercell: 8-atom conventional cubic cell, a = 5.43 A, related by
      M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]   (supercell = M @ primitive)
  Basis: Si7.0-s2p2d1 (13 orbitals per Si atom), Si_PBE19
  pseudopotentials (OpenMX 2019 data set), GGA-PBE, spin unpolarized.

K-PATH
  Gamma-X-W-Gamma-L-W-X, 300 points, fcc special points in primitive
  reciprocal fractional coordinates (Setyawan-Curtarolo):
      G (0,0,0), X (1/2,0,1/2), W (1/2,1/4,3/4), L (1/2,1/2,1/2)
  Segment point counts are proportional to Cartesian length. Energies in
  eV with 0 at the Fermi level of the unfolded run (OpenMX ChemP parsed
  from the scfout); plot window -13..8 eV.

PARAMETERS THAT MATTER
  unfold_sc_mat   M above, row convention supercell = M @ primitive.
  method          "ideal" (default) generic-k weight; "ring" selects the
                  exact torus projection at commensurate momenta.
  match_species   True maps supercell atoms onto same-species primitive
                  sites; the Si7P run needs False so the P dopant folds
                  onto the host Si site it replaces.
  kpts/knames/    the path and its tick labels; the adapter plots
  xqpts/Xqpts     weight-coded bands (alpha encodes weight).

COMMANDS
  python reproduce.py                 # writes openmx_si_unfolded.png
  python reproduce.py out.png         # or any output path
  unfolding --config unfold.toml      # CLI route (same figure)

EXPECTED OUTPUT
  openmx_si_unfolded.png (150 dpi): weight-coded unfolded bands in eV
  with tick labels Gamma X W Gamma L W X and the primitive-cell bands
  overlaid in crimson (labelled in the legend). Runtime: a few seconds
  on one core.

DOPED RUN (optional)
  python reproduce.py --doped
  needs data/openmx_si_sc_p.scfout, which is NOT shipped (the three
  scfouts together exceed the 10 MB bundle cap). Produce it with OpenMX:
  copy inputs/openmx_si_sc_p.dat next to the OpenMX 2019 pseudos
  (Si_PBE19.vps, P_PBE19.vps; set DATA.PATH in the .dat to your
  DFT_DATA19 location), run `openmx openmx_si_sc_p.dat`, and copy the
  resulting openmx_si_sc_p.scfout into data/. The P atom substitutes Si
  site 5 (the (1/4,1/4,1/4) atom) with the same P7.0-s2p2d1 basis
  (13 orbitals); reproduce.py then maps it with match_species=False.

HOW THE FIXTURES WERE PRODUCED
  OpenMX 3.9 with the 2019 data files, GGA-PBE, scf.EigenvalueSolver
  band, HS.fileout on:
    si_prim:  2-atom primitive cell, scf.Kgrid 4 4 4.
    si_sc:    8-atom conventional cell (M @ primitive), scf.Kgrid 2 2 2.
  A k-sampled SCF is essential: a Gamma-only run collapses all supercell
  image shells into the R=0 block, which cannot be unfolded at generic
  momenta. The full inputs ship in inputs/.

LICENSING / REDISTRIBUTION
  The bundle contains no OpenMX pseudopotential data: the .dat inputs
  name Si_PBE19/P_PBE19.vps, which you get from the OpenMX distribution
  (2019 data set). They are NOT needed to reproduce the pristine figure
  (the committed scfouts are our own calculation outputs and ship here).
