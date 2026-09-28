====================================================================
GPAW LCAO Si unfolded bands -- reproduction bundle (gpaw-si)
====================================================================
Docs example: docs/content/examples/gpaw-si.md in the unfolding
repository. Regenerates gpaw_si_unfolded.png (pristine 8-atom
conventional-cubic Si supercell LCAO restart unfolded onto the 2-atom
fcc primitive cell along Gamma-X-W-Gamma-L-W-X, primitive-cell bands
overlaid in crimson) and gpaw_si_p_doped.png (one Si substituted by P,
dopant mapped onto the host site). Every weight-1 unfolded branch lies
on a primitive band to within a few meV.

The .gpw restarts are NOT shipped: the three files total ~80 MB, over
the 10 MB bundle cap. generate_fixtures.py reproduces them exactly with
your own GPAW (~2-3 h serial), after which reproduce.py runs in
seconds.

BUNDLE LAYOUT
  reproduce.py            self-contained script; regenerates both figures
  generate_fixtures.py    GPAW driver producing data/*.gpw
  unfold.toml             config for the `unfolding` CLI (same
                          computation): unfolding --config unfold.toml
  data/                   the .gpw restarts land here (empty as shipped)

PREREQUISITES
  - Python 3.11 with numpy, scipy, matplotlib, GPAW >= 25 (tested with
    gpaw 26) and HamiltonIO, plus the unfolding package importable (pip
    install -e <unfolding repo checkout>, or PYTHONPATH=<unfolding repo
    checkout>).

STRUCTURES
  Primitive cell: 2-atom fcc, a = 5.43 A,
      cell = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]].
  Supercell: 8-atom conventional cubic cell, supercell = M @ primitive
      with M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]; the Si7P fixture
      substitutes P on the (1/4,1/4,1/4) site.
  Basis: GPAW's default szp LCAO set -- 4 atomic orbitals per atom for
  Si AND P (same valence row). Run settings: PBE, h=0.17 grid spacing,
  symmetry='off', mode='all' restarts. GPAW's LCAO matrices already
  carry every PAW contribution (projector-augmented overlap, dH terms
  in the Hamiltonian), so no PAW correction is applied on top.

K-GRIDS (why these sizes)
  Gamma-centered grids: primitive cell 16x16x16, supercells 8x8x8. The
  real-space tables are the inverse lattice Fourier transform of the
  k-grid data; a uniform grid cannot disentangle its +N/2 and -N/2
  shells (half-shifted even grids are not even Hermitian after the
  transform), so the primitive reference needs 16^3 (keeps the
  interpolation error at ~1e-11 eV; 4^3 leaves ~eV wobble). The
  supercell 8^3 mesh folds exactly onto the 16^3 primitive mesh, so
  SCF densities -- and the eigenvalue reference -- match.

K-PATH
  Gamma-X-W-Gamma-L-W-X, 300 points, fcc special points in primitive
  reciprocal fractional coordinates (Setyawan-Curtarolo):
      G (0,0,0), X (1/2,0,1/2), W (1/2,1/4,3/4), L (1/2,1/2,1/2)
  Segment point counts are proportional to Cartesian length. Energies in
  eV with 0 at the unfolded run's own Fermi level (calc.get_fermi_level
  of each .gpw); the primitive overlay is drawn on the same zero.

PARAMETERS THAT MATTER
  supercell matrix M   row convention supercell = M @ primitive.
  orb_counts           4 orbitals per atom (szp), used by RelabelMap.
  match_species        True for pristine; False for Si7P so the P dopant
                       folds onto the host Si site it replaces.
  method               "ideal" generic-k spectral weight ("ring" is the
                       exact torus projection at commensurate momenta).

COMMANDS
  python generate_fixtures.py         # ~2-3 h serial, GPAW
  python reproduce.py                 # writes both PNGs, seconds
  unfolding --config unfold.toml      # CLI route (pristine figure)

EXPECTED OUTPUT
  gpaw_si_unfolded.png and gpaw_si_p_doped.png (150 dpi): weight-coded
  unfolded bands in eV with tick labels Gamma X W Gamma L W X and
  primitive-cell bands overlaid in crimson. Console: pristine max
  |w(1-w)| and the max unfolded-vs-primitive deviation on weight-1
  branches (a few meV with the committed grids).
