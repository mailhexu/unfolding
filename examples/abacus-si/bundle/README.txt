====================================================================
ABACUS LCAO Si unfolded bands -- reproduction bundle (abacus-si)
====================================================================
Docs example: docs/content/examples/abacus-si.md in the unfolding
repository. Regenerates abacus_si_unfolded.png (pristine 8-atom
conventional-cubic Si supercell unfolded onto the 2-atom fcc primitive
cell along Gamma-X-W-Gamma-L-W-X) and abacus_si_p_doped.png (one Si
substituted by P, dopant mapped onto the host site), each overlaid with
the independently computed primitive-cell bands in crimson. Every
weight-1 unfolded branch lies on a primitive band.

BUNDLE LAYOUT
  reproduce.py   self-contained script; regenerates both figures
  unfold.toml    config for the `unfolding` CLI (same computation):
                   unfolding --config unfold.toml
  data/          committed ABACUS run directories (the OUT.* outputs)
                   si_prim/  primitive cell (2 atoms)  + OUT.si_prim
                   si_conv/  supercell   (8 atoms)  + OUT.si_conv
                   si7p/     Si7P supercell          + OUT.si7p
                 Each run dir carries INPUT/STRU/KPT and OUT.<name> with
                 the sparse real-space tables (out_mat_hs2) and the
                 running_scf.log that supplies E_F.
  refs/          pseudopotentials and numerical orbitals named by the
                 STRU files (Si.upf, P.upf, *_gga_10au_100Ry_2s2p1d.orb)

PREREQUISITES
  - Python 3.11 with numpy, scipy, matplotlib and HamiltonIO (the
    ABACUS parser), and the unfolding package importable (pip install
    -e <unfolding repo checkout>, or PYTHONPATH=<unfolding repo
    checkout>). Tested with unfolding 0.1.0 and HamiltonIO 0.3.x.
  - No ABACUS installation is needed: the committed data-HR-sparse /
    data-SR-sparse CSR tables plus the orbital files are everything the
    parser consumes.

STRUCTURES
  Primitive cell: 2-atom fcc, a = 5.43 A (LATTICE_CONSTANT 10.2632 bohr
      for the ABACUS runs), cell = [[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]].
  Supercell: 8-atom conventional cubic cell, supercell = M @ primitive
      with M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]].
  Basis: DZP numerical orbitals Si_gga_10au_100Ry_2s2p1d
      (13 orbitals per Si atom); the Si7P fixture adds
      P_gga_10au_100Ry_2s2p1d for the dopant (also 13 orbitals).
  Run settings: PBE, ecutwfc 100 Ry, symmetry 0, Gamma-centered 2x2x2
      k-grid, out_mat_hs2 1, smearing gauss sigma 0.005 Ha.

K-PATH
  Gamma-X-W-Gamma-L-W-X, 300 points, fcc special points in primitive
  reciprocal fractional coordinates (Setyawan-Curtarolo):
      G (0,0,0), X (1/2,0,1/2), W (1/2,1/4,3/4), L (1/2,1/2,1/2)
  Segment point counts are proportional to Cartesian length. Energies in
  eV with 0 at the Fermi level parsed from each run's running_scf.log;
  plot window -13..8 eV.

PARAMETERS THAT MATTER
  supercell matrix M   row convention supercell = M @ primitive.
  orb_counts           orbitals per atom (13 here) used by RelabelMap.
  match_species        True for pristine; False for Si7P so the P dopant
                       folds onto the host Si site it replaces.
  method               "ideal" generic-k spectral weight.

COMMANDS
  python reproduce.py                 # writes both PNGs next to it
  unfolding --config unfold.toml      # CLI route (pristine figure)

EXPECTED OUTPUT
  abacus_si_unfolded.png and abacus_si_p_doped.png (150 dpi): weight-
  coded unfolded bands in eV with tick labels Gamma X W Gamma L W X and
  primitive-cell bands overlaid in crimson. Runtime: seconds per figure.

HOW THE FIXTURES WERE PRODUCED
  ABACUS (LCAO, ksdft esolver) with the official Si/P UPFs and orbital
  files from refs/: si_prim (2-atom cell, 4x4x4 k-grid), si_conv and
  si7p (8-atom cells, 2x2x2 k-grid), each an SCF with out_mat_hs2 1.
  A k-sampled SCF is essential: a Gamma-only run collapses all image
  shells into the R=0 block, which cannot be unfolded at generic
  momenta. To re-run ABACUS yourself: point pseudo_dir/orbital_dir at
  refs/ (the shipped INPUT files use ../refs) and run abacus in each
  data/<name> directory.
