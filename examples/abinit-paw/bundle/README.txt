====================================================================
ABINIT PAW Si / Si:P unfolding -- reproduction bundle (abinit-paw)
====================================================================
Docs example: docs/content/examples/abinit-paw.md in the unfolding
repository. Regenerates abinit_paw_si7p_gamma.png out of the box: the
committed Si7P Gamma-point PAW WFK unfolded onto the four primitive
folds of supercell Gamma (PAW S-metric weights vs pseudo-only coset
fractions side by side). With the dense-path WFKs regenerated from the
included decks (bash regenerate_path.sh, needs your own ABINIT and
~1 GB per WFK) you can additionally produce the docs-page figure
abinit_paw_si_path.png along the full Gamma-X-W-Gamma-L-W-X path.

BUNDLE LAYOUT
  reproduce.py         self-contained script; Gamma-fold figure
  regenerate_path.sh   runs the dense-path decks with a local abinit
  unfold.toml          config for the `unfolding` CLI (Gamma computation):
                         unfolding --config unfold.toml
  data/                committed fixtures (small Gamma WFKs + JTH XMLs)
                         si8_gammao_WFK.nc            pristine Si8 Gamma WFK
                         si7p_gammao_WFK.nc           Si7P Gamma WFK
                         si_primitive_foldso_WFK.nc   Si2 primitive Gamma WFK
                         Si.xml, P.xml                JTH PAW datasets
  inputs/              the ABINIT decks that produced the fixtures
                         si8_gamma.abi, si7p_gamma.abi,
                         si_primitive_folds.abi         (Gamma runs)
                         si8_paw_path.abi, si7p_paw_path.abi,
                         si_prim_paw_path.abi           (dense-path decks)

PREREQUISITES
  - Python 3.11 with numpy, matplotlib, netCDF4, pypao (JTH PAW XML
    reader) and HamiltonIO, plus the unfolding package importable (pip
    install -e <unfolding repo checkout>, or PYTHONPATH=<unfolding repo
    checkout>). Tested with unfolding 0.1.0.
  - No ABINIT run is needed for the Gamma figure. ABINIT >= 10 is only
    needed to regenerate the dense-path WFKs (hundreds of MB each, not
    shippable under the 10 MB bundle cap).

STRUCTURES
  Primitive cell: 2-atom fcc, acell 3*10.26 bohr,
      rprim 1 0 0  0 1 0  0 0 1, Si at xred (0,0,0) and (1/4,1/4,1/4).
  Supercell: 8-atom conventional cubic cell, supercell = M @ primitive
      with M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]; the Si7P fixture
      replaces the (1/4,1/4,1/4) atom by P (znucl 15).
  Atomic data: JTH PAW datasets (Psdj_paw_pbe_std) Si.xml / P.xml,
      ixc 11, ecut 10 Ha, pawecutdg 20 Ha, nband 32, occopt 3,
      tsmear 0.005, iomode 3 (netCDF WFK), istwfk 1 at Gamma.

K-PATH
  Gamma folds of supercell Gamma (reproduce.py): G, (0,1/2,1/2),
  (1/2,0,1/2), (1/2,1/2,0) in primitive reciprocal fractional
  coordinates.
  Dense path (regenerate_path.sh + docs-page recipe):
  Gamma-X-W-Gamma-L-W-X, 305 points, fcc special points in primitive
  reciprocal fractional coordinates (Setyawan-Curtarolo); the supercell
  decks sample the same path in supercell coordinates K = k_prim @ M.T
  as an explicit kpt2 list (kptopt 0). Energies in eV relative to the
  NSCF Fermi level; the figure window is -13..8 eV.

PARAMETERS THAT MATTER
  datasets            maps chemical symbols to the JTH XML paths.
  matrix              M above, row convention supercell = M @ primitive;
                      verified against the two WFK rprimd on entry.
  resolve_degenerate  1e-3 eV (passed in Hartree): ABINIT stores
                      degenerate states in an arbitrary unitary gauge;
                      the tolerance eigen-assigns gauge-invariant
                      branch weights inside degenerate groups.
  pseudo_weights      the pseudo-coset fractions, resolved with the
                      same degenerate blocks (result attribute).

COMMANDS
  python reproduce.py                 # writes abinit_paw_si7p_gamma.png
  unfolding --config unfold.toml      # CLI route (same Gamma figure)
  bash regenerate_path.sh             # dense-path WFKs -> data/ (ABINIT)

EXPECTED OUTPUT
  abinit_paw_si7p_gamma.png (160 dpi): two panels of weight-coded
  scatter at the four folds, energies in eV relative to the Si7P run's
  Fermi level. Console: maximum PAW norm error (S-metric completeness)
  and the max PAW-pseudo weight difference. Runtime: seconds.

HOW THE FIXTURES WERE PRODUCED
  ABINIT 10.5.8 with the bundled JTH XMLs. Gamma fixtures: ndtset-free
  SCF decks at supercell Gamma (kptopt 0, nkpt 1, istwfk 1, prtwf 1).
  Dense path: ndtset 2 -- dataset 1 SCF at supercell Gamma writing the
  density, dataset 2 frozen-density non-SCF (iscf -2, getden 1,
  tolwfr 1e-16) over the 305-point kpt list with prtwf 1; the primitive
  deck samples the same path in primitive coordinates and supplies the
  embedded reference bank. chkprim 0 is needed because the supercell is
  not primitive.
