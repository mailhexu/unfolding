====================================================================
ABINIT WFK unfolding bundle: pristine Si and Si:P (8-atom conventional cell)
====================================================================

Companion bundle of the "ABINIT WFK Si and Si:P" example
(https://mailhexu.github.io/unfolding/examples/abinit-wfk/). Unfolds
ABINIT supercell wavefunction files (WFK, netCDF) onto the primitive fcc
band path Gamma-X-W-Gamma-L-W-X. The planewave basis is orthonormal, so
the spectral weight is a pure reciprocal-coset projection: for pristine
Si every state folds from a single primitive momentum, the weights are
binary, and the unfolded bands ARE the primitive band structure -- a
case with a known answer. Substituting one Si by P (12.5%) mixes the
fold sectors: host bands keep weight near 1 and render as the darkest
traces, while defect-hybridized states carry fractional weight.

Two figures are produced:

  si8_abinit_unfolded.png   pristine Si 8-atom supercell unfolded onto
                            the path (blue intensity = spectral weight),
                            with the independently computed 2-atom
                            primitive-cell bands overlaid in crimson
                            when the primitive reference WFK is present;
                            after a single constant potential-reference
                            shift the two agree to ~0.07 eV
  si7p_abinit_unfolded.png  Si:P spectral-weight map on the same path

Physics to check: pristine Si shows the LDA gap of ~0.4-0.5 eV with the
Fermi level mid-gap (indirect Gamma-X gap). If your plot shows a metal
with E_F inside the valence manifold and extra flat singlets below the
top valence triplet at Gamma, you have the ghost-pseudopotential hazard
described below.

BUNDLE LAYOUT
  unfold.toml  config for the `unfolding` command -- runnable once the
               dense Si8 WFK has been generated into data/ (see below)
  inputs/      ABINIT input files (two-dataset decks: SCF + non-SCF path)
               si8_gamma.abi              Si8 SCF at Gamma (ecut 12)
               si7p_gamma.abi             Si7P SCF at Gamma (ecut 12)
               si8_gxwglx_corners.abi     Si8, frozen-density run at the
                                          Gamma/X/W/L corners (ecut 25)
               si7p_gxwglx_corners.abi    Si7P, same corners (ecut 25)
               si7p_gamma_x_path.abi      Si7P, dense 305-point path (ecut 25)
               si8_gxwglwx.abi            pristine Si8, dense 305-point path
                                          (ecut 25) -- published figure input
               si_primitive_matched.abi   2-atom primitive cell, matched
                                          lattice (ecut 12)
               si_prim_path.abi           2-atom primitive cell, 8x8x8 SCF
                                          + 305-point path in primitive
                                          coordinates (ecut 25) -- the
                                          reference for the crimson overlay
  pseudos/     Troullier-Martins fhi pseudopotentials from the ABINIT
               test-suite Pspdir: 14-Si.nlcc.fhi, 15-P.LDA.fhi
  data/        WFK output directory; no binary WFK ships with this
               bundle (each dense path WFK is hundreds of MB) -- run the
               ABINIT commands printed by reproduce.py first
  reproduce.py renders both figures; run from this directory

PREREQUISITES
  - Python >= 3.9 with numpy and matplotlib
  - the unfolding package: pip install unfolding
  - only for regenerating the WFKs: ABINIT >= 9 built with netCDF
    support (the decks use iomode 3)

STRUCTURE AND PATH
  - conventional cubic cell, acell 3*10.26 bohr; supercell matrix
    M = [[-1,1,1],[1,-1,1],[1,1,-1]] (conventional = M @ primitive fcc
    cell), so supercell momenta are K = k_prim @ M.T
  - path Gamma-X-W-Gamma-L-W-X, exactly 305 primitive reciprocal-
    fractional points. data/path_kpoints.txt, path_xcoords.txt and
    path_xticks.txt contain the exact grid/axis used in examples; the
    non-SCF deck stores K = k_prim @ M.T.
  - ABINIT settings the WFK reader requires: iomode 3, istwfk 1 (full G
    sphere), prtwf 1, chkprim 0 for the non-primitive supercell; the
    decks are ndtset 2 (dataset 1: SCF at Gamma writing the density,
    dataset 2: iscf -2 / getden 2 non-SCF path sampling, kptopt 0)

HOW TO RUN
  1. Regenerate the WFKs from the unpacked bundle directory (reproduce.py
     prints these exact commands when data/ is empty):

       mkdir -p run_si8 && cd run_si8
       cp ../inputs/si8_gxwglwx.abi ../pseudos/14-Si.nlcc.fhi .
       abinit si8_gxwglwx.abi > si8_gxwglwx.abo 2>&1
       cp si8_gxwglwxo_DS2_WFK.nc ../data/ && cd ..

       mkdir -p run_si7p && cd run_si7p
       cp ../inputs/si7p_gamma_x_path.abi ../pseudos/14-Si.nlcc.fhi ../pseudos/15-P.LDA.fhi .
       abinit si7p_gamma_x_path.abi > si7p_gamma_x_path.abo 2>&1
       cp si7p_gamma_x_patho_DS2_WFK.nc ../data/ && cd ..

       # optional, for the crimson primitive-cell overlay:
       mkdir -p run_prim && cd run_prim
       cp ../inputs/si_prim_path.abi ../pseudos/14-Si.nlcc.fhi .
       abinit si_prim_path.abi > si_prim_path.abo 2>&1
       cp si_prim_patho_DS2_WFK.nc ../data/ && cd ..

     Runtime: Si8 305-point path ~20 min on 8 cores; Si7P and the
     primitive reference similar or cheaper. Disk: ~787 MB per dense
     supercell WFK, ~103 MB for the primitive reference. (The corner
     decks give small WFKs -- a few MB -- enough for coarse figures.)

  2. Render the figures, either the CLI (Si8 shown; edit the [input]
     wfk path for Si:P):

       unfolding --config unfold.toml        # writes si8_abinit_unfolded_cli.png

     or the reproduction script:

       python reproduce.py                   # writes both published PNGs

     reproduce.py automatically uses the dense WFKs when present and
     falls back to the corner WFKs (coarse maps on the same path axis).

EXPECTED OUTPUT
  With the dense WFKs: the published 305-point maps (plus the crimson
  primitive overlay when si_prim_patho_DS2_WFK.nc is present).
  Energies are relative to the WFK Fermi level (eV).

GHOST-PSEUDOPOTENTIAL HAZARD
  Use the pseudopotentials shipped in pseudos/ (trusted ABINIT Pspdir
  fhi files). An earlier version of these fixtures used a locally
  generated ONCVPSP-4.0.1 file whose header ABINIT misparses (lloc read
  as 4): that silently adds flat ghost bands below the valence manifold
  and puts the Fermi level inside the valence -- pristine Si renders as
  a metal. The tell-tale is a Gamma-point spectrum with extra singlets
  below the top valence triplet. If you swap in your own pseudos,
  re-check the Gamma spectrum first.
