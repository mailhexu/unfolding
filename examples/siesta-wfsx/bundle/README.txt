====================================================================
SIESTA WFSX route -- reproduction bundle (siesta-wfsx)
====================================================================

Regenerates the figure of the "SIESTA WFSX route" example
(https://mailhexu.github.io/unfolding/examples/siesta-wfsx/): unfold
from SIESTA's own stored wavefunctions instead of diagonalizing the
Hamiltonian. Same 8-atom conventional-cubic Si supercell and primitive
fcc path as the siesta-si bundle; only the eigen-solve is replaced --
the weight still uses the .HSX overlap shells.

BUNDLE LAYOUT
  unfold.toml    config for the `unfolding` command (see CLI below)
  reproduce.py   self-contained script; regenerates the published figure
  data/          SIESTA fixtures, parsed in place:
                   si_prim.fdf + si_prim.HSX          primitive cell
                   si_sc.fdf   + si_sc.HSX            supercell (overlap shells)
                   si_sc_path.selected.WFSX           stored path wavefunctions
                   si_sc_path.EIG                     eigenvalues + Fermi level
                   si_prim.vasp                       primitive cell as a POSCAR
  inputs/        the SIESTA inputs, including si_sc_path.fdf -- the path
                 run deck with `SaveWFSX true` and a %block WaveFuncKPoints
                 list covering the unfolding path (150 points)
  pseudos/       pseudopotential notes (see PSEUDOPOTENTIALS below)

PREREQUISITES
  - Python 3.11+ with numpy, scipy, matplotlib, sisl, HamiltonIO >= 0.3.6
    (SiestaWFSXParser)
  - the unfolding package: pip install unfolding
  No SIESTA run is needed: the fixtures in data/ are complete.

STRUCTURE AND PATH
  - M = [[-1,1,1],[1,-1,1],[1,1,-1]] (conventional = M @ primitive)
  - path Gamma-X-W-Gamma-L-W-X over the EXACT grid SIESTA stored
    wavefunctions on: the segment-proportional list of the run deck's
    %block WaveFuncKPoints (150 points nominal -> 157 with repeated
    junctions). The unfolder matches requested momenta to the stored
    entries at 1e-6 tolerance, so unfold.toml carries that grid
    explicitly; a regenerated special_points path would not hit the
    stored points.
  - WFSX energies are returned exactly as SIESTA stores them (shifted
    by the writing run's Fermi level); the route subtracts the .EIG
    header value by default (options.efermi = null)

HOW TO RUN
  From the unpacked bundle directory, either the CLI:

    unfolding --config unfold.toml        # writes si_wfsx_unfolded_cli.png
    # equivalent explicit flags (kpoint list as in unfold.toml):
    unfolding siesta-wfsx --wfsx data/si_sc_path.selected.WFSX \
        --hs-fdf data/si_sc.fdf --primitive data/si_prim.fdf \
        --unfold-mat -1 1 1 1 -1 1 1 1 -1 --kpoints ... \
        --names G X W G L W X --method ideal \
        --output si_wfsx_unfolded_cli.png

  or the Python reproduction script (the published figure, with the
  primitive-cell overlay and the .EIG Fermi subtraction):

    python reproduce.py                   # writes si_wfsx_unfolded.png
    python reproduce.py out.png           # or any output path

  Runtime: a few seconds on one core. The weight is computed with the
  same "ideal" method as the HSX route; WFSX coefficients use SIESTA's
  orbital-position gauge, converted internally by the unfolder.

FIXTURE PROVENANCE
  Same SIESTA setup as the pristine example (GGA/PBE, SZ PAO basis,
  MeshCutoff 100 Ry, SaveHS true) plus the path run with SaveWFSX true.

PSEUDOPOTENTIALS
  Si.psf (norm-conserving, SIESTA distribution Examples/Si_Optical
  set) is NOT included and is NOT needed to reproduce the figure; see
  pseudos/PLACEHOLDER-NOT-INCLUDED.txt.
