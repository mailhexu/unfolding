====================================================================
SIESTA Si:P dopant -- reproduction bundle (siesta-p-doped)
====================================================================

Regenerates the figure of the "SIESTA Si:P dopant" example
(https://mailhexu.github.io/unfolding/examples/siesta-p-doped/): the
bundled 8-atom conventional-cubic supercell with one Si replaced by P
unfolded onto the 2-atom primitive fcc cell along
Gamma-X-W-Gamma-L-W-X. The substitution breaks the translation
symmetry: host bands keep weight 1 while donor-derived states appear
with fractional weight.

BUNDLE LAYOUT
  unfold.toml    config for the `unfolding` command (see CLI below)
  reproduce.py   self-contained script; regenerates the published figure
  data/          SIESTA fixtures, parsed in place:
                   si_prim.fdf  + si_prim.HSX    pristine primitive cell
                   si_sc_p.fdf  + si_sc_p.HSX    Si7P supercell
                   si_prim.vasp                 primitive cell as a POSCAR
  pseudos/       P.psml plus a note on Si.psf (see PSEUDOPOTENTIALS)

PREREQUISITES
  - Python 3.11+ with numpy, scipy, matplotlib, sisl, HamiltonIO
  - the unfolding package: pip install unfolding

STRUCTURE AND PATH
  - identical to the pristine siesta-si bundle: supercell matrix
    M = [[-1,1,1],[1,-1,1],[1,1,-1]] (conventional = M @ primitive),
    path Gamma-X-W-Gamma-L-W-X, 300 points, fcc primitive-frame points
  - the P dopant maps onto the host site it replaces via
    match_species = false (same position and orbital count, species
    ignored): host bands stay at weight 1 while donor-derived states
    carry fractional weight

HOW TO RUN
  From the unpacked bundle directory, either the CLI:

    unfolding --config unfold.toml        # writes si_p_doped_unfolded_cli.png
    # equivalent explicit flags:
    unfolding siesta --fdf data/si_sc_p.fdf --primitive data/si_prim.vasp \
        --unfold-mat -1 1 1 1 -1 1 1 1 -1 --special-points GXWGLX \
        --npts 300 --method ideal --no-match-species \
        --output si_p_doped_unfolded_cli.png

  or the Python reproduction script (the published figure, with the
  fraction of binary vs fractional weights printed):

    python reproduce.py                   # writes si_p_doped_unfolded.png
    python reproduce.py out.png           # or any output path

  The script prints the fraction of binary vs fractional weights:
  host bands stay at weight 1, donor-derived states carry fractional
  weight. Runtime: a few seconds on one core.

FIXTURE PROVENANCE
  Same SIESTA setup as the pristine example (GGA/PBE, SZ PAO basis,
  MeshCutoff 100 Ry, 2x2x2 k-grid SCF, a = 5.430 Ang) with one
  substitutional P.

PSEUDOPOTENTIALS
  P.psml ships in pseudos/ (needed only to re-run SIESTA from scratch;
  the shipped .HSX already contains the Hamiltonian). The inputs also
  reference Si.psf (norm-conserving, SIESTA distribution
  Examples/Si_Optical set), which is NOT included; see
  pseudos/PLACEHOLDER-NOT-INCLUDED.txt.
