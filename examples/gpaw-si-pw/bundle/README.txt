====================================================================
GPAW plane-wave Si unfolded bands -- reproduction bundle (gpaw-si-pw)
====================================================================
Docs example: docs/content/examples/gpaw-si-pw.md in the unfolding
repository. Regenerates gpaw_si_p_pw_unfolded.png out of the box: the
shipped Si7P Gamma-point plane-wave restart unfolded into its four
primitive folds of supercell Gamma. With a user-supplied pristine path
restart (data/si8_pw.gpw, ~290 MB, NOT shipped) it also regenerates
gpaw_si_pw_unfolded.png: the 8-atom conventional-cubic Si supercell
unfolded onto the primitive path with the independently computed
primitive-cell plane-wave bands overlaid.

BUNDLE LAYOUT
  reproduce.py   self-contained script; regenerates the figures
  unfold.toml    config for the `unfolding` CLI (Si7P folds computation):
                   unfolding --config unfold.toml
  data/
    si7p_pw.gpw                 shipped Si7P Gamma restart (1.1 MB)
    si_prim_pw_path_bands.npz   primitive-cell PW bands along the path
                                (overlay reference; eigenvalues only)

PREREQUISITES
  - Python 3.11 with numpy, matplotlib and GPAW >= 25 (the .gpw restart
    reader; tested with gpaw 26), plus the unfolding package importable
    (pip install -e <unfolding repo checkout>, or PYTHONPATH=<unfolding
    repo checkout>).
  - The doped figure runs out of the box. The pristine path figure
    needs data/si8_pw.gpw from your own GPAW run (recipe below).

STRUCTURES
  Primitive cell: 2-atom fcc, a = 5.43 A,
      cell = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]].
  Supercell: 8-atom conventional cubic cell, supercell = M @ primitive
      with M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]; the Si7P fixture
      substitutes P on the (1/4,1/4,1/4) site.
  Runs: PBE plane-wave mode, PW cutoff 340 eV, symmetry='off',
      mode='all' restarts (wavefunctions stored). si7p_pw.gpw is a
      Gamma-point-only SCF; si8_pw.gpw samples the path k-set directly
      (an SCF on the fixed path sampling; gpaw >= 25 dropped the old
      fixdensity non-SCF flow).

K-PATH
  Pristine figure: Gamma-X-W-Gamma-L-W-X, 300 points, fcc special
  points in primitive reciprocal fractional coordinates
  (Setyawan-Curtarolo):
      G (0,0,0), X (1/2,0,1/2), W (1/2,1/4,3/4), L (1/2,1/2,1/2)
  Doped figure: the four primitive momenta folding to supercell Gamma:
      G, (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0) -- plotted as separate
  columns because a Gamma-only run never sampled a continuous path.
  Energies in eV relative to each run's own Fermi level.

PARAMETERS THAT MATTER
  matrix               M above, row convention supercell = M @ primitive;
                       the unfolder maps each requested primitive k to
                       its stored supercell momentum K = k @ M.T.
  resolve_degenerate   1e-3 eV; reassigns gauge-invariant weights inside
                       near-degenerate groups so exact degeneracies
                       render as clean lines.
  Weight meaning       pseudo-wavefunction reciprocal-coset fractions
                       (GPAW normalizes in the PAW overlap metric; the
                       parser renormalizes to sum |c|^2 = 1, which leaves
                       coset fractions unchanged). These are NOT PAW
                       all-electron spectral weights.

COMMANDS
  python reproduce.py                 # doped figure; pristine if staged
  python reproduce.py --doped         # only the Si7P folds figure
  python reproduce.py --pristine      # only the path figure
  unfolding --config unfold.toml      # CLI route (Si7P folds figure)

EXPECTED OUTPUT
  gpaw_si_p_pw_unfolded.png: weight-coded scatter (marker area and alpha
  encode the weight) at the four folds; console prints the four-fold
  sum-rule error and the donor-window fold weights. gpaw_si_pw_unfolded.png
  (when si8_pw.gpw is staged): weight-coded unfolded bands with the
  primitive-cell PW bands in crimson (a constant potential shift between
  the two runs' references is reported and applied to the overlay).

HOW THE FIXTURES WERE PRODUCED / PRISTINE RECIPE
  GPAW >= 25, PBE, PW(340 eV), 24 bands (pristine) / default (doped),
  symmetry='off', gamma-centered sampling. The pristine fixture runs an
  SCF directly on the 300-point path mapped to supercell coordinates;
  examples/gpaw_si/generate_fixtures.py in the unfolding repository
  reproduces it exactly (run it and copy si8_pw.gpw into data/).
