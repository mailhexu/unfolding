====================================================================
GPAW plane-wave pristine Si unfolding bundle (gpaw-si-pw)
====================================================================

Companion bundle for the pristine Si path example. Unfold the GPAW
plane-wave restart for the 8-atom conventional cubic Si cell onto the
primitive fcc path Gamma-X-W-Gamma-L-W-X (305 points), with independently
computed primitive-cell plane-wave bands as a reference overlay in the
Python reproducer.

BUNDLE CONTENTS
  generate_restart.py  generate the pristine path restart with GPAW
  reproduce.py         render the pristine unfolded path figure
  unfold.toml          config for the `unfolding` CLI
  data/
    si_prim_pw_path_bands.npz   primitive-cell reference bands

The 290 MB si8_pw.gpw restart is not shipped. From this directory, run
`python generate_restart.py`; the PBE 340 eV, 24-band calculation writes
`data/si8_pw.gpw` using the exact fixture-mapped 305-point path.

PREREQUISITES
  - Python 3.11 with numpy, matplotlib, ASE and GPAW >= 25, plus the
    unfolding package importable (pip install -e <unfolding repo checkout>,
    or PYTHONPATH=<unfolding repo checkout>).
  - A machine with sufficient memory and disk space for the GPAW PW run.

STRUCTURE AND SAMPLING
  Primitive cell: 2-atom fcc, a = 5.43 A.
  Supercell: 8-atom conventional cubic cell, supercell = M @ primitive,
      M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]].
  Run: PBE plane-wave mode, 340 eV cutoff, symmetry='off', mode='all'.
      The SCF samples the mapped 305-point path directly.

The Gamma-only Si7P restart covers only one supercell momentum and does
not sample a band path. Its sparse doped figure is intentionally suppressed;
the restart remains in the source test fixtures, not in this public bundle.

COMMANDS
  python generate_restart.py
  python reproduce.py
  unfolding --config unfold.toml

The reproducer writes `gpaw_si_pw_unfolded.png` with primitive reference
bands overlaid. The CLI writes the distinct `gpaw_si_pw_cli.png` output.
The config lists the exact fixture path; generic `special_points`/`npts`
flags do not recreate its per-segment sampling grid.
