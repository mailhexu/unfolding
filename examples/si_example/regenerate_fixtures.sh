#!/bin/bash
# Regenerate the Si example fixtures with a local SIESTA >= 5.4 build
# (SIESTA_BIN env var overrides the binary; Si.psf must be present in
# this directory -- e.g. from the SIESTA Examples/Si_Optical set).
# Usage: bash regenerate_fixtures.sh   (from examples/si_example/)
# Inputs committed here: si_prim.fdf, si_sc.fdf (SZ PBE, SaveHS true,
# SaveWFSX, WaveFuncKPointsScale ReciprocalLatticeVectors; the SC run uses
# a 2x2x2 k-grid so the .HSX keeps the full supercell shell set -- a
# Gamma-only run collapses the images into a single R=0 shell whose
# H(k) is k-independent and cannot be unfolded at generic k).
# Outputs produced: *.HSX, *.EIG, *.selected.WFSX -> tests/data/si_example/.
set -euo pipefail

SIESTA_BIN=${SIESTA_BIN:-$HOME/projects/siesta_git/siesta_spinor/_build/build_gcc13/Src/siesta}
test -x "$SIESTA_BIN"
test -f Si.psf

run() {  # run <dir> <fdf>
  local d=$1 fdf=$2
  rm -rf "$d" && mkdir -p "$d"
  cp "$fdf" Si.psf "$d"/
  ( cd "$d" && "$SIESTA_BIN" "$fdf" > run.out 2>&1 )
}

run prim si_prim.fdf
run sc   si_sc.fdf

cp prim/si_prim.HSX prim/si_prim.EIG prim/si_prim.selected.WFSX \
   prim/si_prim.fdf tests/data/si_example/
cp sc/si_sc.HSX sc/si_sc.selected.WFSX sc/si_sc.fdf tests/data/si_example/
echo "fixtures refreshed in tests/data/si_example/"
