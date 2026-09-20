#!/bin/bash
# Regenerate the Si example fixtures on nic6 (SIESTA 5.1 perspin build).
# Usage: bash regenerate_fixtures.sh   (from examples/si_example/)
# Inputs committed here: si_prim.fdf, si_sc.fdf (SZ PBE, SaveHS TSHS,
# SaveWFSX). Outputs retrieved: *.HSX, *.selected.WFSX (tests/data/si_example/).
set -euo pipefail
R=nic6:/scratch/hexu/tmp/unfolding-si
ssh nic6 'mkdir -p /scratch/hexu/tmp/unfolding-si/{prim,sc}
          cp ~/src/siesta_perspin@ccbb20e66/Examples/Si_Optical/Si.psf /scratch/hexu/tmp/unfolding-si/'
scp si_prim.fdf si_sc.fdf $R/
scp si_prim.fdf $R/prim/ && scp si_sc.fdf $R/sc/
ssh nic6 'bash -lc "module use ~/privatemodules; module load siesta/dev@ccbb20e66-gcc
  cd /scratch/hexu/tmp/unfolding-si/prim && siesta < si_prim.fdf > si_prim.out 2>&1
  cd /scratch/hexu/tmp/unfolding-si/sc && siesta < si_sc.fdf > si_sc.out 2>&1"'
scp $R/prim/si_prim.HSX $R/prim/si_prim.selected.WFSX $R/prim/si_prim.fdf tests/data/si_example/
scp $R/sc/si_sc.HSX $R/sc/si_sc.selected.WFSX $R/sc/si_sc.fdf tests/data/si_example/
