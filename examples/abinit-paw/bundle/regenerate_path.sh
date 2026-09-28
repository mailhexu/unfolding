#!/usr/bin/env bash
# Regenerate the ABINIT PAW dense-path WFKs with a LOCAL abinit binary.
#
# Runs the three committed decks (pristine Si8 and Si7P supercells on the
# 305-point GXWGLWX supercell path K = k_prim @ M.T, plus the Si2
# primitive reference path used as the embedded reference bank) next to
# the JTH XML datasets, then drops the resulting *_DS2_WFK.nc files into
# data/ for the path-figure recipe on the docs page.
#
# Usage (from the unpacked bundle directory):
#   bash regenerate_path.sh
#
# Requirements: an `abinit` executable on PATH (the committed fixtures
# were produced with ABINIT 10.5.8) and ~1 GB of free space per WFK.
# Each deck runs serially; run them concurrently by hand if preferred.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${here}"

command -v abinit >/dev/null || { echo "no abinit on PATH" >&2; exit 1; }

for input in si_prim_paw_path.abi si8_paw_path.abi si7p_paw_path.abi; do
  base=${input%.abi}
  echo "=== ${input}"
  abinit "${input}" > "${base}.abo" 2>&1
  [[ -s "${base}o_DS2_WFK.nc" ]] || {
    echo "${base} produced no DS2 WFK"; tail -5 "${base}.abo"; exit 1; }
done

mkdir -p data
mv -f si_prim_paw_patho_DS2_WFK.nc si8_paw_patho_DS2_WFK.nc \
      si7p_paw_patho_DS2_WFK.nc data/
echo "done: data/*_DS2_WFK.nc written"
