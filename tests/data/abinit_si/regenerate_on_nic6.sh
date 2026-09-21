#!/usr/bin/env bash
# Regenerate the committed ABINIT Si and Si7P WFK fixtures on nic6.
#
# Usage (from any directory):
#   bash tests/data/abinit_si/regenerate_on_nic6.sh
#
# Defaults are specific to this workstation and the swmgr-recorded nic6 build.
# Override ABINIT_HOST, ABINIT_REMOTE_DIR, SI_PSP8, or P_PSP8 when needed.
# Pseudo-dojo PSP8 files begin with a marker that ABINIT 10.9 does not parse;
# the generator removes only that marker before staging the records.
set -euo pipefail

fixture_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
host=${ABINIT_HOST:-nic6}
remote=${ABINIT_REMOTE_DIR:-/scratch/hexu/unfolding-abinit-si-fixtures}
si_psp8=${SI_PSP8:-/home/hexu/projects/ABACUS-orbitals/generated/Dojo-NC-SR/PBE/Si.psp8}
p_psp8=${P_PSP8:-/home/hexu/projects/ABACUS-orbitals/generated/Dojo-NC-SR/PBE/P.psp8}
stage=${TMPDIR:-/tmp}/unfolding-abinit-si-fixtures-$$

cleanup() {
  rtk rm -rf "${stage}"
  if [[ ${KEEP_REMOTE:-0} != 1 ]]; then
    rtk ssh "${host}" "rm -rf '${remote}'"
  fi
}
trap cleanup EXIT

test -f "${si_psp8}"
test -f "${p_psp8}"
rtk mkdir -p "${stage}"
rtk tail -n +2 "${si_psp8}" > "${stage}/Si.psp8"
rtk tail -n +2 "${p_psp8}" > "${stage}/P.psp8"
rtk md5sum "${stage}/Si.psp8" "${stage}/P.psp8"

rtk ssh "${host}" "rm -rf '${remote}' && mkdir -p '${remote}'"
rtk scp \
  "${fixture_dir}/si_primitive_matched.abi" \
  "${fixture_dir}/si8_gamma.abi" \
  "${fixture_dir}/si7p_gamma.abi" \
  "${fixture_dir}/si7p_gamma_x_path.abi" \
  "${stage}/Si.psp8" \
  "${stage}/P.psp8" \
  "${host}:${remote}/"

rtk ssh "${host}" "bash -s -- '${remote}'" <<'REMOTE'
set -euo pipefail
remote=$1
module use ~/privatemodules
module load abinit/dev@d5380e0cb-gcc
cd "${remote}"
for input in si_primitive_matched.abi si8_gamma.abi si7p_gamma.abi si7p_gamma_x_path.abi; do
  base=${input%.abi}
  abinit "${input}" > "${base}.abo" 2>&1
done
REMOTE

rtk scp \
  "${host}:${remote}/si_primitive_matchedo_WFK.nc" \
  "${host}:${remote}/si8_gammao_WFK.nc" \
  "${host}:${remote}/si7p_gammao_WFK.nc" \
  "${host}:${remote}/si7p_gamma_x_patho_DS2_WFK.nc" \
  "${fixture_dir}/"
