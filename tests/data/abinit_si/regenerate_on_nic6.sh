# Regenerate the committed ABINIT Si and Si7P WFK fixtures on nic6.
#
# Usage (from any directory):
#   bash tests/data/abinit_si/regenerate_on_nic6.sh
#
# Defaults are specific to this workstation and the swmgr-recorded nic6 build.
# The fixtures use the trusted ABINIT Pspdir LDA pseudos (Troullier-Martins
# fhi, committed next to the inputs): the previously used locally generated
# Dojo-NC-SR ONCVPSP-4.0.1 files carry header fields that ABINIT misparses
# (lloc read as 4), which injects flat ghost bands below the valence and
# breaks the Fermi level. Do NOT swap them back in without a ghost check.
set -euo pipefail

fixture_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
host=${ABINIT_HOST:-nic6}
si_psp8=${SI_PSP8:-${fixture_dir}/14-Si.nlcc.fhi}
p_psp8=${P_PSP8:-${fixture_dir}/15-P.LDA.fhi}
stage=${TMPDIR:-/tmp}/unfolding-abinit-si-fixtures-$$

cleanup() {
  rtk rm -rf "${stage}"
  if [[ ${KEEP_REMOTE:-0} != 1 ]]; then
    rtk ssh "${host}" "rm -rf '${remote}'"
  fi
}
trap cleanup EXIT
rtk cp "${si_psp8}" "${stage}/14-Si.nlcc.fhi"
rtk cp "${p_psp8}" "${stage}/15-P.LDA.fhi"
rtk md5sum "${stage}/14-Si.nlcc.fhi" "${stage}/15-P.LDA.fhi"

rtk ssh "${host}" "rm -rf '${remote}' && mkdir -p '${remote}'"
rtk scp \
  "${fixture_dir}/si_primitive_matched.abi" \
  "${fixture_dir}/si8_gamma.abi" \
  "${fixture_dir}/si7p_gamma.abi" \
  "${fixture_dir}/si7p_gamma_x_path.abi" \
  "${stage}/14-Si.nlcc.fhi" \
  "${stage}/15-P.LDA.fhi" \
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
