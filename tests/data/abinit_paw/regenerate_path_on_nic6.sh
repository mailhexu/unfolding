# Regenerate the PAW dense-path WFK fixtures on nic6: pristine Si8 and
# Si7P supercell 305-point GXWGLWX decks plus the Si2 primitive reference
# path used as the embedded reference bank.
#
# Usage (from the unfolding repository):
#   bash tests/data/abinit_paw/regenerate_path_on_nic6.sh
#
# The DS2 WFKs (hundreds of MB) are copied back into tests/data/abinit_paw
# but stay external: a .gitignore next to the decks keeps them out of git.
# The module name tracks the swmgr-recorded nic6 build; the committed PAW
# fixtures were produced with ABINIT 10.5.8 (JTH Psdj_paw_pbe_std XMLs,
# bundled beside the decks).
#
# Staging happens under /scratch (the nic6 home tree is over quota); the
# three decks run concurrently, each serially.
set -euo pipefail

fixture_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
host=${ABINIT_HOST:-nic6}
remote=${ABINIT_REMOTE_DIR:-/scratch/ulg/phythema/hexu/unfolding-abinit-paw-path-$$}
stage=${TMPDIR:-/tmp}/unfolding-abinit-paw-path-$$

cleanup() {
  rtk rm -rf "${stage}"
  if [[ ${KEEP_REMOTE:-0} != 1 ]]; then
    rtk ssh "${host}" "rm -rf '${remote}'" || true
  fi
}
trap cleanup EXIT
rtk mkdir -p "${stage}"
rtk cp "${fixture_dir}/Si.xml" "${stage}/Si.xml"
rtk cp "${fixture_dir}/P.xml" "${stage}/P.xml"

rtk ssh "${host}" "rm -rf '${remote}' && mkdir -p '${remote}'"
rtk scp \
  "${fixture_dir}/si8_paw_path.abi" \
  "${fixture_dir}/si7p_paw_path.abi" \
  "${fixture_dir}/si_prim_paw_path.abi" \
  "${stage}/Si.xml" \
  "${stage}/P.xml" \
  "${host}:${remote}/"

rtk ssh "${host}" "bash -s -- '${remote}'" <<'REMOTE'
set -euo pipefail
remote=$1
module use ~/privatemodules
module load abinit/dev@18e802625-gcc
cd "${remote}"
for input in si_prim_paw_path.abi si8_paw_path.abi si7p_paw_path.abi; do
  base=${input%.abi}
  abinit "${input}" > "${base}.abo" 2>&1 &
done
wait
for base in si_prim_paw_path si8_paw_path si7p_paw_path; do
  [[ -s "${base}o_DS2_WFK.nc" ]] || {
    echo "${base} produced no DS2 WFK"; tail -5 "${base}.abo"; exit 1; }
done
REMOTE

rtk scp \
  "${host}:${remote}/si_prim_paw_patho_DS2_WFK.nc" \
  "${host}:${remote}/si8_paw_patho_DS2_WFK.nc" \
  "${host}:${remote}/si7p_paw_patho_DS2_WFK.nc" \
  "${fixture_dir}/"
