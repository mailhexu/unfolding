# Local example data

## Policy and layout

Keep large/private runtime inputs **outside Git**, in one persistent store:

```text
unfolding_dev/
├── unfolding/                  # Git checkout: code, configs, docs, regression fixtures
├── unfolding-data/             # local raw inputs; back up separately, never publish
│   ├── inventory.json          # bytes, SHA-256, original paths, redistribution status
│   ├── abinit-si/
│   ├── abinit-paw/
│   ├── gpaw/
│   └── vasp-fe/runs/
└── .tmp/local-examples/         # disposable runnable bundles and generated outputs
```

The default store is the checkout's sibling `unfolding-data/`. Override it with
`UNFOLDING_DATA_DIR` or `--data-root`. The store is not a scratch directory:
never include it in `.tmp` cleanup. Back it up together with `inventory.json`;
Git alone cannot restore it.

Existing committed regression fixtures stay in `tests/data/` and the legacy
example directories so clean-checkout tests and CI keep their existing inputs.
They are intentionally distinct from runtime-only data. Do not add new large
raw outputs or licensed files to Git. `.gitignore` excludes new GPAW restarts,
WFKs, WAVECARs and POTCARs; tracked fixtures are unaffected.

There are two authoritative catalogs, with different roles:

- `examples/<name>/bundle.manifest`: **committed regression inputs** pulled into
  downloadable archives by `docgen/pack_bundles.py`.
- `docgen/local-data.toml`: **runtime-only inputs** pulled from the external
  store into local runs. This catalog contains paths, not data. The public
  bundle packer never reads it, so VASP/private and oversized files stay out
  of published archives.

A source `examples/<name>/bundle/` directory is not necessarily runnable by
itself: manifest pulls are applied only during packing or local staging.

## Assemble and run an example

From the checkout, using Python 3.11 (or Python 3.9/3.10 with `tomli`):

```bash
python docgen/stage_examples.py --check
python docgen/stage_examples.py --output ../.tmp/local-examples
```

Or stage only one backend with a different store:

```bash
python docgen/stage_examples.py gpaw-si-pw \
    --data-root /path/to/unfolding-data \
    --output /path/to/scratch/examples
```

The tool copies bundle scripts/configs, then links manifest inputs and external
inputs into their expected locations. Raw input bytes are not duplicated.
Missing or empty files fail before staging starts. Existing bundle destinations
are never overwritten; select a fresh output directory for another run. Both
store and output must be outside the checkout, and output must not be inside
the store. Do not run reproduction scripts that regenerate input files against
this linked staging tree: they could overwrite the canonical source through a
symlink. Use a separate, copied-data calculation directory for regeneration.

```bash
cd ../.tmp/local-examples/abinit-wfk-si
unfolding --config unfold.toml
```

For the DDB bundle, both `unfold.toml` (Cu) and `unfold-catio3.toml` are staged.
Optional Python backends and external executables such as `anaddb` still need
to be installed/on `PATH`. Staging does not generate DFT data or install tools.

## Runtime input inventory

Store-relative paths used by `docgen/local-data.toml`:

| Example | Runtime source | Generation/provenance |
|---|---|---|
| ABINIT Si | `abinit-si/si8_gxwglwxo_DS2_WFK.nc` | Exact dense path; `tests/data/abinit_si/regenerate_on_nic6.sh`, `si8_gxwglwx.abi` |
| ABINIT NiO | `abinit-si/nio_afm_patho_DS2_WFK.nc` | Dense AFM path; same regeneration script and `nio_afm.abi` |
| GPAW LCAO Si | `gpaw/si_sc_lcao.gpw`, `si_prim_lcao.gpw`, `si7p_lcao.gpw` | Local runtime copies of the existing committed fixtures; generation in `examples/gpaw_si/generate_fixtures.py` |
| GPAW PW Si | `gpaw/si8_pw.gpw` | Preserved dense-path restart from workspace `.tmp/story037/`; `examples/gpaw-si-pw/bundle/generate_restart.py` documents generation |
| VASP Fe | `vasp-fe/runs/{sc16_nscf,prim_nscf}/{WAVECAR,POSCAR}` and `sc16_nscf/POTCAR` | Preserved run outputs from `/tmp/unfolding_vasp_fe`; `examples/vasp-fe/bundle/make_inputs.py` and README describe calculation inputs |

The store also preserves ABINIT primitive-reference and dense PAW path WFKs,
GPAW generation restarts and the VASP SCF OUTCARs. They are recorded in
`inventory.json` even when a shipped TOML does not use them. Redundant GPAW
restarts are retained rather than deleting potentially useful provenance.

The initial relocation on 2026-10-09 verified size and SHA-256 before and after
each move. Old ignored file locations are local symlinks to the store so existing
local tests/scripts still resolve them; those links are not committed. The
original private POTCAR seed outside this project was not moved. All tracked
regression fixtures remain unchanged.

VASP POTCARs are licensed: do not publish them or a local staged VASP directory.
WAVECARs and all other local runtime assets are excluded from bundle publication
regardless of their redistribution rights.

## Add or replace local data

1. Generate/retrieve data in a separate calculation directory. Keep the exact
   cell, k-grid/path, spin, basis/pseudopotential and backend-version information.
   A Gamma-only or corner-only restart cannot replace a dense-path restart.
2. Copy the required files into a stable backend subdirectory of the external
   store. Preserve the previous dataset under a distinct name if it is still
   needed. Never overwrite an input used by an active audit.
3. Record each file's byte size, SHA-256, origin/generation recipe and licensing
   in the store's `inventory.json`. The initial inventory uses `version = 1`
   and `assets[store_relative_path]` entries with `bytes`, `sha256`,
   `original_paths`, `committed_fixture_copy`, and `redistribution` fields.
4. Update `docgen/local-data.toml` with `"bundle/destination" = "store/source"`
   under the bundle's table. Destinations are relative to its staged root, not
   to the checkout. Use the actual file paths expected by its TOML and scripts.
   Add committed regression pulls only to `bundle.manifest`, not this catalog.
5. Run `stage_examples.py <name> --check`, stage into a fresh directory, and
   exercise the actual CLI/Python route. Compare numerical arrays, not PNG
   hashes alone. Missing/failed examples are unverified, never counted as passes.

Check a backup against the inventory (Python standard library only):

```python
import hashlib
import json
from pathlib import Path

root = Path("/path/to/unfolding-data")
inventory = json.loads((root / "inventory.json").read_text())
for relative, record in inventory["assets"].items():
    path = root / relative
    assert path.stat().st_size == record["bytes"], relative
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()  # Python 3.11+
    assert actual == record["sha256"], relative
```

Contributor interface wiring is covered in
[Adding a new unfolding](adding-an-unfolding.md).

## Preservation snapshot (2026-10-09)

All **17 TOML configurations in 16 bundles** ran successfully against baseline
`285af7f` and the shared-core version `3f742d6`, using the same staged inputs.
Both DDB configurations (Cu and CaTiO3) are included. Every run produced a
figure and JSON dataset; there were no skips or unverified configurations.

- K-points, eigenvalues, and optional fold/supercell k-points were identical.
- Shapes, units, spin, Fermi/reference and recorded configuration metadata matched.
- Generated PNGs were byte-identical between baseline and current versions,
  both with configured degeneracy handling and in the second probe.
- Largest absolute weight difference: `2.19e-12` (ABACUS); OpenMX was
  `1.13e-12`, GPAW LCAO `1.82e-14`, and the remaining paths were at most
  `7.55e-15` or exact. Weight comparison used `atol=rtol=1e-10`.
- A second 17-configuration probe with optional degeneracy presentation
  disabled also passed, so grouped outputs did not hide raw-weight differences.

Reports, JSON datasets, PNGs, logs and the reproduction harness are preserved
locally under `unfolding-data/audits/2026-10-09/`. This evidence is not published
or committed. Existing unit-suite evidence is separate: 308 passed / 11 skipped
outside docgen; the known docgen `list.T` failure occurs on both versions.
