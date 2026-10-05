# Figure generation (docgen)

Published docs figures have headless regeneration entry points here.
`tests/test_docgen.py` exercises the supported subset in CI; examples
requiring external DFT executables or unbundled inputs are exceptions.

| Script | Figure (docs/static/images/) | Data source | Notes |
|---|---|---|---|
| `fig_phonopy_cu.py` | `phonopy_unfolded_band_structure.png` | `examples/phonopy/FORCE_CONSTANTS` + `SPOSCAR` (committed) | runs `phonopy_unfold` on the 3x3x3 fcc Cu example |
| `fig_siesta_si.py` | `si_unfolded.png` | `tests/data/si_example/` (committed HSX fixtures, story 010) | 8-atom conventional-cell Si unfolded onto the primitive path; requires HamiltonIO + sisl |
| *DDB example* | `cu_fcc_unfolded.png`, `catio3_unfolded.png` | `examples/Cu_fcc/out_DDB`, `examples/CaTiO3_unfold/out.DDB` (committed) | regeneration requires abipy + anaddb (not part of CI); the committed images in `examples/*/unfolded.png` are the authoritative copies |
| `fig_wannier_sto.py` | `wannier_sto_unfolded.png`, `wannier_sto_ti_vacancy.png` | `examples/wannier_STO/data_nodefect/`, `data/` (committed real Wannier90 outputs) | pristine + Ti-vacancy SrTiO3 supercells unfolded via the built-in Wannier90 reader; no args regenerates both |
| `fig_srmmo3_magnon.py` | `srmmo3_magnon_unfolded.png` | committed SrMnO3 VASP/Wannier TB2J pickle | G-AFM magnon unfolding onto the pseudo-cubic cell |
| `fig_kcuf3_model.py` | `tb2j_kcuf3_model_unfolded.png` | `examples/tb2j-kcuf3-model/run.py` | model-only type-A magnon figure; uniform and illustrative flavor-asymmetric exchanges |

Usage:

```bash
python docgen/fig_phonopy_cu.py [out.png]
python docgen/fig_siesta_si.py  [out.png]
python docgen/fig_kcuf3_model.py [out.png]
```
