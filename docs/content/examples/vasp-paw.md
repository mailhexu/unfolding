---
title: "VASP PAW plane-wave unfolding: bcc Fe"
weight: 15
---

Unfold a VASP PAW calculation — spin-polarized bcc Fe in a 2×2×2
conventional supercell (16 atoms) — onto the 1-atom bcc primitive cell
with the `unfold_vasp_paw` adapter, from `WAVECAR` + `POSCAR` + `POTCAR`.
As in the [ABINIT PAW route](/examples/abinit-paw/), overlaps use the
PAW metric `S = I + Σ_aij |p_ai⟩ ΔS_aij ⟨p_aj|` built from the POTCAR
projector tables; primitive reference wavefunctions are embedded into
the supercell G basis (`A = B† S C`, `G = B† S B`, weight = diagonal of
`A† G⁻¹ A`).

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/); this page only covers what is specific to this example.
Primitive cell input: required for the `vasp-paw` route — it provides the primitive-cell wavefunctions the supercell is projected onto.

## Configuration

```toml
# VASP PAW bcc Fe: the 2x2x2 conventional supercell (16 atoms) unfolded
# onto the 1-atom bcc primitive cell, spin-up channel. WAVECARs and the
# POTCAR are user-supplied (never redistributed): produce them with
# make_inputs.py + VASP as described in README.txt.
route = "vasp-paw"

[input]
supercell = "runs/sc16_nscf/WAVECAR"
supercell_poscar = "runs/sc16_nscf/POSCAR"
primitive = "runs/prim_nscf/WAVECAR"
primitive_poscar = "runs/prim_nscf/POSCAR"
# YOUR licensed PAW_PBE Fe POTCAR; never shipped
potcar = "POTCAR"

[structure]
# 2x2x2 conventional bcc cell = M @ primitive, det 16
supercell_matrix = [[0, 2, 2], [2, 0, 2], [2, 2, 0]]

[path]
# tick labels along the stored Gamma-H-N-Gamma-P-H path
# (xticks required with names)
names = ["Γ", "H", "N", "Γ", "P", "H"]
xticks = [0.0, 2.1923186696, 3.7425220675, 5.2927254653, 7.1913291264, 9.0899327875]

[options]
spin = 0
resolve_degenerate = 0.001

[output]
output = "vasp_fe_spin_up.png"
```

## Running the example

Download the [vasp-fe bundle](/downloads/vasp-fe.tar.gz), unpack it,
and work from the unpacked directory. The VASP `WAVECAR`s and the
`POTCAR` can never be redistributed — the POTCAR is licensed and the
fixture WAVECARs are ~1 GB — so the bundle has two reproduction levels:

```bash
tar xzf vasp-fe.tar.gz && cd vasp-fe

# 1. Snapshot (no VASP, out of the box): redraw the committed figure
#    from the shipped unfolding output in data/vasp_fe_path_result.npz
python reproduce.py --snapshot            # -> vasp_fe_path.png

# 2. Full pipeline with your own VASP run
mkdir -p runs && cp -r inputs/* runs/
cd runs/sc16_scf  && cp /path/to/POTCAR . && vasp_std && cd ../..
cd runs/prim_scf  && cp /path/to/POTCAR . && vasp_std && cd ../..
cd runs/sc16_nscf && cp /path/to/POTCAR . && vasp_std && cd ../..
cd runs/prim_nscf && cp /path/to/POTCAR . && vasp_std && cd ../..
cp /path/to/POTCAR POTCAR
python reproduce.py --potcar POTCAR       # -> vasp_fe_path.png
```

Prerequisites: the `unfolding` package importable (`pip install -e
<unfolding repo checkout>`) with numpy and matplotlib; the full
pipeline additionally needs pymatgen and VASP with a PAW_PBE Fe POTCAR
(06Sep2000).

The same spin-up computation through the unified CLI (after the VASP
runs above):

```bash
unfolding --config unfold.toml          # the bundled config
# or, with explicit flags:
unfolding vasp-paw --supercell runs/sc16_nscf/WAVECAR \
  --supercell-poscar runs/sc16_nscf/POSCAR \
  --primitive runs/prim_nscf/WAVECAR \
  --primitive-poscar runs/prim_nscf/POSCAR --potcar POTCAR \
  --unfold-mat 0 2 2 2 0 2 2 2 0 \
  --names 'Γ' H N 'Γ' P H \
  --xticks 0 2.1923186696 3.7425220675 5.2927254653 7.1913291264 9.0899327875 \
  --spin 0 --resolve-degenerate 0.001 --output vasp_fe_spin_up.png
```

```python
# the same TOML through the Python entry points (after the VASP runs:
# the TOML references the user-supplied WAVECARs and POTCAR)
from unfolding import load_config, run

run(load_config("unfold.toml"))
```

and from Python:

```python
from make_inputs import MATRIX
from unfolding.vasp_paw import read_wavecar_ordered, unfold_vasp_paw

sc = read_wavecar_ordered("runs/sc16_nscf/WAVECAR", "runs/sc16_nscf/POSCAR")
prim = read_wavecar_ordered("runs/prim_nscf/WAVECAR", "runs/prim_nscf/POSCAR")
res = unfold_vasp_paw(sc, prim, "POTCAR", MATRIX, spin=0,
                      resolve_degenerate=1e-3)
```

{{< figure src="/images/vasp_fe_path.png" title="VASP PAW unfolding of the 2×2×2 conventional bcc-Fe supercell (16 atoms, det 16) along the primitive Γ-H-N-Γ-P-H path (250 points), spin-up (left) and spin-down (right): Gaussian-smeared spectral-weight map (Blues scale) with the directly computed primitive-cell bands overlaid in crimson; energies relative to each run's own SCF Fermi level (E_F = 0), window E_F ± 8 eV" >}}

`HamiltonIO.vasp.read_wavecar` parses standard full-sphere scalar
WAVECARs through pymatgen; `read_potcar_paw` reads the matching POTCAR's
reciprocal projector tables and AE/pseudo partial waves. Gamma-half and
noncollinear WAVECAR layouts are unsupported by design.

## Structures

- **Primitive cell**: 1-atom bcc, a = 2.866 Å —
  `PRIM = [[-1,1,1],[1,-1,1],[1,1,-1]] * a/2`.
- **Supercell**: 2×2×2 conventional bcc cell (16 atoms), with
  `M = 2·[[0,1,1],[1,0,1],[1,1,0]]` (det 16).
- **Run settings**: `ISPIN=2`, `ENCUT 300` eV, Gaussian smearing 0.05 eV
  (`ISMEAR 0`), `MAGMOM = 16*2.2`, `NBANDS = 96` in **both** banks
  (≥ 82 occupied majority bands plus margin; supercell and primitive
  banks must share NBANDS so resolved weights pair index-by-index).
  SCF charge runs: `ICHARG=2` on 4×4×4 (supercell) / 12×12×12
  (primitive) Γ-centered meshes; path NSCF: `ICHARG=11`, `ISYM=-1`.

## K-path

Γ–H–N–Γ–P–H, 250 points from `ase.dft.kpoints.bandpath` on the
primitive bcc cell; the supercell deck lists the same 250 points in an
explicit `KPOINTS` list (this VASP build requires the weight column).
Weights live at the stored supercell k-points. Window E_F ± 8 eV.

## Parameters

| Parameter | Meaning |
|---|---|
| `supercell` / `primitive` | `(WAVECAR, POSCAR)` pairs; the primitive reference may be pristine bcc Fe while the supercell is anything commensurate |
| `potcar` | the licensed matching POTCAR; `read_potcar_paw` pulls its reciprocal projector tables. Read, never persisted or redistributed |

## Calculation background

- ~10 irreducible k-points converge the supercell SCF; the primitive
  bank needs `ALGO = Exact` (blocked Davidson hits ZHEGV failures in
  the small ~140-plane-wave basis at NBANDS 96).
- **KPAR facts for the path NSCF**: `KPAR = 8` is the fast, stable
  choice on this build; `KPAR = 16` crashes it, and any `KPAR` switches
  the WAVECAR to the single-precision 45200 layout with unwrapped
  k-coordinates.
- Reader handling: `read_wavecar_ordered.py` (shipped in the bundle)
  enumerates each G list in VASP's stored (unwrapped) k frame with the
  fold-lexicographic order VASP writes coefficients in — pymatgen's
  wrapped-k assumption permutes the coefficient↔G assignment for
  general k, which silently corrupts augmentation overlaps on
  high-energy bands. `wrap_wavecar.py` is the alternative fix when
  reading through pymatgen directly.
- Consistency checks printed by the full pipeline: PAW norm residuals
  (~1e-6), sector binarity of the unfolded weights, and the maximum
  supercell-minus-primitive branch deviation on weight-1 branches after
  E_F alignment.
