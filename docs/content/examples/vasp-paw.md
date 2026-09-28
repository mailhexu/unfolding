---
title: "VASP PAW plane-wave unfolding: bcc Fe"
weight: 15
---

Unfold a VASP PAW calculation — spin-polarized bcc Fe in a 2×2×2
conventional supercell (16 atoms) — onto the 1-atom bcc primitive cell
with the `unfold_vasp_paw` adapter, from `WAVECAR` + `POSCAR` + `POTCAR`.
As in the ABINIT PAW route, overlaps use the PAW metric
`S = I + Σ_aij |p_ai⟩ ΔS_aij ⟨p_aj|` built from the POTCAR projector
tables; primitive reference wavefunctions are embedded into the supercell
G basis (`A = B† S C`, `G = B† S B`, weight = diagonal of `A† G⁻¹ A`).

## Running the example

Generate all inputs (primitive bcc cell a = 2.866 Å and its 2×2×2
conventional supercell via `M = 2*[[0,1,1],[1,0,1],[1,1,0]]`, det 16;
250-point primitive path Γ-H-N-Γ-P-H from `ase.dft.kpoints.bandpath`,
mapped as `K_sc = k_prim @ M.T`; spin-polarized SCF charge runs and
`ICHARG=11` NSCF path runs with `NBANDS = 96` in both banks):

```bash
python examples/vasp_fe/make_inputs.py
```

Copy a private PAW_PBE Fe POTCAR into each run directory and run
`vasp_std` (the POTCAR cannot be redistributed, so no licensed bytes are
stored in this repository). Then unfold both spins and draw the figure:

```bash
UNFOLDING_VASP_FE_SEED=/path/to/POTCAR_dir \
UNFOLDING_VASP_FE_RUNS=/tmp/unfolding_vasp_fe \
python examples/vasp_fe/unfold_fe.py
```

To exercise the licensed fixture in the test-suite route instead, set
`UNFOLDING_VASP_FE_SEED` to a private directory with `WAVECAR`, `POSCAR`,
`POTCAR` and run `python -m pytest tests/test_vasp_paw.py`.

{{< figure src="/images/vasp_fe_path.png" title="VASP PAW unfolding of the 2×2×2 conventional bcc-Fe supercell (16 atoms, det 16) along the primitive Γ-H-N-Γ-P-H path (250 points), spin-up (left) and spin-down (right): Gaussian-smeared spectral-weight map (Blues scale) with the directly computed primitive-cell bands overlaid in crimson; energies relative to each run's own SCF Fermi level (E_F = 0), window E_F ± 8 eV" >}}

`HamiltonIO.vasp.read_wavecar` parses standard full-sphere scalar
WAVECARs through pymatgen; `read_potcar_paw` reads the matching POTCAR's
reciprocal projector tables and AE/pseudo partial waves. Gamma-half and
noncollinear WAVECAR layouts are unsupported by design.

## Calculation background

- Code: VASP, PAW_PBE Fe POTCAR (private), `ISPIN=2`, `ENCUT 300` eV,
  Gaussian smearing 0.05 eV (the settings of the validated projector
  seed); `ICHARG=11` NSCF over the 250-point path; `NBANDS = 96`
  (≥ 82 occupied majority-spin bands plus margin) — supercell and
  primitive banks must share NBANDS so resolved weights pair with
  supercell bands index-by-index. ~10 irreducible k-points converge the
  supercell SCF.
- KPAR facts for the path NSCF: `KPAR = 8` is the fast, stable choice on
  this build; `KPAR = 16` writer is unstable, and any `KPAR` switches the
  WAVECAR to the single-precision 45200 layout with unwrapped
  k-coordinates.
- Reader handling in `examples/vasp_fe`: `wrap_wavecar.py` rewraps stored
  k-points so pymatgen accepts the file; `read_wavecar_ordered.py`
  enumerates each G list in VASP's stored (unwrapped) k frame with the
  fold-lexicographic order VASP writes coefficients in (pymatgen's
  wrapped-k assumption permutes the coefficient↔G assignment for general
  k). Both spins are unfolded with `resolve_degenerate=1e-3` eV, as in
  the Si examples.
- Run tree: `/tmp/unfolding_vasp_fe` (`prim_scf`, `prim_nscf`,
  `sc16_scf`, `sc16_nscf`); the supercell NSCF WAVECAR (~0.9 GB) stays
  outside the repository. `tests/test_vasp_paw.py` pins fixture-level
  checks on a path subsample (skipped when the external WAVECARs are
  absent).
- Figure: `python examples/vasp_fe/unfold_fe.py` writes
  `docs/static/images/vasp_fe_path.png`.
