---
title: "VASP PAW plane-wave unfolding: bcc Fe"
weight: 15
---

A PAW wavefunction is normalized in the overlap metric, not by the sum
of its pseudo plane-wave coefficient norms (for bcc Fe the first-band
pseudo coefficient norms range 0.33-1.10). `unfolding.vasp_paw`
evaluates physical overlaps with the PAW metric
`S = I + Σ_aij |p_ai⟩ ΔS_aij ⟨p_aj|` built from the POTCAR projector
tables; only overlap augmentation `ΔS` enters the spectral projection,
not the nonlocal Hamiltonian `D`. Primitive reference wavefunctions are
embedded into the supercell G basis, `A = B† S C` and `G = B† S B`,
and each weight is the diagonal of `A† G⁻¹ A`.

`HamiltonIO.vasp.read_wavecar` parses standard full-sphere scalar
WAVECARs through pymatgen; `read_potcar_paw` reads the matching POTCAR's
reciprocal projector tables and AE/pseudo partial waves. The VASP
license prohibits redistributing POTCAR here. Run
`unfolding.vasp_paw.unfold_vasp_paw((WAVECAR, POSCAR),
(primitive_WAVECAR, primitive_POSCAR), POTCAR, M)` with matching
datasets. Gamma-half and noncollinear WAVECAR layouts are deliberately
unsupported. An 11-fold *constructed* Fe supercell embedding of a real
primitive wavefunction verifies the non-diagonal lattice reciprocal
mapping and restores its PAW norm within 2×10⁻⁸.

To exercise the licensed fixture locally, set `UNFOLDING_VASP_FE_SEED`
to a private directory with `WAVECAR`, `POSCAR`, `POTCAR`, then run
`python -m pytest tests/test_vasp_paw.py`. No licensed bytes are stored
in this repository.

## Dense path: bcc Fe through a 16-fold supercell

`examples/vasp_fe` extends the route to a dense band path, mirroring
the ABINIT/SIESTA style. `make_inputs.py` writes all inputs: the
primitive bcc cell (1 atom, a = 2.866 Å) and its 2×2×2 conventional
supercell (16 atoms) via rows-convention
`M = 2*[[0,1,1],[1,0,1],[1,1,0]]` (det 16), a 250-point primitive path
Γ-H-N-Γ-P-H from `ase.dft.kpoints.bandpath` mapped to supercell
coordinates as `K_sc = k_prim @ M.T`, spin-polarized SCF charge runs
(ISPIN=2, ENCUT 300 eV, Gaussian smearing 0.05 eV, the settings of the
validated projector seed), and `ICHARG=11` NSCF runs with `NBANDS = 96`
(≥ 82 occupied majority-spin bands plus margin; supercell and primitive
banks must share NBANDS so resolved weights pair with supercell bands
index-by-index). Copy a private PAW_PBE Fe POTCAR into each run
directory and run `vasp_std`; ~10 irreducible k-points converge the
supercell SCF, the path NSCF is cheapest with `KPAR = 8` (this build's
`KPAR = 16` writer is unstable and `KPAR` switches the WAVECAR to the
single-precision 45200 layout with unwrapped k-coordinates).

Two reader quirks are handled in `examples/vasp_fe`:
`wrap_wavecar.py` rewraps stored k-points so pymatgen accepts the file,
and `read_wavecar_ordered.py` enumerates each G list in VASP's stored
(unwrapped) k frame with the fold-lexicographic order VASP writes
coefficients in — pymatgen's wrapped-k assumption silently permutes the
coefficient↔G assignment for general k, which shows up as PAW S-norm
residuals of ~0.6 instead of ~10⁻⁵.

`unfold_fe.py` unfolds both spins (`resolve_degenerate=1e-3` eV, as in
the Si examples) and draws the spin-resolved weighted-band map over
[−8, +8] eV (metallic Fe: d manifold plus sp states around E_F). The
unfolded map and the crimson primitive-cell overlay are each referenced
to the Fermi level of their **own** SCF run; the two E_F differ by
−31 meV (16-atom vs 1-atom sampling), which is the visible residual
between the overlay and the weight-1 branches. The consistency report
over all 250 path points: PAW S-norm residual ≤ 5.3×10⁻⁵ on real
supercell states, sector weights binary away from degeneracies
(interstitial fraction ≤ 1.3×10⁻³ of window states), and weight-1
branches matching the primitive dispersion to max 0.063 eV / rms
0.009 eV (spin up) and 0.019 / 0.003 eV (spin down) over ~1438 branches
after removing the E_F gap in the numeric comparison.

{{< figure src="/images/vasp_fe_path.png" title="VASP PAW unfolding of a 2×2×2 conventional bcc-Fe supercell (16 atoms, det 16) along the primitive Γ-H-N-Γ-P-H path (250 points): spin-resolved effective band map with the directly computed primitive-cell bands overlaid (crimson), both E_F-referenced. Marker-free white regions carry no fold of the path momentum; the window is E_F ± 8 eV." >}}

Run `python examples/vasp_fe/unfold_fe.py` with `UNFOLDING_VASP_FE_SEED`
(POTCAR directory) and `UNFOLDING_VASP_FE_RUNS` (run tree) set to redraw
the figure; `tests/test_vasp_paw.py` pins the sector binarity, PAW norm
restoration and primitive agreement on a path subsample (skipped when
the external WAVECARs are absent). The supercell NSCF WAVECAR (~0.9 GB)
stays outside the repository.
