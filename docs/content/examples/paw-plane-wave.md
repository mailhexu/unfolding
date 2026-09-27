---
title: "PAW plane-wave unfolding: ABINIT and VASP"
weight: 14
---

A PAW wavefunction is normalized in the overlap metric, not by the sum
of its pseudo plane-wave coefficient norms. GPAW's own `gpaw/unfold.py`
computes the **pseudo-only** reciprocal-coset ratio; it does not include
PAW augmentation. `unfolding.abinit_paw` and `unfolding.vasp_paw` evaluate
physical overlaps with the PAW metric
`S = I + Σ_aij |p_ai⟩ ΔS_aij ⟨p_aj|`. Only overlap augmentation `ΔS`
enters the spectral projection, not the nonlocal Hamiltonian `D`.

Both routes embed primitive reference wavefunctions into the supercell G
basis, compute `A = B† S C` and `G = B† S B`, then obtain each weight
from the diagonal of `A† G⁻¹ A`. The finite primitive reference band bank
is not complete in general: the sum across folds can differ from one for
a doped system, and close degeneracies require group-projector
eigenvalues rather than arbitrary per-band gauge assignments.

## ABINIT JTH PAW: real Si and Si:P

`HamiltonIO.abinit.read_paw_wfk` reads ABINIT 9/10 netCDF WFKs and JTH PAW
XML datasets (via `pypao`); the WFK stores pseudo coefficients but **not**
projector overlaps. `unfolding.abinit_paw.unfold_abinit_paw` evaluates the
reciprocal projectors (via `abinao`), atom-site overlap correction, and
reference-band weights. The ordinary `unfold_abinit` rejects PAW WFKs
rather than silently treating them as norm-conserving.

The committed inputs and small WFKs in `tests/data/abinit_paw` reproduce
8-atom pristine Si, 8-atom Si7P, and a 2-atom primitive reference at the
four momenta folding to supercell Gamma. Run ABINIT on each `.abi` input
from that directory to regenerate; `iomode 3`, `istwfk 1`, `ecut 10 Ha`,
`pawecutdg 20 Ha` and the bundled Si/P JTH XMLs are required.

```python
from pathlib import Path
from HamiltonIO.abinit import HARTREE_TO_EV
from unfolding.abinit_paw import unfold_abinit_paw
base = Path('tests/data/abinit_paw')
xml = {symbol: base / f'{symbol}.xml' for symbol in ('Si', 'P')}
M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
result = unfold_abinit_paw(base/'si7p_gammao_WFK.nc',
                           base/'si_primitive_foldso_WFK.nc', xml, M,
                           resolve_degenerate=1e-3 / HARTREE_TO_EV)
```

ABINIT stores the supercell-Gamma degenerate manifolds in an arbitrary
unitary gauge, which scrambles raw per-band weight diagonals (up to the
full weight on the wrong fold). Passing `resolve_degenerate` eigen-assigns
the reference projector and the pseudo-coset operator inside each
degenerate group, giving gauge-invariant branch weights; without it the
returned diagonals are only meaningful for non-degenerate bands.

The real PAW Si8 WFK has maximum `|⟨ψ̃|S|ψ̃⟩−1|` of 5.8×10⁻⁷. Its raw
fold weights sum to one within 2×10⁻⁶ per band; after resolving
near-degenerate groups (`resolve_degenerate=1e-5` Hartree), branch weights
are within 2×10⁻⁶ of zero or one. For Si7P, the maximum PAW-minus-pseudo
weight difference is 0.015, and the PAW normalization error is 1.4×10⁻⁶.

{{< figure src="/images/abinit_paw_si7p.png" title="ABINIT Si:P at the four primitive folds of supercell Gamma: PAW-metric reference weights (left) and pseudo-only coset fractions (right). Marker area encodes weight." >}}

Run `python examples/abinit_paw/unfold.py` to redraw the figure.

### Dense GXWGLWX path for PAW Si and Si:P

`tests/data/abinit_paw/regenerate_path_on_nic6.sh` stages the bundled
JTH XMLs and three committed decks on nic6 (scratch staging; the decks
run serially and concurrently) and fetches the dataset-2 WFKs. Each
supercell deck mirrors its committed Gamma deck (`ixc 11`, `ecut 10 Ha`,
`pawecutdg 20 Ha`, `nband 32`) with `ndtset 2`: dataset 1 is an SCF at
supercell Gamma, dataset 2 a non-SCF run over the 305-point
Γ-X-W-Γ-L-W-X path. Dataset 2 reuses the validated norm-conserving
deck's kpt list verbatim in supercell coordinates, `K = k_prim @ M.T`;
a 2-atom primitive deck runs the same path in primitive coordinates and
supplies the embedded reference banks. The fetched WFKs are hundreds of
MB and stay external (gitignored); the committed decks plus script
reproduce them.

Because the supercell k-list *is* the path, each stored supercell state
contributes one primitive momentum sector per path point, and
`unfold_abinit_paw` unfolds it unchanged: the primitive reference bands
at `k = K @ M⁻¹` embed geometrically into the stored supercell G sphere
exactly as for the Gamma-only fixtures. The projector-block cache is
bounded, so a 305-point run costs no more memory than a 4-point one.

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    'ex', 'examples/abinit_paw/unfold_path.py')
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
ex.main()   # unfold both path WFKs, print weight statistics, draw the figure
```

`python examples/abinit_paw/unfold_path.py` renders
`docs/static/images/abinit_paw_si_path.png`: two weight-coded panels
(pristine Si8, Si:P) on the house-style path axis with E at the NSCF
Fermi level. The PAW normalization residual stays below 3×10⁻⁶ along the
path. Pristine Si is binary below 6 eV wherever bands are non-degenerate
(the top of the 32-band manifold, above ~7 eV, is the NSCF Davidson tail
and not weight-converged). Si:P mixes primitive sectors: the donor window
(-1.5 to 2.5 eV) carries hundreds of fractional single-sector weights
(median ≈ 0.36, consistent with the donor spreading over the four
sectors), and sector states anticross along the path, so isolated bands
go fractional at crossings — the dopant fingerprint the Gamma-only
fixture cannot show. Against the norm-conserving route
(`docgen/fig_abinit_si8.py`, ecut 25 vs PAW ecut 10) the matched
high-weight bands agree to 0.16 eV maximum after a +0.07 eV
potential-reference shift.

{{< figure src="/images/abinit_paw_si_path.png" title="ABINIT PAW unfolded GXWGLWX path: pristine Si8 (left) and Si7P (right). Line opacity encodes the PAW S-metric weight on the primitive-sector reference bank; E_F = 0." >}}

`tests/test_abinit_paw.py` pins these numbers (skipped when the external
WFKs are absent): fixture shapes, pristine binarity, the Si:P donor
fractional-weight pattern, and the PAW-vs-NC band agreement.

## VASP PAW: licensed input stays private

`HamiltonIO.vasp.read_wavecar` parses standard full-sphere scalar
WAVECARs through pymatgen; `read_potcar_paw` reads the matching POTCAR's
reciprocal projector tables and AE/pseudo partial waves. The VASP
license prohibits redistributing POTCAR here. Run
`unfolding.vasp_paw.unfold_vasp_paw((WAVECAR, POSCAR),
(primitive_WAVECAR, primitive_POSCAR), POTCAR, M)` with matching datasets.
Gamma-half and noncollinear WAVECAR layouts are deliberately unsupported.

The private bcc-Fe primitive fixture validates the actual PAW metric:
its first-eight-band pseudo coefficient norms range from 0.325 to 1.099,
while the PAW S-norm residual is below 8×10⁻⁶ at the two sampled k-points.
An 11-fold *constructed* Fe supercell embedding of a real primitive
wavefunction verifies the non-diagonal lattice reciprocal mapping and
restores its PAW norm within 2×10⁻⁸; other folds carry <10⁻⁸ weight.
That constructed embedding is not an independent VASP supercell run.

To exercise the licensed fixture locally, set `UNFOLDING_VASP_FE_SEED` to
a private directory with `WAVECAR`, `POSCAR`, `POTCAR`, then run
`python -m pytest tests/test_vasp_paw.py`. No licensed bytes are stored
in this repository.

### VASP PAW dense path: bcc Fe through a 16-fold supercell

`examples/vasp_fe` extends the VASP route to a dense band path, mirroring
the ABINIT/SIESTA style above. `make_inputs.py` writes all inputs: the
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
Si), draws the spin-resolved weighted-band map over [−8, +8] eV (metallic
Fe: d manifold plus sp states around E_F), and overlays the directly
computed primitive-cell bands after E_F alignment. The consistency
report over all 250 path points: PAW S-norm residual ≤ 5.3×10⁻⁵ on real
supercell states, sector weights binary away from degeneracies
(interstitial fraction ≤ 1.3×10⁻³ of window states), and weight-1
branches matching the primitive dispersion to max 0.063 eV / rms
0.009 eV (spin up) and 0.019 / 0.003 eV (spin down) over ~1438 branches;
the raw E_F gap between the 16-atom and 1-atom SCF references is −31 meV
and is removed by the alignment.

{{< figure src="/images/vasp_fe_path.png" title="VASP PAW unfolding of a 2×2×2 conventional bcc-Fe supercell (16 atoms, det 16) along the primitive Γ-H-N-Γ-P-H path (250 points): spin-resolved effective band map with the directly computed primitive-cell bands overlaid (crimson). Marker-free white regions carry no fold of the path momentum; the window is E_F ± 8 eV." >}}

Run `python examples/vasp_fe/unfold_fe.py` with `UNFOLDING_VASP_FE_SEED`
(POTCAR directory) and `UNFOLDING_VASP_FE_PATH` (run tree) set to redraw
the figure; `tests/test_vasp_paw.py` pins the sector binarity, PAW norm
restoration and primitive agreement on a path subsample (skipped when
the external WAVECARs are absent). The supercell NSCF WAVECAR (~0.9 GB)
stays outside the repository.
