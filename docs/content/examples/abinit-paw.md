---
title: "ABINIT PAW plane-wave unfolding: Si and Si:P"
weight: 14
---

A PAW wavefunction is normalized in the overlap metric, not by the sum
of its pseudo plane-wave coefficient norms. `unfolding.abinit_paw`
evaluates physical overlaps with the PAW metric
`S = I + Σ_aij |p_ai⟩ ΔS_aij ⟨p_aj|`; only overlap augmentation `ΔS`
enters the spectral projection, not the nonlocal Hamiltonian `D`.
Primitive reference wavefunctions are embedded into the supercell G
basis, `A = B† S C` and `G = B† S B`, and each weight is the diagonal
of `A† G⁻¹ A`.

`HamiltonIO.abinit.read_paw_wfk` reads ABINIT 9/10 netCDF WFKs and JTH
PAW XML datasets (via `pypao`); the WFK stores pseudo coefficients but
**not** projector overlaps. `unfolding.abinit_paw.unfold_abinit_paw`
evaluates the reciprocal projectors (via `abinao`), atom-site overlap
correction, and reference-band weights. The ordinary `unfold_abinit`
rejects PAW WFKs rather than silently treating them as
norm-conserving.

ABINIT stores degenerate eigenspace blocks (e.g. everything at
supercell Gamma) in an arbitrary unitary gauge, which scrambles raw
per-band weight diagonals — up to the full weight landing on the wrong
fold. Passing `resolve_degenerate` eigen-assigns the reference
projector and the pseudo-coset operator inside each degenerate group,
giving gauge-invariant branch weights; without it the returned
diagonals are only meaningful for non-degenerate bands.

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

The committed inputs and small WFKs in `tests/data/abinit_paw`
reproduce 8-atom pristine Si, 8-atom Si7P, and a 2-atom primitive
reference at the four momenta folding to supercell Gamma
(`python examples/abinit_paw/unfold.py`). The real PAW Si8 WFK has
maximum `|⟨ψ̃|S|ψ̃⟩−1|` of 5.8×10⁻⁷; resolved branch weights are
within 2×10⁻⁶ of zero or one; for Si7P the PAW-minus-pseudo weight
difference stays at 0.015.

## Dense GXWGLWX path

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

`python examples/abinit_paw/unfold_path.py` renders
`docs/static/images/abinit_paw_si_path.png`: two weight-coded panels
(pristine Si8, Si:P) on the house-style path axis with E at the NSCF
Fermi level. The PAW normalization residual stays below 3×10⁻⁶ along
the path. Pristine Si is binary below 6 eV wherever bands are
non-degenerate (the top of the 32-band manifold, above ~7 eV, is the
NSCF Davidson tail and not weight-converged). Si:P mixes primitive
sectors: the donor window (-1.5 to 2.5 eV) carries hundreds of
fractional single-sector weights (median ≈ 0.36, consistent with the
donor spreading over the four sectors), and sector states anticross
along the path, so isolated bands go fractional at crossings — the
dopant fingerprint the Gamma-only fixture cannot show. Against the
norm-conserving route (`docgen/fig_abinit_si8.py`, ecut 25 vs PAW ecut
10) the matched high-weight bands agree to 0.16 eV maximum after a
+0.07 eV potential-reference shift.

{{< figure src="/images/abinit_paw_si_path.png" title="ABINIT PAW unfolded GXWGLWX path: pristine Si8 (left) and Si7P (right). Line opacity encodes the PAW S-metric weight on the primitive-sector reference bank; E_F = 0." >}}

`tests/test_abinit_paw.py` pins these numbers (skipped when the
external WFKs are absent): fixture shapes, pristine binarity, the Si:P
donor fractional-weight pattern, and the PAW-vs-NC band agreement.
