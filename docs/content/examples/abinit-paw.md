---
title: "ABINIT PAW plane-wave unfolding: Si and Si:P"
weight: 14
---

Unfold ABINIT PAW plane-wave calculations with the
`unfold_abinit_paw` adapter: 8-atom pristine Si and 8-atom Si:P
supercells onto the 2-atom primitive fcc cell, with a separately
computed 2-atom primitive WFK supplying the reference bands. The PAW
wavefunction is normalized in the overlap metric, so overlaps are
evaluated with `S = I + Σ_aij |p_ai⟩ ΔS_aij ⟨p_aj|` (only the overlap
augmentation `ΔS` enters the projection); pass `resolve_degenerate=1e-3
/ HARTREE_TO_EV` so weights inside stored degenerate blocks are
eigen-assigned.

## Running the example

Gamma-point fixtures (committed inputs and small WFKs in
`tests/data/abinit_paw`):

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

or `python examples/abinit_paw/unfold.py`. For the dense
Γ-X-W-Γ-L-W-X path figure, stage the path WFKs with
`tests/data/abinit_paw/regenerate_path_on_nic6.sh`, then:

```bash
python examples/abinit_paw/unfold_path.py
```

{{< figure src="/images/abinit_paw_si_path.png" title="ABINIT PAW unfolding of the pristine Si8 (left) and Si:P (right) supercells onto the primitive Γ-X-W-Γ-L-W-X path: line opacity encodes the PAW S-metric spectral weight carried on the primitive-sector reference bank; energies relative to the NSCF Fermi level (E_F = 0)" >}}

`HamiltonIO.abinit.read_paw_wfk` reads ABINIT 9/10 netCDF WFKs and JTH
PAW XML datasets (via `pypao`); the ordinary `unfold_abinit` rejects PAW
WFKs rather than treating them as norm-conserving.

## Calculation background

- Code: ABINIT with JTH PAW datasets (`Si.xml`, `P.xml`, staged by the
  committed decks); `ixc 11`, `ecut 10` Ha, `pawecutdg 20` Ha,
  `nband 32`.
- Supercell matrix `M = [[-1,1,1],[1,-1,1],[1,1,-1]]` (8-atom
  conventional cell = M @ primitive fcc cell); supercell path momenta
  are `K = k_prim @ M.T`.
- Decks (`tests/data/abinit_paw/`): `si8_gamma.abi`, `si7p_gamma.abi`,
  `si_primitive_folds.abi` (Gamma fixtures); `si8_paw_path.abi`,
  `si7p_paw_path.abi`, `si_prim_paw_path.abi` (`ndtset 2`: dataset 1 SCF
  at supercell Γ, dataset 2 frozen-density non-SCF over the 305-point
  Γ-X-W-Γ-L-W-X path; the primitive deck runs the same path in primitive
  coordinates and supplies the embedded reference banks).
- The fetched path WFKs are hundreds of MB and stay gitignored;
  `regenerate_path_on_nic6.sh` stages the XMLs and decks and fetches the
  dataset-2 WFKs. Because the supercell k-list *is* the path, each stored
  supercell state contributes one primitive momentum sector per path
  point; the projector-block cache is bounded, so a 305-point run costs
  no more memory than a 4-point one.
- The top of the 32-band manifold (above ~6 eV) sits at the tail of the
  NSCF Davidson subspace; `unfold_path.py` trusts energies below
  `CONVERGED_EV = 6.0` eV while plotting the house-style −13…8 eV window.
- Figure: `python examples/abinit_paw/unfold_path.py` writes
  `docs/static/images/abinit_paw_si_path.png`. `tests/test_abinit_paw.py`
  pins the fixture-level checks (skipped when the external WFKs are
  absent).
