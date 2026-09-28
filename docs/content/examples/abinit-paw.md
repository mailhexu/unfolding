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
augmentation `ΔS` enters the projection); pass `resolve_degenerate`
(in eV) so weights inside stored degenerate blocks are eigen-assigned.

## Running the example

Download the [abinit-paw bundle](/downloads/abinit-paw.tar.gz), unpack
it, and work from the unpacked directory:

```bash
tar xzf abinit-paw.tar.gz && cd abinit-paw
python reproduce.py             # -> abinit_paw_si7p_gamma.png
```

The bundle ships the committed Γ-point WFKs (Si8, Si7P and the 2-atom
primitive reference) and the JTH PAW XML datasets, so the Γ-fold figure
needs no ABINIT run — only the `unfolding` package importable (`pip
install -e <unfolding repo checkout>`) with numpy, matplotlib, netCDF4
and pypao. The dense Γ-X-W-Γ-L-W-X path figure additionally needs the
path WFKs (~250 MB each, not shippable): regenerate them from the
bundled decks with your own ABINIT >= 10 (`bash regenerate_path.sh`
next to the XMLs) and follow the path recipe below.

The same Γ computation through the unified CLI:

```bash
unfolding --config unfold.toml          # the bundled config
# or, with explicit flags:
unfolding abinit-paw --supercell data/si7p_gammao_WFK.nc \
  --primitive data/si_primitive_foldso_WFK.nc --paw data \
  --unfold-mat -1 1 1 1 -1 1 1 1 -1 \
  --names 'Γ' '(0,½,½)' '(½,0,½)' '(½,½,0)' --xticks 0 1 2 3 \
  --resolve-degenerate 0.001 --output abinit_paw_si7p_gamma.png
```

```python
# the same TOML through the Python entry points
from unfolding import load_config, run

run(load_config("unfold.toml"))
```

and from Python:

```python
from pathlib import Path
from HamiltonIO.abinit import HARTREE_TO_EV
from unfolding.abinit_paw import unfold_abinit_paw

data = Path('data')
xml = {symbol: data / f'{symbol}.xml' for symbol in ('Si', 'P')}
M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
result = unfold_abinit_paw(data/'si7p_gammao_WFK.nc',
                           data/'si_primitive_foldso_WFK.nc', xml, M,
                           resolve_degenerate=1e-3 / HARTREE_TO_EV)
```

For the dense-path figure, place the regenerated
`si8_paw_patho_DS2_WFK.nc`, `si7p_paw_patho_DS2_WFK.nc` and
`si_prim_paw_patho_DS2_WFK.nc` in `data/`, then unfold each supercell
WFK against the primitive one exactly as in `reproduce.py` (the decks
already sample the 305-point path in supercell coordinates
`K = k_prim @ M.T`); energies are referenced to each NSCF run's Fermi
level and the top of the 32-band manifold above ~6 eV sits in the
unconverged Davidson tail (trust energies below that).

{{< figure src="/images/abinit_paw_si_path.png" title="ABINIT PAW unfolding of the pristine Si8 (left) and Si:P (right) supercells onto the primitive Γ-X-W-Γ-L-W-X path: line opacity encodes the PAW S-metric spectral weight carried on the primitive-sector reference bank; energies relative to the NSCF Fermi level (E_F = 0)" >}}

`HamiltonIO.abinit.read_paw_wfk` reads ABINIT 9/10 netCDF WFKs and JTH
PAW XML datasets (via `pypao`); the ordinary
[norm-conserving route](/examples/abinit-wfk/) (`unfold_abinit`)
rejects PAW WFKs rather than treating them as norm-conserving.

## Structures

- **Primitive cell**: 2-atom fcc, `acell 3*10.26` bohr,
  `rprim 1 0 0  0 1 0  0 0 1`, Si at xred (0,0,0) and (¼,¼,¼).
- **Supercell**: 8-atom conventional cubic cell, supercell = **M @
  primitive** with `M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]`; the
  Si7P fixture replaces the (¼,¼,¼) atom by P (znucl 15).
- **Atomic data**: JTH PAW datasets (Psdj_paw_pbe_std) `Si.xml` /
  `P.xml` — they ship with the bundle and are consumed through pypao.
- **Run settings**: `ixc 11`, `ecut 10` Ha, `pawecutdg 20` Ha,
  `nband 32`, `occopt 3`, `tsmear 0.005`, `iomode 3` (netCDF WFK),
  `istwfk 1` at Γ, `chkprim 0` (the supercell is not primitive).

## K-path

Γ-fold figure: the four primitive momenta folding to supercell Γ —
Γ, (0,½,½), (½,0,½), (½,½,0) in **primitive reciprocal fractional
coordinates** (the weights live at the primitive WFK's stored
k-points).

Dense-path figure: Γ–X–W–Γ–L–W–X, 305 points; fcc special points in
primitive reciprocal fractional coordinates (Setyawan–Curtarolo); the
supercell decks sample the same path in supercell coordinates
`K = k_prim @ M.T` as an explicit `kpt` list (`kptopt 0`, `ndtset 2`:
dataset 1 SCF at supercell Γ writing the density, dataset 2
frozen-density non-SCF with `iscf -2`, `getden 1`, `tolwfr 1e-16`).
Energies in eV relative to the NSCF Fermi level; window −13…8 eV.

## Parameters

| Parameter | Meaning |
|---|---|
| `supercell` / `primitive` | netCDF WFK paths (or parsed `AbinitPawData`); the primitive reference may be pristine Si while the supercell carries the dopant |
| `datasets` / `paw` | maps chemical symbols to the matching JTH XML paths |
| `matrix` | `M`, row convention supercell = M @ primitive; verified against the two WFK `rprimd` on entry |
| `resolve_degenerate` | eigen-assigns gauge-invariant branch weights inside degenerate groups (ABINIT stores degenerate states in an arbitrary unitary gauge); pass 1e-3 eV, converted to Hartree internally |
| `pseudo_weights` | result attribute: the pseudo-coset fractions, resolved with the same degenerate blocks |
| `spin` | collinear channel (default 0) |

## Calculation background

All decks ship in `inputs/`: the three Γ decks that produced the
committed WFKs and the three dense-path decks for the regeneration.
Both WFK banks need compatible spin channels and commensurate cells;
the projector-block cache is bounded, so a 305-point run costs no more
memory than a 4-point one. In the doped system the impurity couples
sectors at band crossings, so isolated weights go fractional there
while whole valence regions stay binary; pristine sectors never mix
(max non-degenerate |w(1−w)| ~1e-6 below the Davidson tail).
