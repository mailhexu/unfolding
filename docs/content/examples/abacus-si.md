---
title: "ABACUS LCAO Si bands"
weight: 12
---

Unfold an ABACUS LCAO supercell calculation onto the primitive-cell band
path: an 8-atom conventional-cubic Si supercell (plus a Si:P variant)
with the real-space Hamiltonian and overlap exported (`out_mat_hs2 1`
writes `data-HR-sparse_SPIN0.csr` / `data-SR-sparse_SPIN0.csr`),
unfolded onto the 2-atom fcc primitive cell through
`HamiltonIO.abacus.abacus_wrapper.AbacusParser`, exactly like the
SIESTA example.

## Running the example

Download the [abacus-si bundle](/downloads/abacus-si.tar.gz), unpack
it, and work from the unpacked directory:

```bash
tar xzf abacus-si.tar.gz && cd abacus-si
python reproduce.py             # -> abacus_si_unfolded.png + abacus_si_p_doped.png
```

The bundle ships the committed ABACUS outputs (the sparse H/S tables of
all three runs plus the pseudopotentials and numerical orbitals in
`refs/`), so no ABACUS installation is needed — only the `unfolding`
package importable (`pip install -e <unfolding repo checkout>`) with
numpy, matplotlib and HamiltonIO.

The same computation through the unified CLI:

```bash
unfolding --config unfold.toml          # the bundled config
# or, with explicit flags:
unfolding abacus --supercell data/si_conv/OUT.si_conv \
  --primitive data/si_prim/OUT.si_prim \
  --unfold-mat -1 1 1 1 -1 1 1 1 -1 \
  --special-points GXWGLWX --npts 300 \
  --mode lcao --method ideal --output abacus_si_unfolded.png
```

```python
# the same TOML through the Python entry points
from unfolding import load_config, run

run(load_config("unfold.toml"))
```

and from Python:

```python
import numpy as np
from HamiltonIO.abacus.abacus_wrapper import AbacusParser
from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
from unfolding.mapping import RelabelMap

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])   # conv = B @ prim
prim = AbacusParser(outpath='data/si_prim/OUT.si_prim').get_models()
sc = AbacusParser(outpath='data/si_conv/OUT.si_conv').get_models()

rm = RelabelMap.from_atoms(sc.atoms, prim.atoms, B,
                           orb_counts_sc=[13]*8, orb_counts_prim=[13, 13])
unf = LCAOUnfolder(HamiltonIOModel(sc), rm)
res = unf.compute(kpts, method="ideal")
```

For the doped supercell, replace one Si by P in the STRU and pass
`match_species=False` so the dopant folds onto the host site it
replaces (the shipped `si7p` run already has this substitution).

{{< figure src="/images/abacus_si_unfolded.png" title="ABACUS Si$_8$ LCAO unfolded onto the primitive-cell path; blue color intensity encodes spectral weight, red curves are the independently computed primitive-cell bands; energies in eV with zero at the Fermi level parsed from the run's running_scf.log" >}}

{{< figure src="/images/abacus_si_p_doped.png" title="ABACUS Si$_7$P LCAO unfolded along the same path, dopant mapped onto the host site (match_species=False); same encoding, energies referenced to the Si:P run's Fermi level" >}}

## Structures

- **Primitive cell**: 2-atom fcc, a = 5.43 Å (`LATTICE_CONSTANT
  10.2632` bohr for the ABACUS runs) —
  `cell = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]]`.
- **Supercell**: 8-atom conventional cubic cell, supercell = **M @
  primitive** with `M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]`; the
  Si7P fixture substitutes P on the (¼,¼,¼) site.
- **Basis**: DZP numerical orbitals `Si_gga_10au_100Ry_2s2p1d`
  (13 orbitals per Si atom); the Si:P run adds
  `P_gga_10au_100Ry_2s2p1d` for the dopant (also 13 orbitals). ABACUS
  ≤ 3.10 does not write `OUT.${suffix}/Orbital`, so the basis is
  reconstructed from the STRU — the orbital files ship in `refs/`.
- **Run settings**: `basis_type lcao`, `esolver_type ksdft`, PBE,
  `ecutwfc 100` Ry, `symmetry 0`, Γ-centered 2×2×2 k-grid,
  `out_mat_hs2 1`, Gaussian smearing σ = 0.005 Ha.

## K-path

Γ–X–W–Γ–L–W–X, 300 points; fcc special points in **primitive reciprocal
fractional coordinates** (Setyawan–Curtarolo): Γ (0,0,0), X (½,0,½),
W (½,¼,¾), L (½,½,½). Segment point counts are proportional to
Cartesian length. Energies are in eV with **zero at the Fermi level
parsed from the run's `running_scf.log`** (`model.efermi`); the same
shift applies to the overlaid primitive bands; plot window −13…8 eV.

## Parameters

| Parameter | Meaning |
|---|---|
| `supercell` / `primitive` | the `OUT.*` directories of the supercell and primitive-cell runs (sparse tables + logs) |
| `supercell_matrix` | `M`, row convention supercell = M @ primitive |
| `mode` | `lcao` (this example) or `pw` for plane-wave ABACUS runs |
| `spin` | collinear channel (`up`/`down`, default `up`) |
| `method` | `ideal` generic-k spectral weight; `ring` is the exact torus projection at commensurate momenta |
| `match_species` | `false` maps the P dopant onto the host Si site it replaces |
| `orb_counts` | orbitals per atom (13 for DZP 2s2p1d) used by the atom map |

## Calculation background

The parsed model exposes the full unfolding interface: `atoms` (ASE
Atoms of the supercell), `HR`/`SR` dicts keyed by integer lattice
translations (HR in eV), and `hs_and_eigen(k)`. Every weight-1
unfolded branch lies on the independently computed primitive-cell bands
(the crimson overlay), which validates the parser's orbital ordering
and phase conventions. To re-run ABACUS yourself, point
`pseudo_dir`/`orbital_dir` at `refs/` (the shipped INPUT files use
`../refs`) and run `abacus` in each `data/<name>` directory; a
k-sampled SCF is essential — a Γ-only run collapses all image shells
into the R = 0 block, which cannot be unfolded at generic momenta.
