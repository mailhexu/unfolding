---
title: "GPAW LCAO Si and Si:P"
weight: 10
---

Unfold a GPAW LCAO supercell calculation onto the primitive-cell band
path: pristine Si and P-doped Si in the 8-atom conventional cubic cell,
unfolded onto the 2-atom fcc primitive cell through
`HamiltonIO.gpaw.GpawLcaoModel`. GPAW's LCAO matrices already carry every
PAW contribution — the overlap is the projector-augmented one and the
Hamiltonian includes the `dH` projector terms — so the unfolding
backend consumes them exactly like any other atomic-orbital table: no
PAW correction is applied on top.

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/).
Primitive cell input: required for the LCAO `gpaw` route — it supplies the relabel map and the default path cell.

## Configuration

```toml
# GPAW LCAO Si: 8-atom conventional-cubic supercell unfolded onto the
# 2-atom fcc primitive cell along Gamma-X-W-Gamma-L-W-X. The .gpw restarts
# are user-generated (python generate_fixtures.py); see README.txt.
route = "gpaw"

[input]
supercell = "data/si_sc_lcao.gpw"

[structure]
# 2-atom fcc primitive LCAO restart; supercell = M @ primitive
primitive = "data/si_prim_lcao.gpw"
supercell_matrix = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]

[path]
special_points = "GXWGLWX"
npts = 300

[options]
mode = "lcao"
method = "ideal"

[output]
output = "gpaw_si_unfolded.png"
```

## Running the example

Download the [gpaw-si bundle](/downloads/gpaw-si.tar.gz), unpack it,
and work from the unpacked directory:

```bash
tar xzf gpaw-si.tar.gz && cd gpaw-si
python generate_fixtures.py             # ~2-3 h serial, your GPAW
python reproduce.py                     # -> gpaw_si_unfolded.png + _p_doped.png
```

The `.gpw` restarts (~30 MB each) exceed the bundle cap, so the bundle
ships the generator instead: `generate_fixtures.py` reproduces the
committed fixtures exactly with your own GPAW (>= 25; PBE, h = 0.17,
symmetry off) into `data/`. Prerequisites for the figure step: the
`unfolding` package importable (`pip install -e <unfolding repo
checkout>`) with numpy, scipy, matplotlib and HamiltonIO.

The same computation through the unified CLI:

```bash
unfolding --config unfold.toml          # the bundled config
# or, with explicit flags:
unfolding gpaw --supercell data/si_sc_lcao.gpw \
  --primitive data/si_prim_lcao.gpw \
  --unfold-mat -1 1 1 1 -1 1 1 1 -1 \
  --special-points GXWGLWX --npts 300 \
  --mode lcao --method ideal --output gpaw_si_unfolded.png
```

```python
# the same TOML through the Python entry points
from unfolding import load_config, run

run(load_config("unfold.toml"))
```

and from Python, either through the config file or with explicit
adapter objects:

```python
from HamiltonIO.gpaw import GpawLcaoModel
from unfolding.lcao_unfolder import LCAOUnfolder
from unfolding.mapping import RelabelMap
import numpy as np

B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])  # conv cell = B @ prim

prim = GpawLcaoModel.from_file("data/si_prim_lcao.gpw")
sc = GpawLcaoModel.from_file("data/si_sc_lcao.gpw")
rm = RelabelMap.from_atoms(sc.atoms, prim.atoms, B,
                           orb_counts_sc=[4]*8, orb_counts_prim=[4, 4])
result = LCAOUnfolder(sc, rm).compute(kpts, method="ideal")
```

For the doped supercell, substitute one Si by P (same valence row, so P
also carries 4 szp orbitals) and let the dopant site map onto the host
site it replaces with `match_species=False` — the same geometric
correspondence the [SIESTA Si:P example](/examples/siesta-p-doped/)
uses.

{{< figure src="/images/gpaw_si_unfolded.png" title="GPAW LCAO Si$_8$ (8-atom conventional cell) unfolded onto the primitive Γ-X-W-Γ-L-W-X path; blue color intensity encodes spectral weight, crimson curves are the independently computed primitive-cell bands; energies in eV with zero at the unfolded run's Fermi level" >}}

{{< figure src="/images/gpaw_si_p_doped.png" title="GPAW LCAO Si:P unfolded along the same path (dopant mapped onto the host site); same encoding; the crimson primitive reference is drawn on the doped run's Fermi zero" >}}

## Structures

- **Primitive cell**: 2-atom fcc, a = 5.43 Å —
  `cell = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]]`, Si at
  (0,0,0) and (1/4,1/4,1/4).
- **Supercell**: 8-atom conventional cubic cell with
  `M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]`; the
  Si7P fixture substitutes P on the (¼,¼,¼) site.
- **Basis**: GPAW's default szp LCAO set — 4 atomic orbitals per atom
  for Si *and* P. Run settings: PBE, h = 0.17, `symmetry='off'`,
  `mode='all'` restarts (matrices built from stored density and
  wavefunctions).

## K-path and k-grids

Figure path: Γ–X–W–Γ–L–W–X, 300 points on the primitive cell.

SCF k-grids (gamma-centered, required): primitive cell 16×16×16,
supercells 8×8×8. The real-space tables are the inverse lattice Fourier
transform of the k-grid data; a uniform grid cannot disentangle its
+N/2 and −N/2 shells (the half-shifted even grids GPAW writes by
default are not even Hermitian after the transform — the reader rejects
them), so the primitive reference needs 16³ (interpolation error
~1e-11 eV; 4³ leaves ~eV wobble). The supercell 8³ mesh folds exactly
onto the 16³ primitive mesh, so SCF densities — and the eigenvalue
reference — match between the two.

## Parameters

| Parameter | Meaning |
|---|---|
| `supercell` / `primitive` | `mode='all'` LCAO `.gpw` restarts of the supercell and primitive cell |
| `match_species` | `false` maps the P dopant onto the host Si site it replaces |
| `orb_counts` | orbitals per atom for the atom map (4 for szp; fixed by the adapter's parser) |

## Calculation background

`GpawLcaoModel` turns a converged LCAO calculation into the real-space
model interface the unfolder needs: `.atoms` (the supercell),
`.HR`/`.SR` dictionaries keyed by integer supercell-lattice
translations, and `hs_and_eigen(k) -> (H, S)` at any supercell
fractional k-point (HamiltonIO convention 2,
`H(k) = Σ_T H[T] e^{+2πi k·T}`, H in eV). Pristine weights are binary
except where folds of different primitive momenta are exactly
degenerate (mirror-related sectors along X–W / L–W); with the committed
grids every weight-1 branch lies on the primitive reference to a few
meV. The doped system shows the donor-derived states at fractional
weight around E_F.
