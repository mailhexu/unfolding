---
title: "SIESTA spinors"
weight: 5
---

Non-collinear (`nspin=4`, spinor) SIESTA supercell runs unfold through
the same machinery as the collinear example: the parse → relabel →
weight chain with doubled orbital counts (each PAO times two spin
components), here on the 8-atom conventional-cubic Si supercell onto the
2-atom primitive fcc cell.

## Running the example

A non-collinear supercell run with the Hamiltonian saved, plus the
primitive structure. Spin-polarized collinear runs are even simpler —
see `spin="up"|"down"` in the [Si example](../siesta-si/).

```python
from unfolding.lcao_unfolder import HamiltonIOModelSpinor, LCAOUnfolderSpinor
from unfolding.mapping import RelabelMapSpinor

rm = RelabelMapSpinor.from_atoms(sc_atoms, prim_atoms, unfold_sc_mat)
unf = LCAOUnfolderSpinor(HamiltonIOModelSpinor(sc_model), rm)
result = unf.compute(kpts, method="ideal")
```

Orbital counts are doubled: with 4 PAOs per Si atom, pass `orb_counts_sc`
of `8` per supercell atom (or let them be inferred from the spinor
model).

{{< figure src="/images/si_spinor_unfolded.png" title="Unfolded SIESTA spinor (nspin=4) Si$_8$ bands along Γ-X-W-Γ-L-X; blue color intensity encodes the unfolding weight, red curves are the independently computed primitive-cell spinor bands; energies in eV with zero at the Fermi level" >}}

A complete input bundle is available as
[siesta-spinor.tar.gz](/downloads/siesta-spinor.tar.gz): input files,
pseudopotentials, the fixture data needed for the figure, a
`reproduce.py` script, and a `README.txt` with prerequisites and exact
run instructions.

## Calculation background

The committed fixtures are `tests/data/si_example/si_sc_pso.fdf` +
`si_sc_pso.HSX` (supercell) and `si_prim_pso.fdf` + `si_prim_pso.HSX`
(primitive cell): the same 8-atom Si setup as the
[Si example](../siesta-si/) run non-collinear (`Spin.Orbit` /
`nspin=4`), with a scalar Si pseudopotential without SOC channels, so
spin-orbit coupling is off. Spinor orbital counts are 8 per atom (4 PAO
× 2 spin components); the supercell matrix and Γ-X-W-Γ-L-X path are
identical to the collinear example. Regenerate the figure headless with
`python docgen/fig_siesta_spinor.py`.
