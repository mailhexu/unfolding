---
title: "SIESTA spinors"
weight: 5
---

Non-collinear (spinor, `nspin=4`) supercell runs unfold through the same
machinery: the spinor pipeline is the parse → relabel → weight chain with
doubled orbital counts (each PAO times two spin components).

## What you need

A non-collinear supercell run with the Hamiltonian saved, plus the
primitive structure. Spin-polarized collinear runs are even simpler — see
`spin="up"|"down"` in the [Si example](../siesta-si/).

## Run the unfolding

```python
from unfolding.lcao_unfolder import HamiltonIOModelSpinor, LCAOUnfolderSpinor
from unfolding.mapping import RelabelMapSpinor

rm = RelabelMapSpinor.from_atoms(sc_atoms, prim_atoms, unfold_sc_mat)
unf = LCAOUnfolderSpinor(HamiltonIOModelSpinor(sc_model), rm)
result = unf.compute(kpts, method="ideal")
```

Orbital counts are doubled: with 4 PAOs per Si atom, pass `orb_counts_sc`
of `8` per supercell atom (or let them be inferred from the spinor model).

## Reading the figure

With spin-orbit coupling off, spinor bands equal the scalar bands with
Kramers degeneracy — a clean certification that the spinor bookkeeping
(atom matching, doubled orbital indices, spin-traced weights) is correct.
With SOC on, the same pipeline yields the spinor unfolded spectrum.

{{< figure src="/images/si_spinor_unfolded.png" title="Spinor (nspin=4) run: the spinor pipeline reproduces the primitive bands with Kramers degeneracy" >}}

## Reproduce this example

Download the [complete input bundle](/downloads/siesta-spinor.tar.gz)
(`siesta-spinor.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
