---
title: "Unfolding magnons"
weight: 20
---

How to unfold TB2J magnon bands computed in a cell larger than the
magnetic primitive cell — magnetic supercells (AFM order) and
slab/ribbon models with open boundaries. For the shared run parameters,
k-path conventions, and frames see the
[method, parameters, and k-path guide](/guide/method-parameters-kpaths/);
for a worked AFM example see
[TB2J magnons SrMnO3](/examples/tb2j-magnon-srmmo3/).

The `magnon` route takes a TB2J results directory (the pickle written by
a TB2J run), builds the BdG magnon Hamiltonian with
`TB2J.magnon.magnon3.Magnon` for a collinear reference, and assigns each
supercell mode a weight on primitive-cell momenta through the unfold
matrix `M` (rows = supercell lattice vectors in primitive units,
`A_sc = M @ A_pc`).

## Slab and ribbon magnon models

Surface and finite-width magnons come from the same machinery: replicate
the primitive TB2J model into a supercell with the TB2J edit API, cut the
periodic image along the open direction(s), then unfold with
`M = diag(N_a, N_b, N_c)` for the replication factors.
```console
# 1) replicate the primitive exchange model 6x along a
TB2J_edit supercell -i TB2J_results --matrix 6 1 1 -o TB2J_sc6
# 2) open the a boundary: drops exactly the pair terms whose R winds
#    through a (strict truncation)
TB2J_edit open -i TB2J_sc6 -d a -o TB2J_ribbon
```

Both steps (and the whole pipeline from a TOML file) are provided by
`TB2J.io_exchange.edit` (`make_supercell`, `open_boundary`,
`run_pipeline`); see `TB2J_edit --help`.

Unfolding then needs no new concepts: the ribbon cell is still the
replicated cell, so the unfold matrix is the plain replication matrix.

```toml
route = "magnon"

[input]
results = "TB2J_ribbon"               # directory holding TB2J.pickle

[structure]
supercell_matrix = [[6, 0, 0], [0, 1, 0], [0, 0, 1]]

[path]
special_points = "GXG"
npts = 200

[output]
output = "ribbon_unfolded.png"
```

What to expect physically:

- the open direction carries no periodicity, so the unfolded bands are
  flat along it — the standing-wave ladder of the finite direction
  appears along the remaining periodic axes;
- mode weights still obey the per-mode sum rule (they live on the
  periodic axes' momenta);
- the reference frame is the same collinear one as the bulk case
  (Q = 0, quantization axis from the pickle).

The regression suite seals this flow on a synthetic two-sided FM chain
(`tests/test_tb2j_ribbon_unfold.py`): the replicated 6×1×1 cell unfolds
back to the primitive band with unit weight on the matching branch, and
the cut ribbon shows zero dispersion along the open axis with intact
weights.

### Gauge note

`magnon3` Fourier-transforms exchange tensors with the cell-origin phase
`exp(-i 2π k·R)`. For real two-sided TB2J pickles (each bond stored in
both directions) the spectra are gauge-equivalent to any other Bloch
convention; the unfolding weights consume this convention and must not
be mixed with position-gauge amplitudes. Synthetic fixtures must store
both bond directions `((R, i, j)` and `(-R, j, i))` or J(k) is not
Hermitian.
