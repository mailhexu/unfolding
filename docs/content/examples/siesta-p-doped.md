---
title: "SIESTA Si:P dopant"
weight: 4
---

# SIESTA Si:P dopant

Unfolding earns its keep on defects. Replace one Si atom by P in the
supercell and the unfolding weight measures how much each supercell state
still resembles the ideal crystal.

## What you need

The same ingredients as the [pristine Si example](../siesta-si/), but from
a supercell in which one atom is substituted (geometry otherwise relaxed
as you see fit). No code changes are needed: the weight kernel reads the
position-resolved overlap, so substituted or displaced atoms carry their
local physics automatically.

```python
ax = unfold_siesta(
    fdf='si_p_sc.fdf',                 # one Si replaced by P
    prim_atoms=prim_atoms,
    unfold_sc_mat=unfold_sc_mat,
    kpts=kpts, xqpts=x, Xqpts=X, knames=knames,
)
```

## Reading the figure

- Host bands — states essentially unchanged from ideal Si — keep weight 1.
- Donor/impurity-derived states, which have no counterpart in the primitive
  crystal, appear at reduced weight spread over several fold sectors.
- The weight of every state is still normalized: summing one band's weight
  over the complete unfolding grid gives 1.

{{< figure src="/images/si_p_doped_unfolded.png" title="Si:P: host bands stay at weight 1 while impurity-derived states drop below it" >}}

## Relabel maps with substituted species

When you drive the building blocks directly, atom matching defaults to
same-species matching; for a substituted site use

```python
rm = RelabelMap.from_atoms(sc_atoms, prim_atoms, unfold_sc_mat,
                           match_species=False)
```

so the dopant maps onto the host site it replaces.
