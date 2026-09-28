---
title: "SIESTA Si:P dopant"
weight: 4
---

Unfold a substituted SIESTA supercell — one Si atom of the 8-atom
conventional-cubic cell replaced by P — onto the primitive fcc cell band
path with the same `unfold_siesta` adapter as the
[pristine Si example](../siesta-si/).

## Running the example

The ingredients are the same as the pristine example, but from a
supercell in which one atom is substituted (geometry otherwise relaxed
as you see fit). No code changes are needed: the weight kernel reads the
position-resolved overlap.

```python
ax = unfold_siesta(
    fdf='si_p_sc.fdf',                 # one Si replaced by P
    prim_atoms=prim_atoms,
    unfold_sc_mat=unfold_sc_mat,
    kpts=kpts, xqpts=x, Xqpts=X, knames=knames,
)
```

When you drive the building blocks directly, atom matching defaults to
same-species matching; for a substituted site use

```python
rm = RelabelMap.from_atoms(sc_atoms, prim_atoms, unfold_sc_mat,
                           match_species=False)
```

so the dopant maps onto the host site it replaces.

{{< figure src="/images/si_p_doped_unfolded.png" title="Unfolded SIESTA Si$_7$P bands along Γ-X-W-Γ-L-X; blue color intensity encodes the unfolding weight, red curves are the pristine primitive-cell bands; energies in eV with zero at the Fermi level" >}}

A complete input bundle is available as
[siesta-p-doped.tar.gz](/downloads/siesta-p-doped.tar.gz): input files,
pseudopotentials, the fixture data needed for the figure, a
`reproduce.py` script, and a `README.txt` with prerequisites and exact
run instructions.

## Calculation background

The committed fixture is `tests/data/si_example/si_sc_p.fdf` +
`si_sc_p.HSX`: the 8-atom conventional-cubic Si supercell with one Si
replaced by P, otherwise the same SIESTA setup as the pristine example
(SZ PAO basis, GGA-PBE, `MeshCutoff 100 Ry`, 2×2×2 k-grid, a = 5.430 Å).
The dopant site maps onto the host site it replaces via
`match_species=False`; the supercell matrix and path are identical to
the pristine example. Regenerate the figure headless with
`python docgen/fig_siesta_p_doped.py`.
