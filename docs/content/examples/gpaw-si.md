---
title: "GPAW LCAO Si and Si:P"
weight: 10
---

Unfold a GPAW LCAO supercell calculation onto the primitive-cell band
path. GPAW's LCAO matrices already carry every PAW contribution — the
overlap is the projector-augmented one and the Hamiltonian includes the
``dH`` projector terms — so the unfolding backend consumes them exactly
like any other atomic-orbital table: no PAW correction is applied on top.
Two examples: pristine Si in the 8-atom conventional cubic cell
(validated against the primitive cell), then P-doped Si in the same
cell.

## Reading a GPAW restart

``HamiltonIO.gpaw.GpawLcaoModel`` turns a converged LCAO calculation
(live calculator or ``mode='all'`` ``.gpw`` restart) into the real-space
model interface the unfolder needs: ``.atoms`` (the supercell),
``.HR``/``.SR`` dictionaries keyed by integer supercell-lattice
translations, and ``hs_and_eigen(k) -> (H, S)`` at any supercell
fractional k-point, HamiltonIO convention 2
(``H(k) = sum_T H[T] e^{+2 pi i k.T}``), H in eV.

Two input requirements make the transform well-defined:

| Run setting | Why |
|---|---|
| Gamma-centered k-grid: ``kpts={'size': (n,n,n), 'gamma': True}`` | The real-space tables of GPAW's default half-shifted even grids are not Hermitian (the LCAO matrices carry orbital position phases); the reader rejects such files with this remedy. |
| ``mode='all'`` restart | The Hamiltonian matrices are built from the stored density *and* wavefunctions. |

Internally the tables are the inverse lattice Fourier transform of the
full (`symmetry='off'`) k-grid data. For even grids the Nyquist
translation is split equally between positive and negative images:
sampled matrices reproduce GPAW's LCAO Hamiltonian to 1e-8, while
the generic-k interpolant remains Hermitian. The finite k-grid still
limits the real-space shell range and therefore weight accuracy.

The fixture runs (PBE, default szp LCAO basis — 4 atomic orbitals per
atom for Si *and* P — Gamma-centered 4x4x4 grid) are reproducible with
``examples/gpaw_si/generate_fixtures.py``.

## Pristine Si: a known answer

{{< figure src="/images/gpaw_si_unfolded.png" title="GPAW LCAO Si 8-atom conventional cell unfolded onto the primitive path (blue intensity = spectral weight); crimson curves: independently computed primitive-cell bands" >}}

The supercell matrix is the conventional cubic cell in primitive-lattice
units (row convention ``A_sc = B @ A_prim``):

```python
B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
```

In an infinite, complete basis a pristine state has a single primitive
momentum. The finite szp basis and 4x4x4 real-space archive here yield
near-binary weights away from degeneracies, but degenerate bands can be
arbitrary fold-sector mixtures (the raw 300-point map has weights as
far as 0.5 from binary). The crimson overlay is an independent
primitive-cell calculation. The runnable script is
`examples/gpaw_si/unfold.py`; its core is:

```python
from HamiltonIO.gpaw import GpawLcaoModel
from unfolding.lcao_unfolder import LCAOUnfolder
from unfolding.mapping import RelabelMap

prim = GpawLcaoModel.from_file("si_prim_lcao.gpw")
sc = GpawLcaoModel.from_file("si_sc_lcao.gpw")
rm = RelabelMap.from_atoms(sc.atoms, prim.atoms, B,
                           orb_counts_sc=[4]*8, orb_counts_prim=[4, 4])
weights = LCAOUnfolder(sc, rm).compute(kpts, method="ideal")
```

## Si:P: a defect in the same cell

{{< figure src="/images/gpaw_si_p_doped.png" title="GPAW LCAO Si:P unfolded: impurity-hybridized states redistribute fold weights" >}}

Substituting one Si by P (same valence row, so P also carries 4 szp
orbitals) lets the dopant site map onto the host site it replaces with
``match_species=False`` — the same geometric correspondence the
[SIESTA Si:P example](/examples/siesta-p-doped/) uses. Host-like bands
carry stronger weights while impurity-hybridized states spread among
folds. The finite k-grid limits the generic-k ideal projection; do not
interpret these weights as an exact four-fold Parseval partition.

## Reproduce this example

Run ``python examples/gpaw_si/generate_fixtures.py`` (gpaw >= 26 in the
mydev environment; ~2 minutes serial) to regenerate the ``.gpw``
restarts, then ``python examples/gpaw_si/unfold.py`` to redraw both
figures. Both scripts are headless.
