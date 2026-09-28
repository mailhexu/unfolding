---
title: "GPAW LCAO Si and Si:P"
weight: 10
---

Unfold a GPAW LCAO supercell calculation onto the primitive-cell band
path: pristine Si and P-doped Si in the 8-atom conventional cubic cell,
unfolded onto the 2-atom fcc primitive cell through
`HamiltonIO.gpaw.GpawLcaoModel`. GPAW's LCAO matrices already carry every
PAW contribution — the overlap is the projector-augmented one and the
Hamiltonian includes the ``dH`` projector terms — so the unfolding
backend consumes them exactly like any other atomic-orbital table: no
PAW correction is applied on top.

## Running the example

``HamiltonIO.gpaw.GpawLcaoModel`` turns a converged LCAO calculation
(live calculator or ``mode='all'`` ``.gpw`` restart) into the real-space
model interface the unfolder needs: ``.atoms`` (the supercell),
``.HR``/``.SR`` dictionaries keyed by integer supercell-lattice
translations, and ``hs_and_eigen(k) -> (H, S)`` at any supercell
fractional k-point, HamiltonIO convention 2
(``H(k) = sum_T H[T] e^{+2 pi i k.T}``), H in eV. Two run settings make
the transform well-defined:

| Run setting | Why |
|---|---|
| Gamma-centered k-grid: ``kpts={'size': (n,n,n), 'gamma': True}`` | The real-space tables of GPAW's default half-shifted even grids are not Hermitian (the LCAO matrices carry orbital position phases); the reader rejects such files with this remedy. |
| ``mode='all'`` restart | The Hamiltonian matrices are built from the stored density *and* wavefunctions. |

The supercell matrix is the conventional cubic cell in primitive-lattice
units (row convention ``A_sc = B @ A_prim``):

```python
B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
```

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

For the doped supercell, substitute one Si by P (same valence row, so P
also carries 4 szp orbitals) and let the dopant site map onto the host
site it replaces with ``match_species=False`` — the same geometric
correspondence the [SIESTA Si:P example](/examples/siesta-p-doped/)
uses.

{{< figure src="/images/gpaw_si_unfolded.png" title="GPAW LCAO Si$_8$ (8-atom conventional cell) unfolded onto the primitive Γ-X-W-Γ-L-W-X path; blue color intensity encodes spectral weight, crimson curves are the independently computed primitive-cell bands; energies in eV with zero at the unfolded run's Fermi level" >}}

{{< figure src="/images/gpaw_si_p_doped.png" title="GPAW LCAO Si:P unfolded along the same path (dopant mapped onto the host site); same encoding; the crimson primitive reference is drawn on the doped run's Fermi zero" >}}

## Calculation background

The committed fixtures live in `tests/data/gpaw_example/`
(`si_prim_lcao.gpw`, `si_sc_lcao.gpw`, `si7p_lcao.gpw`): GPAW LCAO runs,
PBE, default szp LCAO basis (4 atomic orbitals per atom for Si *and*
P), Gamma-centered k-grids — 16×16×16 for the primitive cell and 8×8×8
for both supercells. Internally the real-space tables are the inverse
lattice Fourier transform of the full (`symmetry='off'`) k-grid data;
for even grids the Nyquist translation is split equally between positive
and negative images, so the grid must resolve the real-space range of
the tables for the generic-k interpolant to be valid between grid
points. Because a uniform grid cannot disentangle the ``+N/2`` and
``-N/2`` shells, Gamma-centered grids are required.

Regenerate the ``.gpw`` restarts with
``python examples/gpaw_si/generate_fixtures.py`` (gpaw >= 26 in the
mydev environment; the 8×8×8 grids need ~1 hour serial — the planewave
runs are skipped when the committed fixtures are staged into the
workdir), then redraw both figures with
``python examples/gpaw_si/unfold.py``. Both scripts are headless and
write to `docs/static/images/`.
