---
title: "ABINIT WFK Si and Si:P"
weight: 6
---

Unfold ABINIT supercell calculations directly from their wavefunction
files with the `unfold_abinit` adapter (norm-conserving plane waves: the
weight is a pure reciprocal-coset projection — no overlaps or atom maps
are involved). Two examples: pristine Si in the 8-atom conventional cell
(`M = [[-1,1,1],[1,-1,1],[1,1,-1]]`, conventional = M @ primitive fcc
cell) and Si:P with one substitutional P in the same cell, both unfolded
onto the primitive fcc cell along Γ–X–W–Γ–L–W–X.

## Running the example

The WFK reader (`HamiltonIO.abinit.read_wfk`, ABINIT 9/10 ETSF netCDF)
enforces a full-storage contract:

| ABINIT input setting | Role |
|---|---|
| `iomode 3` | write a netCDF WFK (a plain Fortran-binary WFK is rejected with this remedy) |
| `istwfk 1` | store the full G sphere (half-sphere storage cannot be reconstructed) |
| `prtwf 1` | keep the raw coefficients |
| `chkprim 0` | needed whenever the supercell itself is non-primitive |

Use a two-dataset input: dataset 1 an SCF run that writes the density,
dataset 2 a frozen-density non-SCF (`iscf -2`, `getden2 1`, `kptopt 0`)
that samples the path with an explicit `kpt2` list and `prtwf 1`. For a
converged fixture: give the path dataset a few bands of headroom beyond
the plotted window, converge it (`nstep2` generous, `tolwfr2` tight), and
converge `ecut` before producing figures (the fixtures use `ecut 25` Ha;
with the norm-conserving pseudopotentials used here the upper valence is
stable by `ecut ~ 15–18` Ha while the deeper s manifold keeps moving
past that). Use pseudopotentials your ABINIT build parses cleanly: a
locally generated ONCVPSP-4.0.1 file is misparsed by ABINIT and silently
adds flat ghost bands below the valence; the fixtures use the
Troullier-Martins fhi pseudos from the ABINIT test-suite Pspdir.

Unfold the Si:P WFK (fcc special points in primitive reciprocal
coordinates, Setyawan–Curtarolo; pass the SAME primitive points to the
non-SCF dataset as `kpt2 @ matrix.T`):

```python
import numpy as np
import matplotlib.pyplot as plt
from unfolding import unfold_abinit

matrix = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])   # = M @ primitive

special = {'G': (0, 0, 0), 'X': (.5, 0, .5), 'W': (.5, .25, .75),
           'L': (.5, .5, .5)}
names = 'GXWGLWX'
# ... build `kpts` by interpolating between the special points, and pass
# the SAME primitive points to the non-SCF dataset as kpt2 @ matrix.T.

ax = unfold_abinit(
    'si7p_patho_DS2_WFK.nc',      # your netCDF WFK
    matrix,
    kpts,
    knames=[r'$\Gamma$', 'X', 'W', r'$\Gamma$', 'L', 'W', 'X'],
    resolve_degenerate=1e-3,       # eV; keeps degenerate branch weights 0/1
)
```

{{< figure src="/images/si8_abinit_unfolded.png" title="Pristine Si (8-atom conventional cell) unfolded onto the Γ-X-W-Γ-L-W-X primitive path: Gaussian-smeared spectral-weight map (Blues scale, opacity = weight), energies relative to the WFK Fermi level; crimson curves are primitive-cell bands from an independent primitive-cell calculation" >}}

{{< figure src="/images/si7p_abinit_unfolded.png" title="Si:P (one P substituting Si in the 8-atom cell) unfolded onto the same path: Gaussian-smeared spectral-weight map (Blues scale, opacity = weight), energies relative to the WFK Fermi level" >}}

Each requested primitive k-point must be present in the WFK (the adapter
maps it to the stored supercell momentum internally: primitive
`(0, t/2, t/2)` is conventional-cell `(t, 0, 0)` here). Eigenvalues are
converted from Hartree to eV at this boundary and shifted by the WFK's
Fermi energy by default (`fermi_shift=False` for absolute energies);
`average_degenerate` (eV) optionally averages weights over
near-degenerate groups, and `resolve_degenerate` (eV) eigen-assigns
gauge-invariant branch weights inside such groups. Collinear spin
channels are selected with `spin=`.

To get weights without plotting:

```python
from HamiltonIO.abinit import read_wfk
from unfolding.pw_unfolder import PWEigenData, PWUnfolder

data = read_wfk('si7p_patho_DS2_WFK.nc')
eigen = PWEigenData(data.kpoints, data.gvecs, data.coefficients, data.eigenvalues)
result = PWUnfolder(eigen, matrix).compute(kpts)
```

`read_wfk` rejects unreadable WFK variants with actionable messages (the
`iomode 3` and `istwfk 1` remedies above). The same Γ–X–W–Γ–L–W–X path is
used by the [SIESTA Si example](/examples/siesta-si/).

## Calculation background

- Code: ABINIT (9/10 netCDF WFK), norm-conserving Troullier-Martins fhi
  pseudopotentials from the ABINIT test-suite Pspdir
  (`14-Si.nlcc.fhi`, `15-P.LDA.fhi`).
- Cell: 8-atom conventional cubic cell, `acell 3*10.26` bohr;
  supercell matrix `M = [[-1,1,1],[1,-1,1],[1,1,-1]]` (conventional = M
  @ primitive fcc cell), so supercell momenta are `K = k_prim @ M.T`.
- Decks (`tests/data/abinit_si/`): `si8_gxwglwx.abi` and
  `si7p_gamma_x_path.abi` — `ndtset 2`: dataset 1 SCF at supercell Γ
  (`nkpt1 1`, `istwfk1 1`), dataset 2 frozen-density non-SCF
  (`iscf2 -2`, `tolwfr2 1e-16`) over 305 path points; `ecut 25` Ha,
  `nband 24`, `iomode 3`. The 2-atom primitive reference run is
  `si_prim_path.abi`. Dense path WFKs are staged by
  `tests/data/abinit_si/regenerate_on_nic6.sh`.
- Figures: `python docgen/fig_abinit_si8.py` (overlays the primitive
  bands from `si_prim_patho_DS2_WFK.nc`) and
  `python docgen/fig_abinit_si7p.py`; both write into
  `docs/static/images/`.

Download the [complete input bundle](/downloads/abinit-wfk-si.tar.gz)
(`abinit-wfk-si.tar.gz`): input files, pseudopotentials, the fixture data
needed for the figure, a `reproduce.py` script, and a `README.txt`
with prerequisites and exact run instructions.
