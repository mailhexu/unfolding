---
title: "ABINIT WFK Si and Si:P"
weight: 6
---

Unfold ABINIT supercell calculations directly from their wavefunction
files (norm-conserving plane waves: the weight is a pure
reciprocal-coset projection — no overlaps or atom maps are involved).
Two systems: pristine Si in the 8-atom conventional cell and Si:P with
one substitutional P, both unfolded onto the primitive fcc cell along
Γ–X–W–Γ–L–W–X. Pristine weights are binary — the unfolded bands ARE the
primitive band structure — while defect-hybridized states carry
fractional weight and render dimmer.

## The bundle

Download [abinit-wfk-si.tar.gz](/downloads/abinit-wfk-si.tar.gz) and
unpack it:

```console
tar xf abinit-wfk-si.tar.gz && cd abinit-wfk-si
```

Shipped: the ABINIT decks (`inputs/`, two-dataset SCF + non-SCF path
runs for Si8 and Si₇P, plus the 2-atom primitive reference), the
Troullier–Martins fhi pseudopotentials (`pseudos/14-Si.nlcc.fhi`,
`15-P.LDA.fhi`), `unfold.toml`, the primitive-cell POSCAR
(`data/si_prim.vasp`), and `reproduce.py`. Not shipped: the WFK files
themselves (each dense path WFK is hundreds of MB) — regenerate them
with ABINIT as below; this is the bundle's one user-supplied input.
Prerequisites: `pip install unfolding`; ABINIT ≥ 9 with netCDF support
only for the regeneration step.

## ABINIT settings that matter

| setting | role |
|---|---|
| `iomode 3` | write a netCDF WFK (plain Fortran-binary WFKs are rejected with this remedy) |
| `istwfk 1` | store the full G sphere (half-sphere storage cannot be reconstructed) |
| `prtwf 1` | keep the raw coefficients |
| `chkprim 0` | needed whenever the supercell itself is non-primitive |

Each deck is `ndtset 2`: dataset 1 an SCF at supercell Γ writing the
density, dataset 2 a frozen-density non-SCF run (`iscf -2`, `getden2 1`,
`kptopt 0`) sampling the path with an explicit `kpt2` list. Give the
path dataset a few bands of headroom and a tight `tolwfr2`; converge
`ecut` before producing figures (`ecut 25` Ha here; the upper valence is
stable by ~15–18 Ha with these pseudos while the deeper s manifold keeps
moving). Use the shipped pseudos: a locally generated ONCVPSP-4.0.1 file
is misparsed by ABINIT and silently adds flat ghost bands below the
valence — the tell-tale is a Γ-point spectrum with extra singlets below
the top valence triplet and a metallic pristine Si.

## Structure and k-path

| | |
|---|---|
| supercell | 8-atom conventional cubic cell, `acell 3*10.26` bohr; `M = [[-1,1,1],[1,-1,1],[1,1,-1]]`, conventional = M @ primitive fcc, so supercell momenta are `K = k_prim @ M.T` |
| k-path | Γ-X-W-Γ-L-W-X, 305 deck points / 300 plotted, fcc special points in primitive reciprocal coordinates; the decks pass the SAME primitive points as `kpt2 @ M.T` |

## Run it

Regenerate the WFKs (reproduce.py prints these exact commands; the
corner decks give small WFKs in minutes for coarse figures, the dense
decks the published 305-point maps):

```console
mkdir -p run_si8 && cd run_si8
cp ../inputs/si8_gxwglwx.abi ../pseudos/14-Si.nlcc.fhi .
abinit si8_gxwglwx.abi > si8_gxwglwx.abo 2>&1
cp si8_gxwglwxo_DS2_WFK.nc ../data/ && cd ..
```

then, from the bundle root, one command with the shipped config:

```console
unfolding --config unfold.toml          # -> si8_abinit_unfolded_cli.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or with explicit flags:

```console
unfolding abinit-wfk --wfk data/si8_gxwglwxo_DS2_WFK.nc \
    --unfold-mat -1 1 1 1 -1 1 1 1 -1 --primitive data/si_prim.vasp \
    --kpoints $(cat data/path_kpoints.txt) \
    --xcoords $(cat data/path_xcoords.txt) \
    --names G X W G L W X --xticks $(cat data/path_xticks.txt) \
    --resolve-degenerate 0.001 --output si8_abinit_unfolded_cli.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from unfolding import unfold_abinit

kpts = np.loadtxt("data/path_kpoints.txt").reshape(-1, 3)
x = np.loadtxt("data/path_xcoords.txt")
X = np.loadtxt("data/path_xticks.txt")
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
ax = unfold_abinit(
    "data/si8_gxwglwxo_DS2_WFK.nc",
    M,
    kpts,                               # primitive-frame path points
    knames=[r"$\Gamma$", "X", "W", r"$\Gamma$", "L", "W", "X"],
    xqpts=x, Xqpts=X,
    resolve_degenerate=1e-3,            # eV; keeps degenerate branch weights 0/1
)
```

`python reproduce.py` renders both published figures (Si8 with the
crimson primitive-cell overlay when `data/si_prim_patho_DS2_WFK.nc` is
present, Si₇P spectral weight), falling back to the corner WFKs on the
same path axis.

{{< figure src="/images/si8_abinit_unfolded.png" title="Pristine Si (8-atom conventional cell) unfolded onto the Γ-X-W-Γ-L-W-X primitive path: Gaussian-smeared spectral-weight map (Blues scale, opacity = weight), energies relative to the WFK Fermi level; crimson curves are primitive-cell bands from an independent primitive-cell calculation" >}}

{{< figure src="/images/si7p_abinit_unfolded.png" title="Si:P (one P substituting Si in the 8-atom cell) unfolded onto the same path: Gaussian-smeared spectral-weight map (Blues scale, opacity = weight), energies relative to the WFK Fermi level" >}}

## Parameters and weights

Each requested primitive k-point must be present in the WFK (the
adapter maps it to the stored supercell momentum internally: primitive
`(0, t/2, t/2)` is conventional-cell `(t, 0, 0)` here). Eigenvalues are
converted from Hartree to eV and shifted by the WFK's Fermi energy by
default (`fermi_shift=false` for absolute energies). `resolve_degenerate`
(eV) eigen-assigns gauge-invariant branch weights inside
near-degenerate groups — needed at degenerate points where the stored
gauge is arbitrary; `average_degenerate` (eV) instead averages weights
over such groups. Collinear spin channels are selected with `spin=`.
For weights without plotting:
`read_wfk` + `PWEigenData` + `PWUnfolder` (see the bundle's
`reproduce.py` for a worked example).

Physics to check: pristine Si shows the LDA indirect Γ-X gap of
~0.4–0.5 eV with the Fermi level mid-gap, and after a single constant
potential-reference shift the unfolded supercell bands agree with the
independent primitive-cell calculation to ~0.07 eV along the path.
