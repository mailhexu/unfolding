---
title: "ABINIT WFK AFM NiO"
weight: 8
---

Unfold type-II antiferromagnetic NiO from the 4-atom magnetic primitive
cell onto the 2-atom rocksalt primitive cell. The magnetic cell doubles
the rocksalt primitive cell along [111] and carries two Ni moments (+m
and −m along z) and two O; the collinear WFK (`nsppol 2`) stores two
spin channels, and the unfolding weight of a state is computed within
its own spin channel — no spin mixing.

## The bundle

Download [abinit-nio.tar.gz](/downloads/abinit-nio.tar.gz) and unpack
it:

```console
tar xf abinit-nio.tar.gz && cd abinit-nio
```

Shipped: the decks (`inputs/nio_afm.abi` dense 305-point path,
`inputs/nio_afm_corners.abi` Γ/X/W/L corners), the pseudopotentials
(`pseudos/Ni.psp8`, `pseudos/O.psp8`), `unfold.toml`, the
primitive-cell POSCAR (`data/nio_prim.vasp`), and `reproduce.py`. Not
shipped: the WFK itself (the dense nsppol-2 path WFK is ~1.1 GB) —
regenerate it with ABINIT as below; this is the bundle's one
user-supplied input. Prerequisites: `pip install unfolding`; ABINIT ≥ 9
with netCDF support only for the regeneration step.

## Structure and k-path

| | |
|---|---|
| magnetic cell | `acell 3*7.8817` bohr (a = 4.171 Å) with fcc `rprim`; `A_afm = M @ A_prim`, `M = [[1,0,1],[0,1,1],[1,1,0]]` |
| atoms | Ni +m at (0,0,0), Ni −m at (.5,.5,.5), O at (.25,.25,.25) and (.75,.75,.75), `spinat 0 0 2.0 / 0 0 -2.0` on the two Ni |
| primitive cell | 2-atom rocksalt (`data/nio_prim.vasp`), the path-points frame |
| k-path | Γ-X-W-Γ-L-W-X, 305 deck points, fcc special points in primitive reciprocal coordinates (the rocksalt primitive cell is fcc) |
| spin | `spin=0` up, `spin=1` down — the channels coincide band-for-band here (AFM keeps inversion: time reversal × sublattice translation maps them onto each other), so one spin-up panel is plotted |

Deck essentials: `ecut 40` Ha, `nband 40`, `nsppol 2`/`nspden 2`,
`ixc 11`; dataset 1 the AFM SCF on a 4×4×4 mesh, dataset 2 a
frozen-density non-SCF path run (`iscf -2`, `getden 2`, `kptopt 0`,
`istwfk 1`); plus the reader contract `iomode 3`, `prtwf 1`,
`chkprim 0`. The 18-valence-electron Ni puts the Ni-3s semicore
multiplet near −60 eV in the WFK, outside the plotted −16…8 eV window.

## Run it

Regenerate the WFK (reproduce.py prints this exact command; the corner
deck gives a small WFK in minutes for a coarse figure):

```console
mkdir -p run_nio && cd run_nio
cp ../inputs/nio_afm.abi ../pseudos/Ni.psp8 ../pseudos/O.psp8 .
abinit nio_afm.abi > nio_afm.abo 2>&1
cp nio_afmo_DS2_WFK.nc ../data/ && cd ..
```

then, from the bundle root, one command with the shipped config:

```console
unfolding --config unfold.toml          # -> nio_afm_unfolded_cli.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or with explicit flags:

```console
unfolding abinit-wfk --wfk data/nio_afm_patho_DS2_WFK.nc \
    --unfold-mat 1 0 1 0 1 1 1 1 0 --primitive data/nio_prim.vasp \
    --kpoints $(cat data/path_kpoints.txt) \
    --xcoords $(cat data/path_xcoords.txt) \
    --names G X W G L W X --xticks $(cat data/path_xticks.txt) \
    --spin 0 --resolve-degenerate 0.001 --output nio_afm_unfolded_cli.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from unfolding import unfold_abinit

kpts = np.loadtxt("data/path_kpoints.txt").reshape(-1, 3)
x = np.loadtxt("data/path_xcoords.txt")
X = np.loadtxt("data/path_xticks.txt")
m_afm = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
ax = unfold_abinit(
    "data/nio_afm_patho_DS2_WFK.nc", m_afm, kpts,
    knames=[r"$\Gamma$", "X", "W", r"$\Gamma$", "L", "W", "X"],
    xqpts=x, Xqpts=X,
    spin=0,
    resolve_degenerate=1e-3,
)
```

`python reproduce.py` renders the published figure from the dense WFK
(or a coarse map from the corner WFK, on the same path axis).

{{< figure src="/images/nio_afm_unfolded.png" title="AFM NiO unfolded onto the 2-atom rocksalt primitive cell along Γ-X-W-Γ-L-W-X, spin-up channel: Gaussian-smeared spectral-weight map (Blues scale, opacity = weight), energies relative to the WFK Fermi level" >}}

Systems whose spin channels genuinely differ are handled by the same
call with `spin=1` for the partner panel.
