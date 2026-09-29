---
title: "SIESTA WFSX route"
weight: 3
---

If the supercell run already stored its wavefunctions, unfold from
SIESTA's own eigenvectors and eigenvalues instead of diagonalizing the
Hamiltonian: an 8-atom conventional-cubic Si supercell run
(`SaveWFSX true`) unfolded onto the 2-atom primitive fcc cell. Same
cell, same path, same weights as the
[Hamiltonian route](../siesta-si/) — only the eigen-solve is replaced;
the weight still uses the `.HSX` overlap shells.

## The bundle

Download [siesta-wfsx.tar.gz](/downloads/siesta-wfsx.tar.gz) and unpack
it:

```console
tar xf siesta-wfsx.tar.gz && cd siesta-wfsx
```

Shipped: the WFSX path run (`data/si_sc_path.selected.WFSX` plus its
`si_sc_path.EIG`), the Hamiltonian fixtures for the overlap shells and
the primitive reference (`data/si_prim.*`, `data/si_sc.*`), the
primitive POSCAR, the path-run deck (`inputs/si_sc_path.fdf`, with
`SaveWFSX true` and a `%block WaveFuncKPoints` list covering the path),
and `reproduce.py`. Prerequisites: `pip install unfolding` plus
`pip install HamiltonIO sisl` (needs HamiltonIO ≥ 0.3.6 for
`SiestaWFSXParser`). No SIESTA run is needed.

## Structure and k-path

| | |
|---|---|
| primitive cell | 2-atom fcc Si, a = 5.430 Å |
| supercell | 8-atom conventional cubic cell, `M = [[-1,1,1],[1,-1,1],[1,1,-1]]` |
| k-path | Γ-X-W-Γ-L-W-X, **157 stored k-points** (nominal segment-density parameter 150); the WFSX matches only its written `%block WaveFuncKPoints` list, so `unfold.toml` carries those coordinates explicitly |

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/).
Primitive cell input: required for the `siesta-wfsx` route — it supplies the relabel map and the path frame (`data/si_prim.fdf`).

## Configuration

```toml
# SIESTA WFSX route: unfold from SIESTA's own stored wavefunctions.
# Run from the unpacked bundle root:  unfolding --config unfold.toml
# or with explicit flags (the kpoint list is long -- prefer the TOML;
# reproduce.py shows the Python equivalent):
#   unfolding siesta-wfsx --wfsx data/si_sc_path.selected.WFSX \
#     --hs-fdf data/si_sc.fdf --primitive data/si_prim.fdf \
#     --unfold-mat -1 1 1 1 -1 1 1 1 -1 \
#     --kpoints <157 primitive-frame points, below> \
#     --names G X W G L W X --xticks 0.0000, 1.1571, 1.7357, 3.0294, 4.0315, 4.8497, 5.4283 \
#     --method ideal --output si_wfsx_unfolded_cli.png
#
# The [path] kpoints are the EXACT grid SIESTA stored wavefunctions on
# (the segment-proportional Gamma-X-W-Gamma-L-W-X list of the deck's
# %block WaveFuncKPoints, 150 points nominal -> 157 with repeated
# junctions). The unfolder matches requested momenta to the stored
# entries at 1e-6 tolerance, so the plotting grid must be this list --
# a regenerated special_points path would not hit the stored points.
route = "siesta-wfsx"

[input]
wfsx = "data/si_sc_path.selected.WFSX"   # stored path wavefunctions
hs_fdf = "data/si_sc.fdf"                # supercell run providing H/S (overlap shells)

[structure]
primitive = "data/si_prim.fdf"           # 2-atom primitive fcc cell (path frame)
supercell_matrix = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]   # conventional = M @ primitive

[path]
kpoints = [
  [0.0, 0.0, 0.0],
  [0.0151515152, 0.0, 0.0151515152],
  [0.0303030303, 0.0, 0.0303030303],
  [0.0454545455, 0.0, 0.0454545455],
  [0.0606060606, 0.0, 0.0606060606],
  [0.0757575758, 0.0, 0.0757575758],
  [0.0909090909, 0.0, 0.0909090909],
  [0.1060606061, 0.0, 0.1060606061],
  [0.1212121212, 0.0, 0.1212121212],
  [0.1363636364, 0.0, 0.1363636364],
  [0.1515151515, 0.0, 0.1515151515],
  [0.1666666667, 0.0, 0.1666666667],
  [0.1818181818, 0.0, 0.1818181818],
  [0.196969697, 0.0, 0.196969697],
  [0.2121212121, 0.0, 0.2121212121],
  [0.2272727273, 0.0, 0.2272727273],
  [0.2424242424, 0.0, 0.2424242424],
  [0.2575757576, 0.0, 0.2575757576],
  [0.2727272727, 0.0, 0.2727272727],
  [0.2878787879, 0.0, 0.2878787879],
  [0.303030303, 0.0, 0.303030303],
  [0.3181818182, 0.0, 0.3181818182],
  [0.3333333333, 0.0, 0.3333333333],
  [0.3484848485, 0.0, 0.3484848485],
  [0.3636363636, 0.0, 0.3636363636],
  [0.3787878788, 0.0, 0.3787878788],
  [0.3939393939, 0.0, 0.3939393939],
  [0.4090909091, 0.0, 0.4090909091],
  [0.4242424242, 0.0, 0.4242424242],
  [0.4393939394, 0.0, 0.4393939394],
  [0.4545454545, 0.0, 0.4545454545],
  [0.4696969697, 0.0, 0.4696969697],
  [0.4848484848, 0.0, 0.4848484848],
  [0.5, 0.0, 0.5],
  [0.5, 0.0147058824, 0.5147058824],
  [0.5, 0.0294117647, 0.5294117647],
  [0.5, 0.0441176471, 0.5441176471],
  [0.5, 0.0588235294, 0.5588235294],
  [0.5, 0.0735294118, 0.5735294118],
  [0.5, 0.0882352941, 0.5882352941],
  [0.5, 0.1029411765, 0.6029411765],
  [0.5, 0.1176470588, 0.6176470588],
  [0.5, 0.1323529412, 0.6323529412],
  [0.5, 0.1470588235, 0.6470588235],
  [0.5, 0.1617647059, 0.6617647059],
  [0.5, 0.1764705882, 0.6764705882],
  [0.5, 0.1911764706, 0.6911764706],
  [0.5, 0.2058823529, 0.7058823529],
  [0.5, 0.2205882353, 0.7205882353],
  [0.5, 0.2352941176, 0.7352941176],
  [0.5, 0.25, 0.75],
  [0.4864864865, 0.2432432432, 0.7297297297],
  [0.472972973, 0.2364864865, 0.7094594595],
  [0.4594594595, 0.2297297297, 0.6891891892],
  [0.4459459459, 0.222972973, 0.6689189189],
  [0.4324324324, 0.2162162162, 0.6486486486],
  [0.4189189189, 0.2094594595, 0.6283783784],
  [0.4054054054, 0.2027027027, 0.6081081081],
  [0.3918918919, 0.1959459459, 0.5878378378],
  [0.3783783784, 0.1891891892, 0.5675675676],
  [0.3648648649, 0.1824324324, 0.5472972973],
  [0.3513513514, 0.1756756757, 0.527027027],
  [0.3378378378, 0.1689189189, 0.5067567568],
  [0.3243243243, 0.1621621622, 0.4864864865],
  [0.3108108108, 0.1554054054, 0.4662162162],
  [0.2972972973, 0.1486486486, 0.4459459459],
  [0.2837837838, 0.1418918919, 0.4256756757],
  [0.2702702703, 0.1351351351, 0.4054054054],
  [0.2567567568, 0.1283783784, 0.3851351351],
  [0.2432432432, 0.1216216216, 0.3648648649],
  [0.2297297297, 0.1148648649, 0.3445945946],
  [0.2162162162, 0.1081081081, 0.3243243243],
  [0.2027027027, 0.1013513514, 0.3040540541],
  [0.1891891892, 0.0945945946, 0.2837837838],
  [0.1756756757, 0.0878378378, 0.2635135135],
  [0.1621621622, 0.0810810811, 0.2432432432],
  [0.1486486486, 0.0743243243, 0.222972973],
  [0.1351351351, 0.0675675676, 0.2027027027],
  [0.1216216216, 0.0608108108, 0.1824324324],
  [0.1081081081, 0.0540540541, 0.1621621622],
  [0.0945945946, 0.0472972973, 0.1418918919],
  [0.0810810811, 0.0405405405, 0.1216216216],
  [0.0675675676, 0.0337837838, 0.1013513514],
  [0.0540540541, 0.027027027, 0.0810810811],
  [0.0405405405, 0.0202702703, 0.0608108108],
  [0.027027027, 0.0135135135, 0.0405405405],
  [0.0135135135, 0.0067567568, 0.0202702703],
  [0.0, 0.0, 0.0],
  [0.0172413793, 0.0172413793, 0.0172413793],
  [0.0344827586, 0.0344827586, 0.0344827586],
  [0.0517241379, 0.0517241379, 0.0517241379],
  [0.0689655172, 0.0689655172, 0.0689655172],
  [0.0862068966, 0.0862068966, 0.0862068966],
  [0.1034482759, 0.1034482759, 0.1034482759],
  [0.1206896552, 0.1206896552, 0.1206896552],
  [0.1379310345, 0.1379310345, 0.1379310345],
  [0.1551724138, 0.1551724138, 0.1551724138],
  [0.1724137931, 0.1724137931, 0.1724137931],
  [0.1896551724, 0.1896551724, 0.1896551724],
  [0.2068965517, 0.2068965517, 0.2068965517],
  [0.224137931, 0.224137931, 0.224137931],
  [0.2413793103, 0.2413793103, 0.2413793103],
  [0.2586206897, 0.2586206897, 0.2586206897],
  [0.275862069, 0.275862069, 0.275862069],
  [0.2931034483, 0.2931034483, 0.2931034483],
  [0.3103448276, 0.3103448276, 0.3103448276],
  [0.3275862069, 0.3275862069, 0.3275862069],
  [0.3448275862, 0.3448275862, 0.3448275862],
  [0.3620689655, 0.3620689655, 0.3620689655],
  [0.3793103448, 0.3793103448, 0.3793103448],
  [0.3965517241, 0.3965517241, 0.3965517241],
  [0.4137931034, 0.4137931034, 0.4137931034],
  [0.4310344828, 0.4310344828, 0.4310344828],
  [0.4482758621, 0.4482758621, 0.4482758621],
  [0.4655172414, 0.4655172414, 0.4655172414],
  [0.4827586207, 0.4827586207, 0.4827586207],
  [0.5, 0.5, 0.5],
  [0.5, 0.4895833333, 0.5104166667],
  [0.5, 0.4791666667, 0.5208333333],
  [0.5, 0.46875, 0.53125],
  [0.5, 0.4583333333, 0.5416666667],
  [0.5, 0.4479166667, 0.5520833333],
  [0.5, 0.4375, 0.5625],
  [0.5, 0.4270833333, 0.5729166667],
  [0.5, 0.4166666667, 0.5833333333],
  [0.5, 0.40625, 0.59375],
  [0.5, 0.3958333333, 0.6041666667],
  [0.5, 0.3854166667, 0.6145833333],
  [0.5, 0.375, 0.625],
  [0.5, 0.3645833333, 0.6354166667],
  [0.5, 0.3541666667, 0.6458333333],
  [0.5, 0.34375, 0.65625],
  [0.5, 0.3333333333, 0.6666666667],
  [0.5, 0.3229166667, 0.6770833333],
  [0.5, 0.3125, 0.6875],
  [0.5, 0.3020833333, 0.6979166667],
  [0.5, 0.2916666667, 0.7083333333],
  [0.5, 0.28125, 0.71875],
  [0.5, 0.2708333333, 0.7291666667],
  [0.5, 0.2604166667, 0.7395833333],
  [0.5, 0.25, 0.75],
  [0.5, 0.234375, 0.734375],
  [0.5, 0.21875, 0.71875],
  [0.5, 0.203125, 0.703125],
  [0.5, 0.1875, 0.6875],
  [0.5, 0.171875, 0.671875],
  [0.5, 0.15625, 0.65625],
  [0.5, 0.140625, 0.640625],
  [0.5, 0.125, 0.625],
  [0.5, 0.109375, 0.609375],
  [0.5, 0.09375, 0.59375],
  [0.5, 0.078125, 0.578125],
  [0.5, 0.0625, 0.5625],
  [0.5, 0.046875, 0.546875],
  [0.5, 0.03125, 0.53125],
  [0.5, 0.015625, 0.515625],
  [0.5, 0.0, 0.5],
]
names = ["Γ", "X", "W", "Γ", "L", "W", "X"]
xticks = [0.0000, 1.1571, 1.7357, 3.0294, 4.0315, 4.8497, 5.4283]

[options]
method = "ideal"

[output]
output = "si_wfsx_unfolded_cli.png"
```

## Run it

From the unpacked bundle directory, one command with the shipped
config:

```console
unfolding --config unfold.toml          # -> si_wfsx_unfolded_cli.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

The config carries the **exact grid SIESTA stored wavefunctions on**
(the segment-proportional Γ-X-W-Γ-L-W-X list of the run deck's
`%block WaveFuncKPoints`, 150 points nominal → 157 with repeated
junctions): the unfolder matches requested momenta to the stored
entries at 1e-6 tolerance, so a regenerated special-points path would
not hit the stored points. With explicit flags (kpoint list as in the
TOML):

```console
unfolding siesta-wfsx --wfsx data/si_sc_path.selected.WFSX \
    --hs-fdf data/si_sc.fdf --primitive data/si_prim.fdf \
    --unfold-mat -1 1 1 1 -1 1 1 1 -1 --kpoints ... \
    --names G X W G L W X --method ideal \
    --output si_wfsx_unfolded_cli.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from HamiltonIO.siesta import SislParser
from HamiltonIO.siesta.wfsx import SiestaWFSXParser
from unfolding.lcao_unfolder import HamiltonIOModel
from unfolding.mapping import RelabelMap
from unfolding.wfsx_unfolder import WFSXUnfolder

sc_model = SislParser("data/si_sc.fdf").get_model()
prim_model = SislParser("data/si_prim.fdf").get_model()

cell = np.asarray(sc_model.atoms.cell)
wfsx = SiestaWFSXParser("data/si_sc_path.selected.WFSX", cell=cell).read()
M = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
rm = RelabelMap.from_atoms(sc_model.atoms, prim_model.atoms, M)
unf = WFSXUnfolder(wfsx, HamiltonIOModel(sc_model), rm, sc_mat=M)
result = unf.compute(kpts, method="ideal")   # same path as the HSX route
```

The spectrum is SIESTA's own (as written by the run). WFSX energies are
returned exactly as SIESTA stores them — Fermi-shifted by the writing
run — so subtract the `.EIG` header value for absolute eigenvalues; the
route does this by default, and `reproduce.py` does it for the
published figure with the primitive-cell bands overlaid:

```console
python reproduce.py                     # -> si_wfsx_unfolded.png
```

{{< figure src="/images/si_wfsx_unfolded.png" title="Unfolded SIESTA Si$_8$ bands from stored WFSX wavefunctions along Γ-X-W-Γ-L-W-X; blue intensity encodes weight, red curves are primitive-cell bands; energies relative to E_F." >}}

## The fixtures

Same SIESTA setup as the pristine example (GGA-PBE, SZ PAO basis,
`MeshCutoff 100 Ry`, `SaveHS true`) plus the path run with
`SaveWFSX true`. WFSX coefficients use SIESTA's orbital-position gauge,
which the unfolder converts internally. For your own system: run the
supercell with `SaveHS true` (for the overlap shells) and a second run
with `SaveWFSX true` writing wavefunctions along your path.
