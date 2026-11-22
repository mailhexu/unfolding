---
title: "TB2J magnons SrMnO3"
weight: 9
---

Downfold magnon band structures computed from TB2J exchange parameters
onto the primitive magnon Brillouin zone: the antiferromagnetic DFT
cell (√2×√2×√2 G-AFM SrMnO₃, 10 atoms) is larger than the chemical
primitive cell (5-atom pseudo-cubic, one Mn), so the magnon bands fold
and unfolding assigns each folded branch its primitive-cell momentum.

```text
SrMnO3 VASP/Wannier90 DFT (G-AFM)
  -> TB2J pickle: JR tensors, 10-atom sqrt2 x sqrt2 x sqrt2 cell, 2 Mn
  -> TB2J Magnon: BdG magnon bands (collinear reference, Q = 0)
  -> unfolding: weights on the 5-atom pseudo-cubic primitive cell
```

## The bundle

Download
[tb2j-magnon-srmmo3.tar.gz](/downloads/tb2j-magnon-srmmo3.tar.gz) and
unpack it:

```console
tar xf tb2j-magnon-srmmo3.tar.gz && cd tb2j-magnon-srmmo3
```

Shipped: the TB2J exchange parameters (`data/TB2J_results/TB2J.pickle`,
from a VASP/Wannier90 G-AFM run: 10 atoms, 2 Mn at ±2.81 μB, propagation
Q=(½,½,½)), `unfold.toml`, and `reproduce.py`. Prerequisites:
`pip install "unfolding[tb2j]"` (TB2J).

## Structure and k-path

| | |
|---|---|
| unfold matrix | `M = [[0,1,1],[1,0,1],[1,1,0]]`, the 10-atom AFM cell in 5-atom pseudo-cubic units |
| primitive cell | derived as `inv(M) @` TB2J cell; the q-path special points are resolved on it |
| q-path | Γ-X-M-Γ-R, 200 points |
| reference | collinear two-sublattice frame: Q = 0, quantization axis ẑ, moments from the pickle |

Run parameters and shared options (`mode`, `spin`, `method`, `resolve_degenerate`, supercell-matrix row convention, k-path coordinates and frames, energy-reference conventions) are explained together in the [method, parameters, and k-path guide](/guide/method-parameters-kpaths/).
Primitive cell input: not used by the `magnon` route — the path cell is derived from the TB2J cell and the unfolding matrix.

## Configuration

```toml
# TB2J magnons: G-AFM SrMnO3, 10-atom AFM cell -> 5-atom pseudo-cubic cell.
# Run from the unpacked bundle root:  unfolding --config unfold.toml
# or with explicit flags:
#   unfolding magnon --results data/TB2J_results \
#     --unfold-mat 0 1 1 1 0 1 1 1 0 --special-points GXMGR \
#     --npts 200 --output magnon_unfolded_cli.png
route = "magnon"

[input]
results = "data/TB2J_results"          # directory holding TB2J.pickle

[structure]
supercell_matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]   # A_afm = M @ A_pc

[path]
special_points = "GXMGR"               # resolved on inv(M) @ TB2J cell (pseudo-cubic)
npts = 200

[output]
output = "magnon_unfolded_cli.png"
```

## Run it

From the unpacked bundle directory, one command with the shipped
config:

```console
unfolding --config unfold.toml          # -> magnon_unfolded_cli.png
```

The same TOML can be passed as the only Python input:

```python
from unfolding import load_config, run
run(load_config("unfold.toml"))
```

or with explicit flags:

```console
unfolding magnon --results data/TB2J_results \
    --unfold-mat 0 1 1 1 0 1 1 1 0 --special-points GXMGR \
    --npts 200 --output magnon_unfolded_cli.png
```

or through the Python API with explicit parameters:

```python
import numpy as np
from unfolding import unfold_tb2j

m_afm = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])   # A_afm = M @ A_pc

ax = unfold_tb2j(
    "data/TB2J_results",              # directory with TB2J.pickle
    m_afm,
    qpts,                             # pseudo-cubic primitive q-path
    knames=[r"$\Gamma$", "X", "M", r"$\Gamma$", "R"],
    xqpts=x, Xqpts=X,
)
```

`reproduce.py` renders the published figure through the Python API
**and** checks the physics seals — a Goldstone mode at Γ
(< 0.05 meV) and binary unfolding weights (every weight 0 or 1 to
1e-8) for the pristine G-AFM:

```console
python reproduce.py                     # -> srmmo3_unfolded_api.png, srmmo3.png
```

{{< figure src="/images/srmmo3_magnon_unfolded.png" title="G-AFM SrMnO₃ magnons downfolded from the 10-atom AFM cell onto the 5-atom pseudo-cubic cell along Γ-X-M-Γ-R: line opacity encodes the unfolded weight per primitive momentum; energy axis in meV, dashed line at 0 meV" >}}

For your own system: run TB2J on the AFM supercell (any supported DFT
code), keep the `TB2J_results` directory, and set the unfold matrix
that relates your AFM cell to the primitive cell.

## Commensurate spiral supercell

A periodic spiral can use the same adapter: encode the sitewise moments in
the magnetic supercell and keep TB2J's `Q = 0`. The original spiral vector
is already in the moments and must not also be applied between supercells.
The following synthetic chain is a complete six-site example, not DFT data.
It uses both directions of $J_1=1$ meV and $J_2=-0.5$ meV, moments
$2\,\mu_B$ ($S=1$), and a stationary spiral of pitch $Q=\pi/3$:

```python
import numpy as np
from TB2J.magnon.magnon3 import Magnon
from unfolding import MagnonUnfolder
from unfolding.tb2j_unfold import magnon_eigendata_from_tb2j

n = 6
rs = np.arange(-2, 3)
jr = np.zeros((len(rs), n, n, 3, 3))
for i in range(n):
    for distance, coupling in ((1, 0.001), (2, -0.0005)):  # eV
        for d in (-distance, distance):
            cell, j = divmod(i + d, n)
            jr[cell + 2, i, j] += coupling * np.eye(3)
theta = np.arange(n) * np.pi / 3
moments = 2 * np.column_stack((np.cos(theta), np.sin(theta), np.zeros(n)))
model = Magnon(nspin=n, magmom=moments,
               Rlist=np.column_stack((rs, 0 * rs, 0 * rs)), JR=jr,
               cell=np.diag([n, 1, 1]), _Q=np.zeros(3),
               _uz=np.array([[0., 0., 1.]]), _n=np.array([0., 0., 1.]),
               positions=np.column_stack((np.arange(n), np.zeros((n, 2)))))
model.set_reference([0, 0, 0], [[0, 0, 1]], [0, 0, 1], magmoms=moments)
M = np.diag([6, 1, 1])  # A_supercell = M @ A_primitive
q = np.array([[0.05, 0, 0], [0.10, 0, 0], [0.20, 0, 0]])
data = magnon_eigendata_from_tb2j(model, np.mod(q @ M.T, 1))
result = MagnonUnfolder(data, M).compute(q)
```

For the directed-pair convention $E=-2\sum_{i,d}J_d\mathbf e_i\cdot
\mathbf e_{i+d}$, define $\mathcal J(k)=2[J_1\cos k+J_2\cos(2k)]$.
The helix maximizes $\mathcal J$ at $Q=\pi/3$ and its analytic branch is
$\omega(k)=\frac{2}{S}\sqrt{[\mathcal J(Q)-\mathcal J(k)]
[\mathcal J(Q)-(\mathcal J(k+Q)+\mathcal J(k-Q))/2]}$.
The regression compares its six folded copies with the computed bands and
with an independently assembled 30-site classical Hessian.

The adapter requires the matching TB2J version with its shared Goldstone
factorization. `result.energies` are in eV. TB2J's `uz=[[0,0,1]]` frame for
this planar helix has $U_i=e^{i\theta_i}(i\hat e_{\theta_i}-\hat z)$;
the positional character weights are defined in that local transverse gauge.
They are gauge-dependent and are not lab-frame neutron-scattering intensities.

Check that the input configuration is stationary and sample enough wavevectors
to test stability. TB2J rejects negative sampled Hessian eigenvalues beyond
roundoff; Goldstone points receive only roundoff-scale regularization when
needed. Neither that check nor finite sampling proves global stability.
For material calculations, use a converged noncollinear periodic supercell,
consistent site ordering/moments and exchange tensors in one Cartesian frame,
including both bond directions; verify that the DFT-to-TB2J extraction supports
that spinor input and the desired interactions. No such DFT calculation is
provided by this example.
