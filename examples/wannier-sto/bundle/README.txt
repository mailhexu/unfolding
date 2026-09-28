====================================================================
Wannier90 SrTiO3 -- reproduction bundle (wannier-sto)
====================================================================

Companion bundle of the "Wannier90 SrTiO3" example
(https://mailhexu.github.io/unfolding/examples/wannier-sto/). Unfolds
tight-binding Hamiltonians produced by Wannier90 -- pristine and
oxygen-vacancy supercells of SrTiO3 -- onto the primitive cell: the
Wannier90 output is converted to a tight-binding model (reader from the
minimulti package, pip install minimulti), the supercell generalized
eigenproblem is solved along the path, and each supercell state is
assigned its primitive-cell spectral weight.

What the example shows
----------------------
For a pristine supercell every band folds from exactly one primitive
momentum, so the unfolding weights are binary: bold branches trace the
primitive bands and the folded copies stay invisible. With a defect,
the translation symmetry is broken: defect-derived states appear with
fractional weights while the host bands remain at weight 1 -- the
electronic analogue of the SIESTA dopant example
(https://mailhexu.github.io/unfolding/examples/siesta-p-doped/).

The bundled model is SYNTHETIC
------------------------------
The published figures (sto_nodefect.png / sto_defect.png on the docs
page) were computed from Wannier90 runs on 5-atom-cell SrTiO3
supercells (56 Wannier functions each); their wannier90_hr.dat files
are ~38 MB and are NOT part of this bundle. Instead the bundle ships a
small synthetic stand-in, data/example_hr.dat (+ data/example.win), so
the script is demonstrably runnable without DFT:

  * Ti t2g conduction-band model: 3 orbitals (d_xy, d_yz, d_zx) on a
    simple-cubic Ti sublattice, nearest-neighbour hopping t = 1 eV,
    on-site 0 (the standard cubic SrTiO3 conduction-band model),
  * 2x2x2 supercell: 24 orbitals, Wannier90 hr format,
  * --defect weakens the hoppings of one site (to 30%) and raises its
    on-site energy to +6 eV, so partly-localized impurity levels split
    off above the t2g bands with reduced, fractional weights.

The dispersion is schematic, not fitted to SrTiO3; the unfolding
physics demonstrated (binary pristine weights, fractional defect
weights) is the same as in the published figures.

BUNDLE LAYOUT
  unfold.toml                 config for the `unfolding` command (the
                              bundled synthetic model)
  reproduce.py                runs the unfolding and saves the figure
  data/example_hr.dat         synthetic t2g model in wannier90_hr.dat
                              format
  data/example.win            minimal win file so minimulti's MyTB
                              reader can open the synthetic model
  inputs/seedname.win.example annotated Wannier90 input sketch for
                              producing a real wannier90_hr.dat

PREREQUISITES
  - Python 3.11+ with numpy, matplotlib, ase
  - the unfolding package: pip install unfolding
  The `unfolding wannier` CLI route reads wannier90 runs with the
  package's own reader when minimulti's MyTB interface is unavailable;
  the bundled unfold.toml pins the win unit cell explicitly, so the CLI
  runs on the synthetic model as shipped.

STRUCTURE AND PATH
  - supercell matrix scmat = diag(2,2,2) of the primitive cell
  - labels name one supercell orbital each: the bundled model uses
    d_xy/d_yz/d_zx x 8 sites (24); the published SrTiO3 example uses
    ['pz','px','py']*12 + ['dz2','dxy','dyz','dx2','dxz']*4 (56)
  - path Gamma-X-M-Gamma-R, 200 points, vertices in primitive
    fractional coordinates: (0,0,0), (.5,0,0), (.5,.5,0), (0,0,0),
    (.5,.5,.5)

HOW TO RUN
  From the unpacked bundle directory, either the CLI (requires
  minimulti with its Wannier90 reader):

    unfolding --config unfold.toml        # writes sto_unfolded.png
    # equivalent explicit flags:
    unfolding wannier --path data --prefix example \
        --unfold-mat 2 0 0 0 2 0 0 0 2 \
        --labels d_xy d_yz d_zx d_xy d_yz d_zx ... \
        --kpoints 0 0 0 0.5 0 0 0.5 0.5 0 0 0 0 0.5 0.5 0.5 \
        --names G X M G R --npoints 200 --output sto_unfolded.png

  or the reproduction script (works without minimulti):

    python reproduce.py                   # pristine  -> sto_unfolded.png
    python reproduce.py --defect          # impurity  -> sto_defect.png

  Each run prints the fraction of binary vs fractional weights and
  writes the weight-coded figure (300 dpi) in the working directory.
  Expected: the pristine run shows 100% binary weights; the defect run
  shows fractional weights on the impurity-derived states (~6 eV flat
  levels and their hybridized partners) while host bands stay at
  weight 1. Runtime: a few seconds.

OPTIONS (reproduce.py)
  --hr PATH            wannier90_hr.dat to read (default data/example_hr.dat)
  --engine {auto,builtin,minimulti}
  --prefix NAME        file prefix for the minimulti route (default example)
  --defect             apply the impurity to the synthetic model
  --npoints INT        k-points along the path (default 200)
  --output FILE        output figure
  --regenerate-hr      rewrite the bundled synthetic data/example_hr.dat

PRODUCING A REAL wannier90_hr.dat
  Run Wannier90 on your supercell (for SrTiO3: e.g. a 2x2x2 supercell
  of the 5-atom cell, or the Pnma 20-atom cell as in the published
  example) and request the Hamiltonian in real space; the annotated
  sketch in inputs/seedname.win.example shows the essential keys
  (num_wann, projections Ti:d O:p, mp_grid, write_hr = true). After
  wannier90.x seedname you get seedname_hr.dat; run this bundle's
  script with --hr pointing at it (labels matching its number of
  Wannier functions; centres are optional but useful). The route is
  generic for any Wannier-derived (or other) tight-binding model
  exposing .atoms, ._orb and .solve_all(k_list=..., eig_vectors=...).
