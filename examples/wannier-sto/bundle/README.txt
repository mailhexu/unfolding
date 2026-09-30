====================================================================
Wannier90 SrTiO3 -- reproduction bundle (wannier-sto)
====================================================================

Companion bundle of the "Wannier90 SrTiO3" example
(https://mailhexu.github.io/unfolding/examples/wannier-sto/). Unfolds
real Wannier90 tight-binding Hamiltonians -- pristine and Ti-vacancy
supercells of cubic SrTiO3 -- onto the 5-atom primitive cell: the
supercell generalized eigenproblem is solved along the k-path and each
supercell state is assigned its primitive-cell spectral weight.

What the example shows
----------------------
For the pristine supercell every band folds from exactly one primitive
momentum, so the unfolding weights are binary: bold branches trace the
primitive bands (O-2p valence complex, Ti-3d conduction manifold with a
1.8 eV gap) and the folded copies stay invisible. In the Ti-vacancy
supercell the translation symmetry is broken: vacancy-derived states
appear with fractional weight (flat branches around 8.3-8.6 eV and at
the conduction-band edge) while the host bands stay at weight 1.

REAL DATASETS (the page's headline figures)
-------------------------------------------
  data/pristine/     pristine sqrt(2)xsqrt(2)x2 supercell of cubic
                     SrTiO3 (a = 3.9 A): 20 atoms, 56 Wannier functions
                     (12 O sites x pz/px/py + 4 Ti sites x
                     dz2/dxy/dyz/dx2/dxz), wannier90.win / .wout /
                     _hr.dat as written by Wannier90
  data/ti_vacancy/   same cell with one Ti site vacant (dummy species
                     "V" in the win); the vacancy-site d shell is kept
                     in the Wannier set, so the supercell has the same
                     56 orbitals and the identical labels/scmat apply

The orbital positions are read from the final WF centres in the .wout
(the shipped _centres.xyz files in the upstream dataset are stale, from
a 76-orbital test run, and are deliberately not shipped). The compressed
bundle is ~7 MB; the raw hr files are 38 MB each.

SYNTHETIC SMOKE FIXTURE
-----------------------
  data/example_hr.dat + data/example.win

A tiny SYNTHETIC Ti t2g stand-in (3 orbitals on a simple-cubic Ti
sublattice, nearest-neighbour hopping t = 1 eV, 2x2x2 supercell, 24
orbitals, wannier90 hr format; --defect weakens one site's hoppings to
30% and adds a +6 eV on-site term). It lets you smoke-test the route in
seconds without the large real files; the published figures come from
the real datasets above, not from this model.

BUNDLE LAYOUT
  unfold.toml                 config for the `unfolding` command (the
                              real pristine dataset; the page shows this
                              file verbatim)
  reproduce.py                runs the unfolding and saves the figure
  data/pristine/...           real pristine Wannier90 output
  data/ti_vacancy/...         real Ti-vacancy Wannier90 output
  data/example_hr.dat         synthetic t2g smoke fixture (+ .win)
  inputs/seedname.win.example annotated Wannier90 input sketch for
                              producing your own wannier90_hr.dat

PREREQUISITES
  - Python 3.11+ with numpy, matplotlib, ase
  - the unfolding package: pip install unfolding
  The route reads wannier90 runs with the package's own Wannier90
  reader (hr + win + WF centres); minimulti is not required.

HOW TO RUN
  From the unpacked bundle directory:

    unfolding --config unfold.toml            # pristine -> sto_unfolded.png
    # Ti-vacancy: change [input] path to data/ti_vacancy and remove
    # [options] resolve_degenerate to retain raw vacancy weights.

    python reproduce.py --real pristine       # -> sto_unfolded.png
    python reproduce.py --real ti_vacancy     # -> sto_ti_vacancy.png

    python reproduce.py                       # synthetic smoke test
    python reproduce.py --defect              # synthetic impurity model

  The real runs parse the 38 MB hr file (~5 s) and solve a 56x56
  eigenproblem at 200 path points; expect under a minute. The synthetic
  runs finish in seconds.

OPTIONS (reproduce.py)
  --real {pristine,ti_vacancy}
                       unfold a bundled real Wannier90 dataset
  --hr PATH            wannier90_hr.dat to read (synthetic route,
                       default data/example_hr.dat)
  --engine {auto,builtin,minimulti}
                       synthetic route engine
  --prefix NAME        file prefix for the minimulti route (default example)
  --defect             apply the impurity to the synthetic model
  --npoints INT        k-points along the path (default 200)
  --output FILE        output figure
  --regenerate-hr      rewrite the bundled synthetic data/example_hr.dat

PRODUCING A REAL wannier90_hr.dat
  Run Wannier90 on your supercell (for SrTiO3: e.g. the
  sqrt(2)xsqrt(2)x2 20-atom cell as here) and request the Hamiltonian
  in real space; the annotated sketch in inputs/seedname.win.example
  shows the essential keys (num_wann, projections Ti:d O:p, mp_grid,
  write_hr = true). After wannier90.x seedname you get seedname_hr.dat;
  point --hr at it (labels matching its number of Wannier functions;
  keep the .win and .wout so the reader finds the unit cell and the WF
  centres). The route is generic for any Wannier-derived (or other)
  tight-binding model exposing .atoms, ._orb and
  .solve_all(k_list=..., eig_vectors=...).
