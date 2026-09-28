====================================================================
SIESTA spinors (nspin=4) -- reproduction bundle (siesta-spinor)
====================================================================

Regenerates the figure of the "SIESTA spinors" example
(https://mailhexu.github.io/unfolding/examples/siesta-spinor/): the
8-atom conventional-cubic Si supercell run non-collinear
(SpinPolarized true + Spin.Orbit true, nspin=4) unfolded onto the
2-atom primitive fcc cell along Gamma-X-W-Gamma-L-W-X. The scalar Si
pseudopotential has no SOC channels, so the spinor bands equal the
scalar bands with Kramers degeneracy -- the figure certifies the
spinor machinery end to end (parse -> relabel -> weights), not spinor
physics.

BUNDLE LAYOUT
  reproduce.py   self-contained script; regenerates the published figure
  data/          SIESTA fixtures, parsed in place:
                   si_prim_pso.fdf + si_prim_pso.HSX   primitive cell
                   si_sc_pso.fdf   + si_sc_pso.HSX     supercell
                   si_prim_pso.vasp                    primitive POSCAR
  pseudos/       pseudopotential notes (see PSEUDOPOTENTIALS below)

PREREQUISITES
  - Python 3.11+ with numpy, scipy, matplotlib, sisl, HamiltonIO
  - the unfolding package: pip install unfolding

STRUCTURE AND PATH
  - same cell and path as the collinear example: M = [[-1,1,1],[1,-1,1],
    [1,1,-1]] (conventional = M @ primitive), Gamma-X-W-Gamma-L-W-X,
    300 points, fcc primitive-frame points
  - spinor orbital counts are doubled: 8 per atom (4 PAO x 2 spin
    components); they are derived automatically from the parsed model,
    and the same weight pipeline as the collinear route applies to the
    doubled 64-orbital basis

HOW TO RUN
  From the unpacked bundle directory, either the CLI:

    unfolding --config unfold.toml        # writes si_spinor_unfolded_cli.png
    # equivalent explicit flags:
    unfolding siesta --fdf data/si_sc_pso.fdf --primitive data/si_prim_pso.vasp \
        --unfold-mat -1 1 1 1 -1 1 1 1 -1 --special-points GXWGLX \
        --npts 300 --method ideal --output si_spinor_unfolded_cli.png

  or the Python reproduction script (adds the primitive spinor-cell
  overlay of the published figure):

    python reproduce.py                   # writes si_spinor_unfolded.png
    python reproduce.py out.png           # or any output path

  Runtime: a few seconds on one core.

FIXTURE PROVENANCE
  Same 8-atom Si setup as the collinear example run non-collinear with
  a scalar Si pseudopotential (spin-orbit coupling off). Spinor
  eigenvalues are Fermi-shifted by the writing run, as usual in SIESTA.

PSEUDOPOTENTIALS
  The fdf files name Si.psf (norm-conserving, SIESTA distribution
  Examples/Si_Optical set), which is NOT included and is NOT needed to
  reproduce the figure (the shipped .HSX carries the Hamiltonian and
  overlap). See pseudos/PLACEHOLDER-NOT-INCLUDED.txt.
