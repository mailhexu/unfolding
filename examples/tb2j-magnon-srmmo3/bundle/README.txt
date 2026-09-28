====================================================================
TB2J magnons SrMnO3 -- reproduction bundle (tb2j-magnon-srmmo3)
====================================================================

Companion bundle of the "TB2J magnons SrMnO3" example
(https://mailhexu.github.io/unfolding/examples/tb2j-magnon-srmmo3/).
Downfolds magnon band structures computed from TB2J exchange parameters
onto the primitive magnon Brillouin zone: the antiferromagnetic DFT
cell (sqrt(2) x sqrt(2) x sqrt(2) G-AFM SrMnO3, 10 atoms) is larger
than the chemical primitive cell (5-atom pseudo-cubic, one Mn), so the
magnon bands fold and unfolding assigns each folded branch its
primitive-cell momentum.

Pipeline
--------
  DFT (any TB2J-supported code) -> TB2J exchange parameters in the AFM
  phase -> magnon bands on the supercell BZ -> downfold onto the
  primitive-cell magnon BZ:

  SrMnO3 VASP/Wannier90 DFT (G-AFM)
    -> TB2J pickle: JR tensors, 10-atom sqrt2 x sqrt2 x sqrt2 cell, 2 Mn
    -> TB2J Magnon: BdG magnon bands (collinear reference, Q = 0)
    -> unfolding: weights on the 5-atom pseudo-cubic primitive cell

BUNDLE LAYOUT
  unfold.toml                 config for the `unfolding` command
  reproduce.py                renders the figure through both documented
                              routes and checks the physics seals
  data/TB2J_results/TB2J.pickle   TB2J exchange parameters (JR tensors)
                              from a VASP/Wannier90 G-AFM run (10 atoms,
                              2 Mn at +/-2.81 muB, propagation Q=(1/2,1/2,1/2))

PREREQUISITES
  - Python 3.11+ with numpy, matplotlib, ase
  - pip install "unfolding[tb2j]" (TB2J)

STRUCTURE AND PATH
  - supercell (unfold) matrix M = [[0,1,1],[1,0,1],[1,1,0]]: rows are
    the AFM-cell axes in pseudo-cubic units, A_afm = M @ A_pc
  - the primitive cell is derived as inv(M) @ TB2J cell; the q-path
    Gamma-X-M-Gamma-R (200 points) is resolved on that cell with
    Setyawan-Curtarolo special points
  - the reference defaults to the collinear two-sublattice frame
    (Q = 0, quantization axis z, moments from the pickle); spiral
    references are not supported in v1

HOW TO RUN
  From the unpacked bundle directory, either the CLI:

    unfolding --config unfold.toml        # writes magnon_unfolded_cli.png
    # equivalent explicit flags:
    unfolding magnon --results data/TB2J_results \
        --unfold-mat 0 1 1 1 0 1 1 1 0 --special-points GXMGR \
        --npts 200 --output magnon_unfolded_cli.png

  or the legacy single-route command:

    unfolding-magnon data/TB2J_results --unfold-mat 0 1 1 1 0 1 1 1 0 \
        --kpath GXMGR --npts 200 --output magnon_unfolded.png

  or the reproduction script (Python API route + physics checks):

    python reproduce.py                   # writes srmmo3_unfolded_api.png
                                          # and srmmo3.png / srmmo3.json

EXPECTED OUTPUT
  Weight-coded magnon bands along Gamma-X-M-Gamma-R (line opacity =
  unfolded weight per primitive momentum; meV, dashed 0 line).
  reproduce.py verifies a Goldstone mode at Gamma (< 0.05 meV) and
  binary unfolding weights (0 or 1 to 1e-8) for the pristine G-AFM.
  Runtime: seconds on a laptop.
