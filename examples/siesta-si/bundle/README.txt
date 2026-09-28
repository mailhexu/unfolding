====================================================================
SIESTA Si unfolded bands -- reproduction bundle (siesta-si)
====================================================================

Regenerates the figure of the "SIESTA Si bands" example
(https://mailhexu.github.io/unfolding/examples/siesta-si/): the bundled
8-atom conventional-cubic Si supercell (HSX from a 2x2x2 k-grid SCF)
unfolded onto the 2-atom primitive fcc cell along the high-symmetry path
Gamma-X-W-Gamma-L-W-X, with the independently computed primitive-cell
bands overlaid in crimson. Every weight-1 unfolded branch lies on a
primitive band, which is the practical validation of the unfolding.

BUNDLE LAYOUT
  unfold.toml    config for the `unfolding` command (see CLI below)
  reproduce.py   self-contained script; regenerates the published figure
  data/          SIESTA fixtures, parsed in place:
                   si_prim.fdf + si_prim.HSX   primitive cell (2 atoms)
                   si_sc.fdf   + si_sc.HSX     supercell   (8 atoms)
                   si_prim.vasp                primitive cell as a POSCAR
                 Each fdf sits next to its .HSX because the parser
                 resolves the Hamiltonian archive relative to the fdf.
  pseudos/       pseudopotential notes (see PSEUDOPOTENTIALS below)

PREREQUISITES
  - Python 3.11+ with numpy, scipy, matplotlib, sisl, HamiltonIO
    (pip install HamiltonIO sisl)
  - the unfolding package: pip install unfolding
    (or unpack the sdist/repository and put it on PYTHONPATH)

STRUCTURE AND PATH
  - supercell matrix M = [[-1,1,1],[1,-1,1],[1,1,-1]] with the row
    convention conventional cell = M @ primitive cell (8 = det M atoms)
  - k-path Gamma-X-W-Gamma-L-W-X, 300 points, fcc special points in
    primitive reciprocal fractional coordinates (Setyawan-Curtarolo)
  - weight: "ideal" (Popescu-Zunger/Lee) along the generic path

HOW TO RUN
  From the unpacked bundle directory (the directory containing this
  README), either the CLI:

    unfolding --config unfold.toml        # writes si_unfolded_cli.png
    # equivalent explicit flags:
    unfolding siesta --fdf data/si_sc.fdf --primitive data/si_prim.vasp \
        --unfold-mat -1 1 1 1 -1 1 1 1 -1 --special-points GXWGLX \
        --npts 300 --method ideal --output si_unfolded_cli.png

  or the Python reproduction script (adds the crimson primitive-cell
  overlay of the published figure):

    python reproduce.py                   # writes si_unfolded.png
    python reproduce.py out.png           # or any output path

  or the equivalent Python API from the bundle root:

    import numpy as np
    from ase.io import read
    from unfolding import unfold_siesta
    ax = unfold_siesta(
        fdf="data/si_sc.fdf", prim_atoms=read("data/si_prim.vasp"),
        unfold_sc_mat=np.array([[-1,1,1],[1,-1,1],[1,1,-1]]),
        kpts=..., knames=["$\\Gamma$","X","W","$\\Gamma$","L","W","X"],
        method="ideal",
    )

EXPECTED OUTPUT
  Weight-coded unfolded bands in eV with tick labels Gamma X W Gamma L
  W X; the CLI/TOML route writes si_unfolded_cli.png, reproduce.py
  writes si_unfolded.png (with the primitive-cell bands overlaid in
  crimson, labelled in the legend). Runtime: a few seconds on one core.

FIXTURE PROVENANCE
  SIESTA >= 5.4, GGA/PBE, SZ PAO basis, MeshCutoff 100 Ry, SaveHS true.
  si_prim: 2-atom fcc primitive cell, 4x4x4 k-grid SCF. si_sc: the
  8-atom conventional cubic supercell with a 2x2x2 k-grid SCF (matched
  to the primitive sampling). A k-sampled SCF is essential: a Gamma-only
  SaveHS run collapses all supercell image shells into a single R=0
  block, which cannot be unfolded at generic momenta. To regenerate the
  fixtures from scratch you need SIESTA and the pseudopotential below.

PSEUDOPOTENTIALS
  The fdf files name Si.psf (Si, norm-conserving). That file is NOT
  included in this bundle and is NOT needed to reproduce the figure:
  the shipped .HSX already carries the Hamiltonian and overlap, and the
  parse/unfolding path never opens the pseudo. You only need Si.psf to
  re-run SIESTA from scratch; it comes from the SIESTA distribution's
  Examples/Si_Optical set. See pseudos/PLACEHOLDER-NOT-INCLUDED.txt.
