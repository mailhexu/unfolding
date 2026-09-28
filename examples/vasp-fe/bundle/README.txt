====================================================================
VASP PAW bcc-Fe unfolding -- reproduction bundle (vasp-fe)
====================================================================
Docs example: docs/content/examples/vasp-paw.md in the unfolding
repository. The VASP WAVECAR/POTCAR route: a spin-polarized bcc-Fe
2x2x2 conventional supercell (16 atoms) unfolded onto the 1-atom bcc
primitive cell along Gamma-H-N-Gamma-P-H, with the PAW overlap metric
S = I + sum_aij |p_ai> dS_aij <p_aj| built from the POTCAR projector
tables and the primitive reference wavefunctions embedded into the
supercell G basis.

VASP's WAVECAR and the PAW_PBE Fe POTCAR are NOT in this bundle and can
never be redistributed: the POTCAR is licensed, and the fixture
WAVECARs are ~1 GB. Two reproduction levels:

  1. python reproduce.py --snapshot
     Redraws the committed figure vasp_fe_path.png from the shipped
     data/vasp_fe_path_result.npz -- the plain-array output (unfolded
     weights, eigenvalues, primitive reference bands, E_F values) of
     the fixture run. No VASP, no POTCAR.

  2. Full pipeline (your own VASP + POTCAR):
       mkdir -p runs && cp -r inputs/* runs/
       cd runs/sc16_scf  && cp <your POTCAR> POTCAR && vasp_std
       cd ../prim_scf    && cp <your POTCAR> POTCAR && vasp_std
       cd ../sc16_nscf   && cp <your POTCAR> POTCAR && vasp_std
       cd ../prim_nscf   && cp <your POTCAR> POTCAR && vasp_std
       cd ../.. && cp <your POTCAR> POTCAR
       python reproduce.py --potcar POTCAR      # or: unfolding --config unfold.toml

BUNDLE LAYOUT
  reproduce.py             snapshot + full-pipeline driver
  make_inputs.py           regenerates inputs/ (POSCAR/KPOINTS/INCAR)
  read_wavecar_ordered.py  WAVECAR reader enumerating G-lists in VASP's
                           stored-k frame (KPAR writer layout)
  wrap_wavecar.py          rewraps stored k-points so pymatgen accepts
                           the file (any KPAR run writes unwrapped k)
  unfold.toml              config for the `unfolding` CLI (spin-up
                           route): unfolding --config unfold.toml
  inputs/                  the four VASP run directories (text only)
                           sc16_scf, prim_scf, sc16_nscf, prim_nscf
  data/vasp_fe_path_result.npz   committed unfolding output arrays

PREREQUISITES
  - Snapshot mode: numpy, matplotlib and the unfolding package
    importable (pip install -e <unfolding repo checkout>, or
    PYTHONPATH=<unfolding repo checkout>).
  - Full pipeline: VASP (6.x) with a PAW_PBE Fe POTCAR (06Sep2000,
    private), pymatgen, HamiltonIO. The POTCAR is NEVER shipped or
    stored by any script here.

STRUCTURES
  Primitive cell: 1-atom bcc, a = 2.866 A,
      PRIM = [[-1,1,1],[1,-1,1],[1,1,-1]] * a/2.
  Supercell: 2x2x2 conventional (16 atoms), supercell = M @ primitive
      with M = 2*[[0,1,1],[1,0,1],[1,1,0]] (det 16).
  Run settings: ISPIN=2, ENCUT 300 eV, Gaussian smearing 0.05 eV,
      MAGMOM 16*2.2, NBANDS 96 in both banks (>= 82 occupied majority
      bands plus margin; supercell and primitive banks must share
      NBANDS so resolved weights pair index-by-index). SCF: ICHARG=2,
      4x4x4 (supercell) / 12x12x12 (primitive) gamma-centered meshes.
      Path NSCF: ICHARG=11, ISYM=-1, 250-point explicit KPOINTS lists
      (weights required by this build); KPAR=8 for the supercell.

K-PATH
  Gamma-H-N-Gamma-P-H, 250 points from ase bandpath on the primitive
  bcc cell; the supercell deck lists the same points mapped as
  K_sc = k_prim @ M.T. Energies in eV relative to each run's own SCF
  Fermi level (OUTCAR E-fermi), window E_F +/- 8 eV.

PARAMETERS THAT MATTER
  matrix               M above, row convention supercell = M @ primitive;
                       verified against the two POSCAR lattices.
  potcar               the matching licensed POTCAR; read_potcar_paw
                       pulls its reciprocal projector tables. The
                       primitive reference may be pristine bcc Fe while
                       the supercell is anything commensurate: overlaps
                       are evaluated in the supercell PAW metric.
  resolve_degenerate   1e-3 eV: eigen-assigns gauge-invariant branch
                       weights inside near-degenerate groups.
  spin                 0 = up, 1 = down (both drawn side by side).

EXPECTED OUTPUT
  vasp_fe_path.png (200 dpi): two panels (spin up/down), Gaussian-
  smeared spectral-weight map (Blues) with primitive-cell bands
  overlaid in crimson and E_F = 0 dashed. The full pipeline also
  prints PAW norm residuals per spin (fixture: ~1e-6).

KPAR / READER NOTES
  Any KPAR makes this VASP build write single-precision (rtag 45200)
  WAVECARs with unwrapped k-coordinates. read_wavecar_ordered.py
  handles that layout directly (it enumerates each G list around the
  stored k in VASP's fold-lexicographic order); wrap_wavecar.py is the
  alternative fix if you read through pymatgen. KPAR=16 crashed this
  build; KPAR=8 is the validated choice for the supercell NSCF.
