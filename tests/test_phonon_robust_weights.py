"""Gauge-robust phonon unfolding weights (Cu_fcc DDB fractional-weights fix).

Regression background: ``phonon_unfolder.get_weights`` is a bare translation
character sum that assumes eigenvectors in a pure fold-label gauge. anaddb
stores Bloch-gauge eigenvectors and, on mirror-symmetric q-paths where the
dynamical matrix is real, arbitrary (real, cosine-like) mixtures of the
degenerate fold sectors. The character sum then returns ~0.5 for the correct
modes of a pristine supercell — the symptom reported for examples/Cu_fcc.

``get_weights_robust`` projects onto Bloch sums over the translation group
and diagonalizes the Hermitian weight matrix within degenerate frequency
groups, which is invariant under any within-group unitary gauge.
"""
import shutil
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ase import Atoms
from unfolding.phonon_unfolder import phonon_unfolder

REPO_ROOT = Path(__file__).resolve().parents[1]

SC_MAT = np.linalg.inv(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2.0)
CONV_CELL = Atoms(
    "Cu4",
    scaled_positions=[(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
    cell=np.eye(3),
    pbc=True,
)
FOLD_CLASSES = [np.array(g, dtype=float) for g in
                ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])]
QPOINTS = np.array([[0.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.13, 0.29, 0.41]])


def _block_unitary(rng, sizes):
    """Deterministic block-diagonal unitary with the given block sizes."""
    blocks = []
    for n in sizes:
        if n == 1:
            phase = np.exp(2j * np.pi * rng.random())
            blocks.append(np.array([[phase]]))
        else:
            m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            q, r = np.linalg.qr(m)
            q = q * (np.diag(r) / np.abs(np.diag(r)))
            blocks.append(q)
    n_all = sum(sizes)
    U = np.eye(n_all, dtype=complex)
    i = 0
    for b in blocks:
        U[i:i + b.shape[0], i:i + b.shape[0]] = b
        i += b.shape[0]
    return U


def _build_synthetic(kind, rng):
    """Return (freqs, evecs) for pristine folded modes in a chosen gauge.

    kind: 'bloch'  — pure Bloch-gauge eigenvectors e_i ~ exp(2 pi i k.R_i)
          'fold'   — phonopy-style fold gauge (q folded out)
          'mixed'  — Bloch gauge followed by a random unitary inside each
                     degenerate frequency group (arbitrary diagonalizer gauge,
                     includes the real cosine mixtures anaddb produces)
    """
    uf = phonon_unfolder(CONV_CELL, SC_MAT,
                         np.zeros((3, 12, 12), dtype=complex), QPOINTS)
    R = np.array(uf._trans_rs)
    idx = uf._trans_indices
    # translation mapping home atom 0 to each supercell atom
    atom_R = {}
    for k in range(len(R)):
        atom_R[int(idx[k, 0]) // 3] = R[k]
    assert len(atom_R) == 4

    freqs = np.zeros((3, 12))
    evecs = np.zeros((3, 12, 12), dtype=complex)
    for iq, q in enumerate(QPOINTS):
        cols = []
        fs = []
        for m, g in enumerate(FOLD_CLASSES):
            k = q + g
            u = _block_unitary(rng, [1, 1, 1])  # orthonormal 3x3
            for b in range(3):
                col = np.zeros(12, dtype=complex)
                for atom, Rv in atom_R.items():
                    col[3 * atom:3 * atom + 3] = (
                        u[:, b] * np.exp(2j * np.pi * np.dot(k, Rv)))
                cols.append(col / np.linalg.norm(col))
                # sigma-invariant, sector-pairwise-equal frequencies
                fs.append(1.0 + 0.31 * np.cos(2 * np.pi * k[1])
                          + 0.22 * np.cos(2 * np.pi * k[0]) * np.cos(2 * np.pi * k[2])
                          + 0.17 * b)
        E = np.array(cols).T  # (comp, branch)
        f = np.array(fs)
        if kind == "fold":
            phase = np.array([np.exp(-2j * np.pi * np.dot(q, atom_R[a]))
                              for a in range(4)])
            phase = np.repeat(phase, 3)
            E = E * phase[:, None]
        if kind == "mixed":
            order = np.argsort(f)
            # group by equal frequency
            fsorted = f[order]
            grp = np.zeros(12, dtype=int)
            for j in range(1, 12):
                grp[j] = grp[j - 1] + (1 if fsorted[j] - fsorted[j - 1] > 1e-9 else 0)
            sizes = [int((grp == gg).sum()) for gg in np.unique(grp)]
            U = _block_unitary(rng, sizes)
            Eo = E[:, order] @ U
            E = np.zeros_like(E)
            E[:, order] = Eo
        evecs[iq] = E
        freqs[iq] = f
    return freqs, evecs


def _bres(W):
    return float(np.minimum(np.abs(W), np.abs(W - 1.0)).max())


def test_bare_character_sum_is_fractional_on_mixed_gauge():
    """The reported Cu_fcc symptom: pristine weights are neither 0 nor 1."""
    rng = np.random.default_rng(7)
    freqs, evecs = _build_synthetic("mixed", rng)
    uf = phonon_unfolder(CONV_CELL, SC_MAT, evecs, QPOINTS, phase=False)
    W = uf.get_weights()
    # generic q-points: the character sum cannot resolve the mixed sectors
    for iq in (1, 2):
        assert _bres(W[iq]) > 0.1, W[iq]


def test_robust_weights_binary_for_arbitrary_degenerate_gauge():
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        freqs, evecs = _build_synthetic("mixed", rng)
        uf = phonon_unfolder(CONV_CELL, SC_MAT, evecs, QPOINTS, phase=False)
        W = uf.get_weights_robust(freqs, gauge="bloch")
        assert _bres(W) < 1e-8
        # exactly the three modes of the identity target sector carry weight 1
        assert (W > 0.99).sum(axis=1).tolist() == [3, 3, 3]


def test_robust_weights_binary_for_pure_bloch_gauge():
    rng = np.random.default_rng(11)
    freqs, evecs = _build_synthetic("bloch", rng)
    uf = phonon_unfolder(CONV_CELL, SC_MAT, evecs, QPOINTS, phase=False)
    W = uf.get_weights_robust(freqs, gauge="bloch")
    assert _bres(W) < 1e-8
    assert (W > 0.99).sum(axis=1).tolist() == [3, 3, 3]


def _group_sums(W, freqs, tol=1e-9):
    """Sorted per-degenerate-group weight sums (gauge-invariant per group)."""
    out = []
    for iq in range(len(freqs)):
        order = np.argsort(freqs[iq])
        f = freqs[iq][order]
        grp = np.zeros(len(f), dtype=int)
        for j in range(1, len(f)):
            grp[j] = grp[j - 1] + (1 if f[j] - f[j - 1] > tol else 0)
        w = W[iq][order]
        out.append(sorted(float(w[grp == gg].sum()) for gg in np.unique(grp)))
    return out


def test_robust_fold_gauge_matches_shipped_character_sum():
    rng = np.random.default_rng(13)
    freqs, evecs = _build_synthetic("fold", rng)
    uf = phonon_unfolder(CONV_CELL, SC_MAT, evecs, QPOINTS, phase=False)
    W_robust = uf.get_weights_robust(freqs, gauge="fold")
    W_shipped = uf.get_weights()
    # per-branch values may differ only inside degenerate groups (gauge);
    # the per-group sums must agree exactly
    flat = lambda W: [v for row in _group_sums(W, freqs) for v in row]
    assert np.allclose(flat(W_robust), flat(W_shipped), atol=1e-10)
    assert _bres(W_robust) < 1e-8

def test_real_phonopy_fixture_robust_weights_binary(cu_pristine):
    uf = cu_pristine["unfolder"]
    freqs = cu_pristine["frequencies"]
    W_robust = uf.get_weights_robust(freqs, gauge="fold")
    W_shipped = uf.get_weights()
    assert _bres(W_robust) < 1e-8
    assert ((W_robust > 0.99).sum(axis=1) == 3).all()
    # the shipped character sum is already correct on this pure gauge; the
    # two methods may only redistribute weight inside degenerate groups
    flat = lambda W: [v for row in _group_sums(W, freqs) for v in row]
    assert np.allclose(flat(W_robust), flat(W_shipped), atol=1e-10)


def _anaddb_available():
    if shutil.which("anaddb"):
        return True
    return Path("/phythema/Abinit_versions/abinit_git/build/ubuntu2204_cmake_pymb"
                "/src/98_main/anaddb").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("abipy") is None or not _anaddb_available(),
    reason="abipy and a working anaddb are required for the real Cu DDB seal",
)
def test_real_cu_ddb_weights_are_binary():
    """End-to-end seal on examples/Cu_fcc/out_DDB (the reported example)."""
    import abipy.abilab as abilab
    from ase.data import atomic_masses
    from ase.build import bulk
    from ase.dft.kpoints import get_special_points
    from unfolding.DDB_unfolder import displacement_cart_to_evec

    DDB = abilab.abiopen(str(REPO_ROOT / "examples" / "Cu_fcc" / "out_DDB"))
    struct = DDB.structure
    numbers = struct.atomic_numbers
    masses = [atomic_masses[Z] for Z in numbers]
    scaled = struct.frac_coords
    pa = bulk("Cu", "fcc")
    pts = get_special_points(pa.cell, eps=0.01)
    kpath_bounds = [np.array(pts[k]) @ SC_MAT for k in "GXWGL"]
    phbst, _ = DDB.anaget_phbst_and_phdos_files(
        nqsmall=2, asr=1, chneut=1, dipdip=1, verbose=0, ndivsm=40,
        lo_to_splitting=True, qptbounds=kpath_bounds)
    qpoints = np.array(phbst.qpoints.frac_coords)
    nb = 3 * len(numbers)
    evals = np.zeros((len(qpoints), nb))
    evecs = np.zeros((len(qpoints), nb, nb), dtype=complex)
    for iq, q in enumerate(qpoints):
        for ib in range(nb):
            pm = phbst.get_phmode(q, ib)
            evals[iq, ib] = pm.freq
            evecs[iq, :, ib] = displacement_cart_to_evec(
                pm.displ_cart, masses, scaled, add_phase=False)
    atoms = struct.to_ase_atoms()
    uf = phonon_unfolder(atoms, SC_MAT, evecs, qpoints, phase=False)
    W = uf.get_weights_robust(evals, gauge="bloch")
    assert _bres(W) < 1e-6
    assert ((W > 0.99).sum(axis=1) == 3).all()
