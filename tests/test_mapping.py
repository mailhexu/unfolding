"""RelabelMap: supercell orbital -> (normal-cell orbital, cell offset) map.

Contract tests for unfolding/mapping.py (story 006).
"""
import numpy as np
import pytest
from ase import Atoms

from unfolding.mapping import RelabelMap, RelabelMapError


def prim_single():
    return Atoms("Si", positions=[(0, 0, 0)], cell=np.eye(3))


def sc_222(atoms=None):
    a = atoms or prim_single()
    return a.repeat((2, 2, 2))


def test_pristine_222_bijection():
    sc = sc_222()
    rm = RelabelMap.from_atoms(sc, prim_single(), np.eye(3) * 2)
    assert len(rm.orb_to_m) == 8
    assert np.all(rm.orb_to_m == 0)  # one orbital per atom, single prim atom
    assert rm.offsets.shape == (8, 3)
    # offsets are the 8 distinct triples {0,1}^3
    assert len({tuple(r) for r in rm.offsets}) == 8
    assert set(map(tuple, rm.offsets)) == {(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    # bijection: rep_orbital is the inverse of (orb_to_m, orb_to_cell)
    for o in range(8):
        assert rm.rep_orbital[rm.orb_to_m[o], rm.orb_to_cell[o]] == o
    # every (m, cell) slot used exactly once
    slots = [(rm.orb_to_m[o], rm.orb_to_cell[o]) for o in range(8)]
    assert len(set(slots)) == 8


def test_multi_atom_prim_cell():
    prim = Atoms("SiO", positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=np.eye(3))
    sc = prim.repeat((2, 1, 1))
    rm = RelabelMap.from_atoms(sc, prim, np.eye(3)[[0]] * np.array([2, 1, 1]) if False else np.diag([2, 1, 1]))
    assert len(rm.orb_to_m) == 4
    # Si atoms -> prim orbital 0, O atoms -> prim orbital 1
    assert list(rm.orb_to_m) == [0, 1, 0, 1]
    assert len(rm.offsets) == 2


def test_vacancy_raises():
    sc = sc_222()
    del sc[3]
    with pytest.raises(RelabelMapError):
        RelabelMap.from_atoms(sc, prim_single(), np.eye(3) * 2)


def test_species_mismatch_raises():
    sc = sc_222()
    sc[5].symbol = "Ge"
    with pytest.raises(RelabelMapError):
        RelabelMap.from_atoms(sc, prim_single(), np.eye(3) * 2)


def test_displacement_within_tolerance_matches():
    prim = prim_single()
    sc = sc_222()
    tol = 0.04
    # displace atom 0 of the supercell by 0.5 * tol along x (Cartesian)
    sc.positions[0, 0] += 0.5 * tol
    rm = RelabelMap.from_atoms(sc, prim, np.eye(3) * 2, tol_r=tol)
    # same (m, offset) classification as the pristine map for atom 0
    ref = RelabelMap.from_atoms(sc_222(), prim, np.eye(3) * 2, tol_r=tol)
    assert rm.orb_to_m[0] == ref.orb_to_m[0]
    assert np.array_equal(rm.orb_to_r0[0], ref.orb_to_r0[0])


def test_displacement_beyond_tolerance_raises():
    prim = prim_single()
    sc = sc_222()
    tol = 0.04
    sc.positions[0, 0] += 2.0 * tol
    with pytest.raises(RelabelMapError):
        RelabelMap.from_atoms(sc, prim, np.eye(3) * 2, tol_r=tol)


def test_wraparound_positions():
    # prim atom at fractional 0.98 along x (inside the primitive cell)
    prim = Atoms("Si", positions=[(0, 0, 0)], cell=np.eye(3))
    prim.positions[0] = np.array([0.98, 0.0, 0.0]) @ np.eye(3)
    sc = prim.repeat((2, 1, 1))
    # sc atom 1 sits at fractional 1.98 in primitive coordinates:
    # floor -> cell offset 1, fractional part 0.98 matches the prim atom
    rm = RelabelMap.from_atoms(sc, prim, np.diag([2, 1, 1]))
    assert rm.orb_to_m[1] == 0
    assert tuple(rm.orb_to_r0[1]) == (1, 0, 0)
    # a slightly negative position wraps to the last cell without raising
    sc2 = prim.repeat((2, 1, 1))
    sc2.positions[0, 0] -= 0.01  # 0.97 -> still well inside
    rm2 = RelabelMap.from_atoms(sc2, prim, np.diag([2, 1, 1]))
    assert tuple(rm2.orb_to_r0[0]) == (0, 0, 0)
    # negative fractional coordinate: atom pushed just below 0 belongs to cell -1
    sc3 = Atoms(["Si", "Si"], positions=[[0.98, 0, 0], [-0.02, 0, 0]], cell=np.diag([2, 1, 1]))
    rm3 = RelabelMap.from_atoms(sc3, prim, np.diag([2, 1, 1]))
    assert tuple(rm3.orb_to_r0[1]) == (-1, 0, 0)
    assert rm3.orb_to_m[1] == 0


def test_orbital_counts_expansion_and_mismatch():
    prim = Atoms("SiO", positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=np.eye(3))
    sc = prim.repeat((2, 1, 1))
    counts_prim = [4, 2]  # Si carries 4 orbitals, O carries 2
    counts_sc = [4, 2, 4, 2]
    rm = RelabelMap.from_atoms(
        sc, prim, np.diag([2, 1, 1]),
        orb_counts_sc=counts_sc, orb_counts_prim=counts_prim,
    )
    assert len(rm.orb_to_m) == 12
    # first Si block (4 orbitals) -> prim orbitals 0..3; first O block -> 4..5
    assert list(rm.orb_to_m[:4]) == [0, 1, 2, 3]
    assert list(rm.orb_to_m[4:6]) == [4, 5]
    # second copy of the same cell translated by (1,0,0)
    assert list(rm.orb_to_m[6:10]) == [0, 1, 2, 3]
    assert np.array_equal(rm.orb_to_r0[6], [1, 0, 0])
    assert rm.rep_orbital.shape == (6, 2)

    # per-atom count mismatch between matched atoms raises
    bad_sc = [4, 3, 4, 2]
    with pytest.raises(RelabelMapError):
        RelabelMap.from_atoms(
            sc, prim, np.diag([2, 1, 1]),
            orb_counts_sc=bad_sc, orb_counts_prim=counts_prim,
        )
    # wrong total count raises too
    with pytest.raises(RelabelMapError):
        RelabelMap.from_atoms(
            sc, prim, np.diag([2, 1, 1]),
            orb_counts_sc=[4, 2, 4, 2, 1], orb_counts_prim=counts_prim,
        )


def test_supercell_matrix_inconsistency_raises():
    # claim 2x2x2 but supply a 2x1x1 supercell
    sc = prim_single().repeat((2, 1, 1))
    with pytest.raises(RelabelMapError):
        RelabelMap.from_atoms(sc, prim_single(), np.eye(3) * 2)


def test_same_determinant_wrong_matrix_raises():
    # same |det| = 2 but replication along the wrong axis
    sc = prim_single().repeat((2, 1, 1))  # cell diag(2, 1, 1)
    with pytest.raises(RelabelMapError):
        RelabelMap.from_atoms(sc, prim_single(), np.diag([1, 2, 1]))


def test_displacement_exactly_at_tolerance_matches():
    prim = prim_single()
    tol = 0.04
    sc = sc_222()
    sc.positions[0, 1] += tol  # exactly at the tolerance boundary
    rm = RelabelMap.from_atoms(sc, prim, np.eye(3) * 2, tol_r=tol)
    ref = RelabelMap.from_atoms(sc_222(), prim, np.eye(3) * 2, tol_r=tol)
    assert rm.orb_to_m[0] == ref.orb_to_m[0]
    assert np.array_equal(rm.orb_to_r0[0], ref.orb_to_r0[0])


def test_displacement_across_cell_face_keeps_offset():
    # primitive atom near a cell face; its cell-1 copy displaced past the
    # x=2 supercell boundary still labels as offset 1 (copy identity),
    # not as offset 2 with a wrapped minimum image
    prim = Atoms("Si", positions=[[0.98, 0.0, 0.0]], cell=np.eye(3))
    sc = Atoms(
        ["Si", "Si"],
        positions=[[0.98, 0, 0], [2.01, 0, 0]],  # copy 1 at 1.98 + 0.03
        cell=np.diag([2, 1, 1]),
    )
    rm = RelabelMap.from_atoms(sc, prim, np.diag([2, 1, 1]), tol_r=0.04)
    assert rm.orb_to_m[1] == 0
    assert tuple(rm.orb_to_r0[1]) == (1, 0, 0)  # copy identity preserved


def test_skew_cell_minimum_image():
    # triclinic primitive cell: minimum image is not the per-component round
    cell = np.array(
        [[1.0, 0.0, 0.0], [0.9, 1.0, 0.0], [0.0, 0.3, 1.0]]
    )
    prim = Atoms("Si", positions=[[0.1, 0.1, 0.1]], cell=cell)
    sc = prim.repeat((2, 2, 2))
    rm = RelabelMap.from_atoms(sc, prim, np.eye(3) * 2, tol_r=0.04)
    assert len(rm.orb_to_m) == 8
    assert np.all(rm.orb_to_m == 0)


def test_hamiltonio_orb_dict_form():
    # HamiltonIO-style orb_dict: {atom index: [orbital names]}
    prim = Atoms("Si", positions=[(0, 0, 0)], cell=np.eye(3))
    sc = prim.repeat((2, 1, 1))
    rm = RelabelMap.from_atoms(
        sc, prim, np.diag([2, 1, 1]),
        orb_counts_sc={0: ["2s", "2p_x"], 1: ["2s", "2p_x"]},
        orb_counts_prim={0: ["2s", "2p_x"]},
    )
    assert len(rm.orb_to_m) == 4
    assert list(rm.orb_to_m) == [0, 1, 0, 1]
    assert rm.representatives.shape == (2,)
    # canonical cell = offsets[0] = (0,0,0): its orbitals are 0 and 1
    assert list(rm.representatives) == [0, 1]


def test_story2_toy_fixture_mapping():
    # the shared story-2 toy: one atom, two orbitals, 1D chain a = 1
    from unfolding.toy_models import toy_lcao_model

    model = toy_lcao_model()  # 2x2 S(k): one atom carrying two orbitals
    assert model["S_of_k"](0.0).shape == (2, 2)

    a = 1.0
    prim = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[[a, 0, 0], [0, 8.0, 0], [0, 0, 8.0]])
    sc = prim.repeat((4, 1, 1))
    rm = RelabelMap.from_atoms(
        sc, prim, np.diag([4, 1, 1]),
        orb_counts_sc={i: ["2s", "2p_x"] for i in range(4)},
        orb_counts_prim={0: ["2s", "2p_x"]},
    )
    assert len(rm.orb_to_m) == 8  # 4 cells x 2 orbitals
    # each supercell orbital maps back to one of the two prim orbitals
    assert set(rm.orb_to_m) == {0, 1}
    assert np.array_equal(rm.orb_to_r0[2], [1, 0, 0])
    assert rm.rep_orbital.shape == (2, 4)
    # bijection at the orbital level
    slots = [(rm.orb_to_m[o], rm.orb_to_cell[o]) for o in range(8)]
    assert len(set(slots)) == 8
