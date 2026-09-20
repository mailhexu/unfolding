"""Shared production SIESTA fixture helpers for story-010 tests."""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIM_FDF = os.path.join(ROOT, "tests", "data", "si_example", "si_prim.fdf")
B_DIAMOND = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])


def make_si_unfolder(sc_fdf, *, tol_r=0.04):
    from HamiltonIO.siesta.sisl_wrapper import SislParser
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    class RListParser(SislParser):
        def read_Rlist(self, geom=None):
            return self.ham.lattice.sc_off

    prim = RListParser(PRIM_FDF).get_model()
    sc = RListParser(sc_fdf).get_model()
    relabel = RelabelMap.from_atoms(
        sc.atoms,
        prim.atoms,
        B_DIAMOND,
        tol_r=tol_r,
        orb_counts_sc=[4] * 8,
        orb_counts_prim=[4, 4],
    )
    return LCAOUnfolder(HamiltonIOModel(sc), relabel)


def eigenvalue_run_sums(energies, weights, tol=1e-6):
    order = np.argsort(energies)
    energies = np.asarray(energies)[order]
    weights = np.asarray(weights)[order]
    out = []
    start = 0
    for i in range(1, len(energies) + 1):
        if i == len(energies) or energies[i] - energies[i - 1] > tol:
            out.append(float(weights[start:i].sum()))
            start = i
    return np.asarray(out)
