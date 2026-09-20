"""Shared production SIESTA fixture helpers for story-010 tests."""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIM_FDF = os.path.join(ROOT, "tests", "data", "si_example", "si_prim.fdf")
B_DIAMOND = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])


def read_si_model(fdf):
    from HamiltonIO.siesta.sisl_wrapper import SislParser

    class RListParser(SislParser):
        def read_Rlist(self, geom=None):
            return self.ham.lattice.sc_off

    return RListParser(fdf).get_model()


def read_fermi_energy(path):
    with open(path) as fh:
        return float(fh.readline())


def make_si_unfolder(sc_fdf, *, tol_r=0.04):
    from unfolding.lcao_unfolder import HamiltonIOModel, LCAOUnfolder
    from unfolding.mapping import RelabelMap

    prim = read_si_model(PRIM_FDF)
    sc = read_si_model(sc_fdf)
    relabel = RelabelMap.from_atoms(
        sc.atoms,
        prim.atoms,
        B_DIAMOND,
        tol_r=tol_r,
        orb_counts_sc=[4] * 8,
        orb_counts_prim=[4, 4],
    )
    return LCAOUnfolder(HamiltonIOModel(sc), relabel)
