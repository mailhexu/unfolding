#!/usr/bin/env python
"""Render the pristine-Si ABINIT unfolded-band figure (Si8 supercell).

Unfolds the 8-atom conventional-cell WFK onto the primitive fcc path and,
when the primitive-cell path WFK is available, overlays the independently
computed primitive bands. The two agree to ~0.03 eV after a constant
potential-reference shift (the supercell run uses a Gamma-only SCF
density, the primitive run an 8x8x8-sampled one), demonstrating the
planewave unfolding on a case with a known answer.
"""
import sys

import numpy as np

from abinit_si_common import (
    DATA, DEGEN_TOL_EV, MATRIX, ROOT, match_path_subset, spectral_weight_map,
    draw_map, siesta_style_path,
)


def main(out_path):
    from unfolding.abinit_unfold import HARTREE_TO_EV
    from unfolding.pw_unfolder import PWUnfolder

    dense = DATA / "si8_gxwglwxo_DS2_WFK.nc"
    sparse = DATA / "si8_gxwglx_cornerso_DS2_WFK.nc"
    wfk = dense if dense.is_file() else sparse
    if not wfk.is_file():
        raise FileNotFoundError(f"no Si8 path WFK found (tried {dense.name}, {sparse.name})")

    data, kpts, x = match_path_subset(wfk)
    res = PWUnfolder(data, MATRIX).compute(
        kpts, resolve_degenerate=DEGEN_TOL_EV / HARTREE_TO_EV
    )
    E, W, egrid, A = spectral_weight_map(res, data)

    overlay = None
    prim_wfk = DATA / "si_prim_patho_DS2_WFK.nc"
    if prim_wfk.is_file():
        pdata, pkpts, px = match_path_subset(prim_wfk, np.eye(3, dtype=int))
        pres = PWUnfolder(pdata, np.eye(3, dtype=int)).compute(pkpts)
        pe = pres.eigenvalues * HARTREE_TO_EV - pdata.fermi_energy * HARTREE_TO_EV
        # Align the potential reference by the median high-weight offset.
        hi = W > 0.9
        devs = []
        for ik in range(len(E)):
            unfolded = np.sort(E[ik][hi[ik]])
            primitive = np.sort(pe[ik])
            n = min(len(unfolded), len(primitive))
            if n:
                devs.append(unfolded[:n] - primitive[:n])
        shift = np.median(np.concatenate(devs))
        overlay = (px, pe + shift)

    _, _, Xqpts = siesta_style_path()
    return draw_map(
        x, egrid, A, Xqpts,
        "ABINIT Si unfolded: supercell vs primitive bands", out_path,
        overlay=overlay,
    )


if __name__ == "__main__":
    output = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ROOT / "docs" / "static" / "images" / "si8_abinit_unfolded.png"
    )
    print(main(output))
