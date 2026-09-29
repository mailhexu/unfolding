#!/usr/bin/env python
"""Render the ABINIT Si7P unfolded spectral-weight figure.

P-doped Si in the 8-atom conventional cell, unfolded onto the primitive
fcc path (identical to the SIESTA example and the pristine-Si ABINIT
figure). Rendered as a Gaussian-smeared spectral-weight map; the
substitutional defect mixes fold sectors, so host bands carry the dark
weight while defect-scattered states appear dimmer.

Coverage note: the committed Si7P WFK fixtures store only the four path
corners -- supercell momenta (0,0,0), (0,1,0), (0.5,1,0), (0.5,0.5,0.5)
-- so only 7 of the 305 requested path folds have a stored match and
this figure renders the corner subset (weight at Gamma/X/W/Gamma/L on
the cropped path axis) until si7p_gamma_x_patho_DS2_WFK.nc is
regenerated at full 305-point density.
"""
import sys

from abinit_si_common import (
    DATA, DEGEN_TOL_EV, MATRIX, ROOT, draw_map, match_path_subset,
    siesta_style_path, spectral_weight_map, pw_data,
)


def main(out_path):
    from HamiltonIO.abinit import HARTREE_TO_EV
    from unfolding.pw_unfolder import PWUnfolder

    wfk = DATA / "si7p_gamma_x_patho_DS2_WFK.nc"
    if not wfk.is_file():
        raise FileNotFoundError(f"ABINIT Si7P path WFK not found: {wfk}")

    data, kpts, x = match_path_subset(wfk)
    res = PWUnfolder(pw_data(data), MATRIX).compute(
        kpts, resolve_degenerate=DEGEN_TOL_EV / HARTREE_TO_EV
    )
    _, _, egrid, A = spectral_weight_map(res, data)
    _, _, Xqpts = siesta_style_path()
    return draw_map(
        x, egrid, A, Xqpts, "ABINIT Si$_7$P unfolded spectral weight", out_path
    )


if __name__ == "__main__":
    output = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ROOT / "docs" / "static" / "images" / "si7p_abinit_unfolded.png"
    )
    print(main(output))
