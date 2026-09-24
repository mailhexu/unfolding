"""Thin CLI for TB2J magnon downfolding (story 024, FR-025, ADR-013).

Console entry point ``unfolding-magnon``: downfold a TB2J magnon band
structure onto a primitive-cell q-path in one command. The primitive
cell is derived from the TB2J cell through the unfold matrix
(A_prim = M^-1 A_sc); the q-path special points come from ase on that
primitive cell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unfolding-magnon",
        description=(
            "Downfold a TB2J magnon band structure onto a primitive-cell "
            "q-path with unfolding weights."
        ),
    )
    parser.add_argument(
        "path",
        help="TB2J results directory containing TB2J.pickle",
    )
    parser.add_argument(
        "--unfold-mat",
        nargs=9,
        type=int,
        required=True,
        metavar=("M00", "M01", "M02", "M10", "M11", "M12", "M20", "M21", "M22"),
        help="unfold matrix rows: A_sc = M @ A_prim (row convention)",
    )
    parser.add_argument(
        "--kpath",
        default="GXMGRX",
        help="colon-free special-point letters for the primitive q-path "
        "(default: GXMGRX, simple-cubic letters)",
    )
    parser.add_argument("--npts", type=int, default=200, help="path points")
    parser.add_argument(
        "--degen-tol",
        type=float,
        default=1e-5,
        help="degenerate-group energy tolerance in eV (default 1e-5)",
    )
    parser.add_argument(
        "--spin-conf",
        nargs="+",
        type=float,
        default=None,
        help="collinear moments as nspin x 3 floats, e.g. 0 0 3 0 0 -3 "
        "(default: moments from the pickle)",
    )
    parser.add_argument(
        "--output",
        default="magnon_unfolded.png",
        help="output figure path",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="optional path to dump energies (meV) and weights as JSON",
    )
    parser.add_argument(
        "--style", default="alpha", choices=("alpha", "width", "scatter"),
        help="weight-coded plot style",
    )
    return parser


def _primitive_path(cell_sc, M, letters, npts):
    """(kpts, x, X, labels) primitive q-path from the derived cell."""
    import numpy as np
    from ase.dft.kpoints import bandpath, get_special_points

    prim = np.linalg.solve(np.asarray(M, dtype=float), np.asarray(cell_sc, dtype=float))
    points = get_special_points(prim, eps=0.01)
    known = [c for c in letters if c in points]
    if not known:
        raise SystemExit(
            f"no special points of {letters!r} exist for the derived "
            f"primitive cell; available: {sorted(points)}"
        )
    path = bandpath([points[c] for c in known], prim, npts)
    x, X, labels = path.get_linear_kpoint_axis()
    return path.kpts, x, X, labels


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    import numpy as np

    from .tb2j_unfold import magnon_eigendata_from_tb2j

    try:
        from TB2J.magnon.magnon3 import Magnon
    except ModuleNotFoundError as exc:
        print(
            "TB2J is required for unfolding-magnon. "
            "Install it with: pip install unfolding[tb2j]",
            file=sys.stderr,
        )
        return 2

    M = np.array(args.unfold_mat, dtype=int).reshape(3, 3)
    magnon = Magnon.from_TB2J_results(path=args.path)
    # set_reference also initializes Snorm (required by Hq); always call
    # it with the collinear defaults, overriding moments if given
    magmoms = (
        np.array(args.spin_conf, dtype=float).reshape(-1, 3)
        if args.spin_conf is not None
        else np.asarray(magnon.magmom, dtype=float)
    )
    magnon.set_reference(
        Q=(0, 0, 0),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([1.0, 0.0, 0.0]),
        magmoms=magmoms,
    )

    kpts, x, X, labels = _primitive_path(magnon.cell, M, args.kpath, args.npts)

    from .magnon_unfolder import MagnonUnfolder

    eigendata = magnon_eigendata_from_tb2j(magnon, np.mod(kpts @ M.T, 1.0))
    unf = MagnonUnfolder(eigendata, M)
    res = unf.compute(kpts, resolve_degenerate=args.degen_tol)

    if args.json:
        payload = {
            "kpoints": np.asarray(kpts).tolist(),
            "x": np.asarray(x).tolist(),
            "X": list(map(float, X)),
            "labels": labels,
            "energies_mev": (res.energies * 1000.0).tolist(),
            "weights": res.weights.tolist(),
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh)

    from .magnon_unfolder import MagnonWeights

    res_mev = MagnonWeights(
        res.kpoints, res.energies * 1000.0, res.weights, res.sc_kpoints
    )
    ax = res_mev.plot(
        xqpts=x,
        xticks=(labels, X),
        style=args.style,
        ylabel="Energy (meV)",
        title=f"{args.path} unfolded",
    )
    ax.figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}" + (f" and {args.json}" if args.json else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
