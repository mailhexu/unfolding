"""Unfold the bundled real SrTiO3 Wannier90 datasets without pythtb.

From the repository root, run one system or both:

    python examples/wannier_STO/wannier_unfold.py pristine
    python examples/wannier_STO/wannier_unfold.py ti-vacancy
    python examples/wannier_STO/wannier_unfold.py both --output-dir /tmp/sto-bands

The script uses the package's built-in Wannier90 reader. Required inputs
are the ``wannier90.win``, ``wannier90.wout``, ``wannier90_centres.xyz``
and ``wannier90_hr.dat`` files in ``data_nodefect`` or ``data``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from unfolding.wannier_unfold import Wannier90Model, WannierUnfolder

SCMAT = [[1, -1, 0], [1, 1, 0], [0, 0, 2]]
LABELS = ["pz", "px", "py"] * 12 + ["dz2", "dxy", "dyz", "dx2", "dxz"] * 4
KVECTORS = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
KNAMES = [r"$\Gamma$", "X", "M", r"$\Gamma$", "R"]
EXAMPLE_DIR = Path(__file__).resolve().parent


def unfold(system: str, output: Path) -> Path:
    """Render one shipped Wannier90 system to ``output``."""
    model_dir = EXAMPLE_DIR / ("data_nodefect" if system == "pristine" else "data")
    model = Wannier90Model(model_dir, "wannier90", scmat=SCMAT)
    unfolder = WannierUnfolder(model, labels=LABELS, sc_matrix=SCMAT)
    ax = unfolder.plot_unfolded_band(
        kvectors=KVECTORS, knames=KNAMES, npoints=200,
        resolve_degenerate=0.1 if system == "pristine" else None)
    ax.figure.set_size_inches(7.2, 5.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(ax.figure)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=("pristine", "ti-vacancy", "both"),
                        nargs="?", default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    systems = ("pristine", "ti-vacancy") if args.system == "both" else (args.system,)
    for system in systems:
        suffix = "unfolded" if system == "pristine" else "ti_vacancy"
        output = unfold(system, args.output_dir / f"sto_{suffix}.png")
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
