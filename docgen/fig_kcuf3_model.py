#!/usr/bin/env python
"""Regenerate the published KCuF3 model-magnon figure.

The Heisenberg model, analytic checks, and plotting are in the runnable
example; this docgen entry point only selects the published output path.
"""
import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "tb2j-kcuf3-model" / "run.py"
DEFAULT = ROOT / "docs" / "static" / "images" / "tb2j_kcuf3_model_unfolded.png"


def main(out_path=DEFAULT):
    spec = importlib.util.spec_from_file_location("kcuf3_model_example", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TemporaryDirectory(prefix="kcuf3-docgen-") as scratch:
        module.main(output_path=out_path, results_path=Path(scratch) / "results.json")
    return out_path


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    print(main(output))
