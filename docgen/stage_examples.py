"""Assemble local examples without copying raw inputs into Git.

Uses the packer's existing bundle manifests and docgen/local-data.toml.
Unlike downloadable archives, local staging may link private/large inputs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pack_bundles import REPO, _collect_files, _read_manifest

ROOT = Path(REPO)


def _relative(path):
    value = Path(path)
    if value.is_absolute() or ".." in value.parts or value == Path("."):
        raise ValueError(f"expected a nonempty relative path: {path!r}")
    return value


def _inputs(name, data_root, catalog):
    """Destination -> source for manifest and external input files."""
    files = {}
    for source, destination in _read_manifest(name):
        src = ROOT / _relative(source)
        dst = _relative(destination)
        if src.is_dir():
            for abs_path, rel in _collect_files(src):
                files[dst / rel] = Path(abs_path)
        else:
            files[dst] = src
    for destination, source in catalog.get(name, {}).items():
        dst = _relative(destination)
        if dst in files:
            raise ValueError(f"{name}: duplicate manifest/runtime input: {dst}")
        files[dst] = data_root / _relative(source)
    for dst, src in files.items():
        if not src.is_file() or src.stat().st_size == 0:
            raise ValueError(f"{name}: missing or empty input {src} (needed as {dst})")
    return files


def stage(name, output, files):
    destination = output / name
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"already exists: {destination}; use a fresh --output directory")
    shutil.copytree(ROOT / "examples" / name / "bundle", destination,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel, source in files.items():
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            # Manifest inputs are authoritative, as in the bundle packer.
            if target.is_dir() and not target.is_symlink():
                raise ValueError(f"cannot replace source directory with input: {target}")
            target.unlink()
        target.symlink_to(source.resolve())
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="bundle names (default: all)")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("UNFOLDING_DATA_DIR",
                                                   ROOT.parent / "unfolding-data")))
    parser.add_argument("--output", type=Path,
                        default=ROOT.parent / ".tmp" / "local-examples")
    parser.add_argument("--check", action="store_true", help="check inputs without staging")
    args = parser.parse_args(argv)
    data_root, output = args.data_root.expanduser().resolve(), args.output.expanduser().resolve()
    # Do not place private bytes or links in a checkout or the data store.
    if output == ROOT or ROOT in output.parents or output == data_root or data_root in output.parents:
        parser.error("--output must be outside the checkout and outside --data-root")
    if data_root == ROOT or ROOT in data_root.parents:
        parser.error("--data-root must be outside the checkout")
    with (ROOT / "docgen" / "local-data.toml").open("rb") as fh:
        catalog = tomllib.load(fh)
    available = {p.parent.parent.name for p in (ROOT / "examples").glob("*/bundle/unfold.toml")}
    names = args.names or sorted(available)
    unknown = set(names) - available
    if unknown:
        parser.error(f"unknown bundles: {', '.join(sorted(unknown))}")
    try:
        # Check the entire selection before writing anything.
        plans = [(name, _inputs(name, data_root, catalog)) for name in names]
        if not args.check:
            for name, _ in plans:
                dest = output / name
                if dest.exists() or dest.is_symlink():
                    raise ValueError(f"already exists: {dest}; use a fresh --output directory")
        for name, files in plans:
            if args.check:
                print(f"OK {name}: {len(files)} input files")
            else:
                dest = stage(name, output, files)
                print(f"STAGED {name}: {dest}")
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
