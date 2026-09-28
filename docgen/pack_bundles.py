"""Pack the downloadable example bundles deterministically.

Every ``examples/<name>/bundle/`` directory is a complete bundle source:
scripts, README.txt, unfold.toml and the small text inputs/data. Binary
fixture data that already lives in the repository (tests/data/, other
examples/ trees) is pulled in AT PACK TIME through an optional manifest
file ``examples/<name>/bundle.manifest``, one pull per line::

    <repo-relative source> -> <destination path under the bundle>

so the committed sources stay small and nothing large is duplicated.
Bundles without a manifest are packed from their directory alone.

Usage:

    python docgen/pack_bundles.py                 # all bundles
    python docgen/pack_bundles.py openmx-si ...   # a subset

Output: ``docs/static/downloads/<name>.tar.gz`` -- gzip with mtime 0,
tar members sorted, normalized metadata (uid/gid 0, mtime 0), so
re-packing an unchanged tree is byte-stable. Every bundle is checked
against the 10 MB cap; licensed bytes (POTCAR/WAVECAR) are never
manifest sources.
"""
from __future__ import annotations

import gzip
import io
import os
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = os.path.join(REPO, "docs", "static", "downloads")
CAP_BYTES = 10 * 1024 * 1024

# Pruned from copied trees (regenerable code output the parsers never
# read) -- BANDS_*.dat is ABACUS's own plotting dump.
PRUNE_NAMES = {"BANDS_1.dat"}


def _collect_files(root):
    """Sorted (abs_path, rel_path) of every file under root."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn in PRUNE_NAMES or fn.endswith(".pyc"):
                continue
            abs_path = os.path.join(dirpath, fn)
            out.append((abs_path, os.path.relpath(abs_path, root)))
    return out


def _read_manifest(name):
    """[(repo-relative source, dest under bundle)] from bundle.manifest."""
    path = os.path.join(REPO, "examples", name, "bundle.manifest")
    if not os.path.exists(path):
        return []
    pulls = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src, sep, dst = line.partition("->")
            if not sep:
                raise SystemExit(f"{path}: expected '<src> -> <dst>', got {line!r}")
            pulls.append((src.strip(), dst.strip().rstrip("/")))
    return pulls


def _members_for(name):
    """Sorted [(tar_info, abs_source_path)] for one bundle."""
    src_dir = os.path.join(REPO, "examples", name, "bundle")
    if not os.path.isdir(src_dir):
        raise SystemExit(f"missing bundle sources: {src_dir}")

    members = []  # (abs_path, arcname)
    for abs_path, rel in _collect_files(src_dir):
        members.append((abs_path, f"{name}/{rel}"))
    for rel_src, rel_dst in _read_manifest(name):
        full = os.path.join(REPO, rel_src)
        if not os.path.exists(full):
            raise SystemExit(f"{name} manifest source missing: {rel_src}")
        if os.path.isdir(full):
            for abs_path, rel in _collect_files(full):
                members.append((abs_path, f"{name}/{rel_dst}/{rel}"))
        else:
            members.append((full, f"{name}/{rel_dst}"))

    members.sort(key=lambda m: m[1])
    infos = []
    for abs_path, arcname in members:
        info = tarfile.TarInfo(arcname)
        info.size = os.path.getsize(abs_path)
        info.mtime = 0
        info.mode = 0o755 if os.access(abs_path, os.X_OK) else 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        infos.append((info, abs_path))
    return infos


def pack(name: str) -> str:
    """Assemble and compress one bundle; returns the tarball path."""
    infos = _members_for(name)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for info, abs_path in infos:
            with open(abs_path, "rb") as fh:
                tar.addfile(info, fh)
    out_path = os.path.join(DOWNLOADS, f"{name}.tar.gz")
    compressed = io.BytesIO()
    with gzip.GzipFile(filename=name + ".tar", mode="wb",
                       fileobj=compressed, mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    data = compressed.getvalue()
    if len(data) > CAP_BYTES:
        raise SystemExit(f"{out_path} is {len(data)} bytes (> {CAP_BYTES} cap)")
    with open(out_path, "wb") as fh:
        fh.write(data)
    return out_path


def _bundle_names():
    examples = os.path.join(REPO, "examples")
    return sorted(
        d for d in os.listdir(examples)
        if os.path.isdir(os.path.join(examples, d, "bundle"))
    )


def main(argv):
    names = argv or _bundle_names()
    unknown = [n for n in names
               if not os.path.isdir(os.path.join(REPO, "examples", n, "bundle"))]
    if unknown:
        raise SystemExit(f"no bundle sources for: {unknown}")
    for name in names:
        out = pack(name)
        print(f"packed {name}: {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1:])
