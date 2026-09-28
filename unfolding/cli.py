"""Unified command-line interface: ``unfolding <route> [flags] | --config file.toml``.

Subparsers and flags are generated from the :data:`unfolding.config.ROUTES`
registry, so CLI, TOML schema, and the Python adapters share one definition.
``--config`` is mutually exclusive with route flags; route configs are
executed through :func:`unfolding.routes.run`.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

from .config import ROUTES, ConfigError, config_from_flags, load_config
from .routes import run


def _field_default(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


def _add_flag(sub, f):
    """Add one argparse flag for a route dataclass field."""
    fs = f.metadata["spec"]
    dashed = fs.flag or f.name.replace("_", "-")
    default = _field_default(f)
    if fs.kind == "matrix" and default is not None:
        # argparse carries flat 9-tuples; config_from_flags re-nests
        default = [v for row in default for v in row]
    kwargs = {
        "dest": f.name,
        "default": default,
        "help": fs.help,
    }
    if kwargs["default"] is not None and fs.kind != "bool":
        kwargs["help"] += f" (default: {kwargs['default']!r})"
    if fs.kind in ("path", "out", "str"):
        kwargs["type"] = str
        kwargs["metavar"] = dashed.upper()
    elif fs.kind == "int":
        kwargs["type"] = int
        kwargs["metavar"] = "N"
    elif fs.kind == "float":
        kwargs["type"] = float
        kwargs["metavar"] = "X"
    elif fs.kind == "bool":
        kwargs["action"] = argparse.BooleanOptionalAction
    elif fs.kind == "matrix":
        kwargs["type"] = int
        kwargs["nargs"] = 9
        kwargs["metavar"] = ("M00", "M01", "M02", "M10", "M11", "M12",
                             "M20", "M21", "M22")
    elif fs.kind == "kpoints":
        kwargs["type"] = float
        kwargs["nargs"] = "+"
        kwargs["metavar"] = "K"
    elif fs.kind == "float-list":
        kwargs["type"] = float
        kwargs["nargs"] = "+"
        kwargs["metavar"] = "V"
    elif fs.kind == "str-list":
        kwargs["nargs"] = "+"
        kwargs["metavar"] = "S"
    else:  # pragma: no cover
        raise ValueError(f"unknown field kind {fs.kind!r}")
    if fs.choices:
        kwargs["choices"] = list(fs.choices)
    sub.add_argument(f"--{dashed}", **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unfolding",
        description=(
            "Bloch-wave unfolding: run a route (siesta, phonopy, abinit-wfk, "
            "abinit-ddb, magnon) from flags or from a TOML config file."
        ),
    )
    parser.add_argument(
        "--config", metavar="FILE",
        help="TOML config file (mutually exclusive with route flags)")
    sub = parser.add_subparsers(dest="route", metavar="ROUTE")
    for name, spec in ROUTES.items():
        sp = sub.add_parser(name, help=f"{spec.name} route")
        for f in dataclasses.fields(spec.config):
            _add_flag(sp, f)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.config is not None:
            if args.route is not None:
                parser.error(
                    "--config is mutually exclusive with route flags; give "
                    "either --config FILE or a route with its flags")
            cfg = load_config(args.config)
        elif args.route is None:
            parser.error("a route subcommand or --config FILE is required")
        else:
            cfg = config_from_flags(args.route, args)
    except ConfigError as exc:
        print(f"unfolding: error: {exc}", file=sys.stderr)
        return 2

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
