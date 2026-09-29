"""Unfolded-band JSON dataset (story 038, ADR-011).

Serializes a route result (any object with ``kpoints``, ``eigenvalues``
and ``weights`` arrays, plus optional ``fold_kpoints``/``sc_kpoints``)
into a versioned, plain-list JSON document and loads it back:

    from unfolding.dataset import save_dataset, load_dataset
    save_dataset(result, "bands.json", route="pw",
                 fermi_energy=-0.25, energy_reference="run Fermi level")
    ds = load_dataset("bands.json")

Schema ``unfolding.dataset/1`` (sorted keys, one array per line):

    schema             "unfolding.dataset/1"
    route              route name of the producing run
    config_provenance  dict of route parameters worth recording
    units              {"energy": <energy_unit>} (default "eV")
    fermi_energy       in energy_unit, or null          (optional)
    energy_reference   provenance string for the zero   (optional)
    spin_channels      1 or 2
    kpoints            (nk, 3) primitive fractional, user order
    fold_kpoints       (nk, 3) reduced folds            (optional)
    sc_kpoints         (nk, 3) stored SC momenta        (optional)
    eigenvalues        (nk, nband) or (spin, nk, nband)
    weights            same shape as eigenvalues

NaN weights (masked bands) are written as JSON ``null`` and read back
as NaN; documents stay strictly parseable and git-diffable. The module
depends only on json/numpy/dataclasses.
"""
import dataclasses
import json
import os
from typing import Any

import numpy as np

__all__ = ["SCHEMA", "Dataset", "save_dataset", "load_dataset"]

SCHEMA = "unfolding.dataset/1"


def _owned(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _nullify(array: np.ndarray, name: str) -> list:
    """Plain nested lists with NaN mapped to None; infinities rejected."""
    arr = np.asarray(array, dtype=float)
    if np.isinf(arr).any():
        raise ValueError(f"{name}: infinite values cannot be serialized")
    if name != "weights" and np.isnan(arr).any():
        raise ValueError(f"{name}: NaN values are only supported in weights")
    return _nullify_list(arr.tolist())


def _nullify_list(value):
    if isinstance(value, list):
        return [_nullify_list(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _load_array(value, name):
    """Nested lists with nulls -> float ndarray (null -> NaN)."""
    if value is None:
        raise ValueError(f"{name}: missing")

    def conv(v):
        if v is None:
            return np.nan
        if isinstance(v, list):
            return [conv(x) for x in v]
        return float(v)

    return _owned(np.asarray(conv(value), dtype=float))


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Parsed unfolded-band dataset (frozen float arrays)."""

    route: str
    kpoints: np.ndarray
    eigenvalues: np.ndarray
    weights: np.ndarray
    spin_channels: int = 1
    energy_unit: str = "eV"
    fold_kpoints: np.ndarray | None = None
    sc_kpoints: np.ndarray | None = None
    fermi_energy: float | None = None
    energy_reference: str | None = None
    provenance: dict | None = None

    def __post_init__(self):
        if self.spin_channels not in (1, 2):
            raise ValueError(f"spin_channels: {self.spin_channels} not in (1, 2)")
        object.__setattr__(self, "kpoints", _owned(self.kpoints))
        object.__setattr__(self, "eigenvalues", _owned(self.eigenvalues))
        object.__setattr__(self, "weights", _owned(self.weights))
        for name in ("fold_kpoints", "sc_kpoints"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _owned(getattr(self, name)))
        if self.weights.shape != self.eigenvalues.shape:
            raise ValueError(
                f"weights shape {self.weights.shape} != eigenvalues shape "
                f"{self.eigenvalues.shape}")
        for name in ("kpoints", "eigenvalues", "weights",
                     "fold_kpoints", "sc_kpoints"):
            value = getattr(self, name)
            if value is None:
                continue
            if np.isinf(value).any():
                raise ValueError(f"{name}: infinite values are not valid")
            if name != "weights" and np.isnan(value).any():
                raise ValueError(f"{name}: NaN values are not valid")
        if self.kpoints.ndim != 2 or self.kpoints.shape[1] != 3:
            raise ValueError(
                f"kpoints must have shape (nk, 3), got {self.kpoints.shape}")
        nk = len(self.kpoints)
        for name in ("fold_kpoints", "sc_kpoints"):
            coord = getattr(self, name)
            if coord is not None and coord.shape != self.kpoints.shape:
                raise ValueError(
                    f"{name} shape {coord.shape} != kpoints shape "
                    f"{self.kpoints.shape}")
        eig = self.eigenvalues
        if eig.ndim == 3:
            if self.spin_channels == 1 or eig.shape[0] != self.spin_channels \
                    or eig.shape[1] != nk:
                raise ValueError(
                    f"eigenvalues shape {eig.shape} incompatible with "
                    f"spin_channels={self.spin_channels} and {nk} k-points")
        elif eig.ndim == 2:
            if self.spin_channels != 1 or eig.shape[0] != nk:
                raise ValueError(
                    f"eigenvalues shape {eig.shape} incompatible with "
                    f"spin_channels={self.spin_channels} and {nk} k-points")
        else:
            raise ValueError("eigenvalues must be (nk, nband) or (spin, nk, nband)")

    @classmethod
    def from_result(cls, result, *, route, spin_channels=None,
                    fermi_energy=None, energy_reference=None,
                    provenance=None):
        """Dataset from a duck-typed engine result.

        ``spin_channels`` defaults to 2 for (spin, nk, nband) arrays
        and 1 for (nk, nband) arrays.
        """
        eig = np.asarray(
            getattr(result, "eigenvalues", None)
            if getattr(result, "eigenvalues", None) is not None
            else result.energies, dtype=float)
        if spin_channels is None:
            spin_channels = 2 if eig.ndim == 3 else 1
        return cls(
            route=route,
            kpoints=np.asarray(result.kpoints, dtype=float),
            eigenvalues=eig,
            weights=np.asarray(result.weights, dtype=float),
            spin_channels=spin_channels,
            fold_kpoints=None if getattr(result, "fold_kpoints", None) is None
            else np.asarray(result.fold_kpoints, dtype=float),
            sc_kpoints=None if getattr(result, "sc_kpoints", None) is None
            else np.asarray(result.sc_kpoints, dtype=float),
            fermi_energy=fermi_energy,
            energy_reference=energy_reference,
            provenance=provenance,
        )

    def to_document(self) -> dict:
        doc = {
            "schema": SCHEMA,
            "route": self.route,
            "config_provenance": self.provenance or {},
            "units": {"energy": self.energy_unit},
            "fermi_energy": None if self.fermi_energy is None
            else float(self.fermi_energy),
            "energy_reference": self.energy_reference,
            "spin_channels": self.spin_channels,
            "kpoints": _nullify(self.kpoints, "kpoints"),
            "eigenvalues": _nullify(self.eigenvalues, "eigenvalues"),
            "weights": _nullify(self.weights, "weights"),
        }
        if self.fold_kpoints is not None:
            doc["fold_kpoints"] = _nullify(self.fold_kpoints, "fold_kpoints")
        if self.sc_kpoints is not None:
            doc["sc_kpoints"] = _nullify(self.sc_kpoints, "sc_kpoints")
        return doc


def save_dataset(result, path, *, route=None, spin_channels=None,
                 provenance=None, fermi_energy=None, energy_reference=None,
                 energy_unit=None):
    """Serialize ``result`` to ``path`` as a schema-v1 JSON document.

    ``result`` may be a duck-typed engine result or an already parsed
    :class:`Dataset`; parsed datasets keep their stored metadata
    (Fermi level, energy reference/unit, provenance, spin channels)
    unless the corresponding keyword overrides it.
    """
    if route is None and not isinstance(result, Dataset):
        raise ValueError("route: required when saving an engine result")
    if isinstance(result, Dataset):
        ds = Dataset(
            route=route if route is not None else result.route,
            kpoints=result.kpoints,
            eigenvalues=result.eigenvalues,
            weights=result.weights,
            spin_channels=(spin_channels if spin_channels is not None
                           else result.spin_channels),
            energy_unit=(energy_unit if energy_unit is not None
                         else result.energy_unit),
            fold_kpoints=result.fold_kpoints,
            sc_kpoints=result.sc_kpoints,
            fermi_energy=(fermi_energy if fermi_energy is not None
                          else result.fermi_energy),
            energy_reference=(energy_reference if energy_reference is not None
                              else result.energy_reference),
            provenance=provenance if provenance is not None else result.provenance,
        )
    else:
        ds = Dataset.from_result(
            result, route=route, spin_channels=spin_channels,
            fermi_energy=fermi_energy, energy_reference=energy_reference,
            provenance=provenance)
        object.__setattr__(ds, "energy_unit",
                           energy_unit if energy_unit is not None else "eV")
    with open(os.fspath(path), "w", encoding="utf-8") as fh:
        json.dump(ds.to_document(), fh, sort_keys=True, indent=1,
                  allow_nan=False)
        fh.write("\n")
    return ds


def load_dataset(path) -> Dataset:
    """Parse a dataset document back into a :class:`Dataset`."""
    with open(os.fspath(path), encoding="utf-8") as fh:
        doc = json.load(fh)
    schema = doc.get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"schema: unsupported {schema!r}; expected {SCHEMA!r}")
    return Dataset(
        route=doc["route"],
        kpoints=_load_array(doc["kpoints"], "kpoints"),
        eigenvalues=_load_array(doc["eigenvalues"], "eigenvalues"),
        weights=_load_array(doc["weights"], "weights"),
        spin_channels=int(doc.get("spin_channels", 1)),
        energy_unit=doc.get("units", {}).get("energy", "eV"),
        fold_kpoints=None if doc.get("fold_kpoints") is None
        else _load_array(doc["fold_kpoints"], "fold_kpoints"),
        sc_kpoints=None if doc.get("sc_kpoints") is None
        else _load_array(doc["sc_kpoints"], "sc_kpoints"),
        fermi_energy=doc.get("fermi_energy"),
        energy_reference=doc.get("energy_reference"),
        provenance=doc.get("config_provenance"),
    )
