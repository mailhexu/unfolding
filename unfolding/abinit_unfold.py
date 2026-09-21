"""ABINIT ETSF-WFK reader and one-call planewave unfolding adapter.

The reader intentionally uses only ``netCDF4`` rather than abipy: ABINIT
WFK files carry the raw planewave coefficients needed by :class:`PWUnfolder`,
while abipy 1.0.0 exposes structure/bands but no raw-cg surface. The import is
lazy so the package core remains usable without the optional ``abinit`` extra.

Accepted WFK contract (ABINIT 9/10 ETSF Nanoquanta netCDF):

* ``iomode 3`` produced a netCDF WFK rather than a Fortran-binary WFK;
* every stored k-point has ``istwfk == 1`` (full G sphere);
* G vectors are SC reduced integers, coefficients are real/imag pairs;
* ``primitive_vectors`` are rows in ABINIT's ``rprim`` convention (Bohr).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re

import numpy as np

from .pw_unfolder import PWEigenData, PWUnfolder, PWWeights, _owned_array

HARTREE_TO_EV = 27.211386245988
_SUPPORTED_ABINIT_MAJORS = {9, 10}

def _decode_char(value) -> str:
    """Decode an ETSF fixed-width character variable without netCDF helpers."""
    array = np.asarray(value)
    if array.dtype.kind == "S":
        raw = b"".join(array.reshape(-1).tolist())
        return raw.decode("ascii", errors="replace").rstrip("\x00 ")
    if array.dtype.kind == "U":
        return "".join(array.reshape(-1).tolist()).rstrip("\x00 ")
    return str(array.item()).rstrip("\x00 ")


def _require_netcdf4():
    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as exc:
        if exc.name != "netCDF4":
            raise
        raise ImportError(
            "netCDF4 is required for ABINIT WFK unfolding. "
            "Install it with: pip install unfolding[abinit]"
        ) from exc
    return Dataset


@dataclass(frozen=True)
class WFKData(PWEigenData):
    """ETSF-WFK eigendata plus ABINIT provenance needed by the adapter.

    Inherits :class:`PWEigenData`, so it can be passed directly to
    :class:`PWUnfolder`. ``rprimd`` preserves the WFK's **row-oriented**
    ``primitive_vectors`` in Bohr; ``fermi_energy`` remains Hartree until
    :func:`unfold_abinit` crosses the public plotting boundary.
    """

    rprimd: np.ndarray
    codvsn: str
    istwfk: np.ndarray
    fermi_energy: float | None = None

    def __post_init__(self):
        super().__post_init__()
        rprimd = _owned_array(self.rprimd, float)
        istwfk = _owned_array(self.istwfk, int)
        if rprimd.shape != (3, 3):
            raise ValueError("primitive_vectors must have shape (3, 3)")
        if istwfk.shape != (len(self.kpoints),):
            raise ValueError("istwfk must have one entry per k-point")
        if not isinstance(self.codvsn, str) or not self.codvsn.strip():
            raise ValueError("codvsn must be a non-empty ABINIT version string")
        fermi = None if self.fermi_energy is None else float(self.fermi_energy)
        object.__setattr__(self, "rprimd", rprimd)
        object.__setattr__(self, "istwfk", istwfk)
        object.__setattr__(self, "fermi_energy", fermi)

    @property
    def nspinor(self) -> int:
        return self.coefficients[0].shape[2]


def _wfk_variable(nc, name):
    try:
        return nc.variables[name]
    except KeyError as exc:
        raise ValueError(f"not an ETSF ABINIT WFK: missing variable {name!r}") from exc


def read_wfk(path) -> WFKData:
    """Read an ABINIT 9/10 ETSF netCDF WFK into backend-free eigendata.

    Header and metadata arrays are small. Basis and band arrays are read one
    k-point at a time, with padded WFK axes sliced to
    `number_of_coefficients` and `number_of_states` before materializing
    them. Fortran WFKs, half-sphere storage, and real-only coefficient storage
    are rejected deliberately because their reconstruction semantics are
    outside the first adapter contract.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ABINIT WFK not found: {path!s}")
    Dataset = _require_netcdf4()
    try:
        nc = Dataset(path, "r")
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"cannot read {path!s} as a netCDF ABINIT WFK; "
            "rerun ABINIT with iomode 3 to produce an ETSF netCDF WFK"
        ) from exc

    try:
        kpoints = np.asarray(_wfk_variable(nc, "reduced_coordinates_of_kpoints")[:], dtype=float)
        npw = np.asarray(_wfk_variable(nc, "number_of_coefficients")[:], dtype=int)
        nstates = np.asarray(_wfk_variable(nc, "number_of_states")[:], dtype=int)
        istwfk = np.asarray(_wfk_variable(nc, "istwfk")[:], dtype=int)
        gvec_var = _wfk_variable(nc, "reduced_coordinates_of_plane_waves")
        coeff_var = _wfk_variable(nc, "coefficients_of_wavefunctions")
        eig_var = _wfk_variable(nc, "eigenvalues")
        rprimd = np.asarray(_wfk_variable(nc, "primitive_vectors")[:], dtype=float)
        codvsn = _decode_char(_wfk_variable(nc, "codvsn")[:])
        fermi = float(nc.variables["fermi_energy"][:]) if "fermi_energy" in nc.variables else None

        match = re.match(r"\s*(\d+)", codvsn)
        if match is None or int(match.group(1)) not in _SUPPORTED_ABINIT_MAJORS:
            supported = ", ".join(str(v) for v in sorted(_SUPPORTED_ABINIT_MAJORS))
            raise ValueError(
                f"unsupported ABINIT WFK major version {codvsn!r}; supported ABINIT WFK major versions: {supported}"
            )
        if kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise ValueError("not an ETSF ABINIT WFK: invalid reduced_coordinates_of_kpoints shape")
        if npw.shape != (len(kpoints),) or istwfk.shape != (len(kpoints),):
            raise ValueError("not an ETSF ABINIT WFK: inconsistent per-k npw or istwfk dimensions")
        if np.any(istwfk != 1):
            bad = np.where(istwfk != 1)[0].tolist()
            raise ValueError(
                f"WFK has istwfk={istwfk.tolist()} (bad k-point indices {bad}); "
                "rerun ABINIT with istwfk 1 for full-G planewave unfolding"
            )

        coeff_shape = tuple(coeff_var.shape)
        if len(coeff_shape) != 6:
            raise ValueError("not an ETSF ABINIT WFK: invalid coefficient array dimensions")
        nspin, nk, mband, nspinor, max_npw, ncomplex = coeff_shape
        if ncomplex != 2:
            raise ValueError(
                "WFK uses real coefficient storage; this adapter requires "
                "complex real/imag coefficient pairs"
            )
        eig_shape = tuple(eig_var.shape)
        if nk != len(kpoints) or len(eig_shape) != 3 or eig_shape[:2] != (nspin, nk):
            raise ValueError("not an ETSF ABINIT WFK: inconsistent coefficient/eigenvalue dimensions")
        if nstates.shape != (nspin, nk):
            raise ValueError("not an ETSF ABINIT WFK: invalid number_of_states shape")
        gvec_shape = tuple(gvec_var.shape)
        if gvec_shape != (nk, max_npw, 3):
            raise ValueError("not an ETSF ABINIT WFK: invalid reduced plane-wave dimensions")

        gvecs: list[np.ndarray] = []
        coefficients: list[np.ndarray] = []
        eigenvalues: list[np.ndarray] = []
        for ik in range(nk):
            if not np.all(nstates[:, ik] == nstates[0, ik]):
                raise ValueError("WFK has spin-dependent nband; this adapter requires common nband")
            nband = int(nstates[0, ik])
            nplane = int(npw[ik])
            if not (0 < nband <= mband and nband <= eig_shape[2] and 0 < nplane <= max_npw):
                raise ValueError(f"WFK has invalid nband/npw at k-point {ik}")
            raw = np.asarray(coeff_var[:, ik, :nband, :, :nplane, :], dtype=float)
            coefficients.append(raw[..., 0] + 1j * raw[..., 1])
            eigenvalues.append(np.asarray(eig_var[:, ik, :nband], dtype=float))
            gvecs.append(np.asarray(gvec_var[ik, :nplane, :], dtype=int))

        return WFKData(
            kpoints=kpoints,
            gvecs=tuple(gvecs),
            coefficients=tuple(coefficients),
            eigenvalues=tuple(eigenvalues),
            rprimd=rprimd,
            codvsn=codvsn,
            istwfk=istwfk,
            fermi_energy=fermi,
        )
    finally:
        nc.close()


def unfold_abinit(
    wfk=None,
    unfold_sc_mat=None,
    kpts=None,
    *,
    data: WFKData | None = None,
    knames=None,
    xqpts=None,
    Xqpts=None,
    spin: int = 0,
    average_degenerate: float | None = None,
    fermi_shift: bool = True,
    axis=None,
    output=None,
    style="alpha",
    color="blue",
    yrange=None,
    ylabel="Energy (eV)",
    ypad=1.5,
    width=2,
    title=None,
):
    """Unfold an ABINIT WFK on a primitive-cell k-path and return Axes.

    Parameters
    ----------
    wfk : path-like or WFKData, optional
        Supercell WFK path, or a pre-parsed record. Supply exactly one of
        wfk and data.
    unfold_sc_mat : (3, 3) integer array
        Row-convention supercell matrix satisfying A_sc = M @ A_prim.
    kpts : (nk, 3) array-like
        Primitive-cell fractional k-path to unfold.
    data : WFKData, optional
        Pre-parsed WFK record; avoids filesystem and netCDF I/O.
    knames, xqpts, Xqpts : sequence, optional
        Plot labels, path coordinates, and tick positions.
    spin : int
        Collinear WFK spin-channel index.
    average_degenerate : float, optional
        Band-energy grouping tolerance in eV.
    fermi_shift : bool
        Subtract the WFK Fermi header and draw zero as the Fermi level.
        Requires that the WFK carries a fermi_energy header.
    axis : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    output : path-like, optional
        Figure filename to save after drawing.
    style, color, width, yrange, ylabel, ypad, title
        Plotting options forwarded to PWWeights.plot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing weight-coded unfolded bands.
    """
    if isinstance(wfk, WFKData):
        if data is not None:
            raise ValueError("provide WFKData through either wfk or data, not both")
        data = wfk
    elif data is None:
        if wfk is None:
            raise ValueError("provide a WFK path or WFKData")
        data = read_wfk(wfk)
    elif wfk is not None:
        raise ValueError("provide either wfk or data, not both")

    if not isinstance(data, WFKData):
        raise TypeError("data must be WFKData")
    if unfold_sc_mat is None:
        raise ValueError("unfold_sc_mat is required")
    if kpts is None:
        raise ValueError("kpts (the primitive-cell k-path) is required")
    if fermi_shift and data.fermi_energy is None:
        raise ValueError(
            "WFK has no fermi_energy header; pass fermi_shift=False "
            "to plot unshifted energies"
        )

    result = PWUnfolder(data, unfold_sc_mat).compute(kpts, spin=spin)
    shift = data.fermi_energy * HARTREE_TO_EV if fermi_shift else 0.0
    result_ev = PWWeights(
        result.kpoints,
        result.eigenvalues * HARTREE_TO_EV - shift,
        result.weights,
        result.fold_kpoints,
        result.sc_kpoints,
    )
    if average_degenerate is not None:
        result_ev = result_ev.average_degenerate(float(average_degenerate))
    ax = result_ev.plot(
        xqpts=xqpts,
        style=style,
        axis=axis,
        ylabel=ylabel,
        efermi=0.0 if fermi_shift else None,
        yrange=yrange,
        color=color,
        width=width,
        xticks=[knames, Xqpts] if knames is not None and Xqpts is not None else None,
        title=title,
        ypad=ypad,
    )
    if output is not None:
        ax.figure.savefig(output)
    return ax
