---
title: "Installation"
weight: 10
---

Requires Python >= 3.9. Core dependencies: numpy, scipy, matplotlib, ase.

```bash
pip install unfolding
```

## Optional extras

Install the extra matching the code you want to read output from:

| Extra | Enables | Dependencies |
|---|---|---|
| `phonopy` | `phonopy_unfold` (phonon supercell unfolding) | phonopy |
| `siesta` | `unfold_siesta` (SIESTA LCAO unfolding) | HamiltonIO >= 0.3.8, sisl |
| `abinit` | `unfold_abinit` (ABINIT WFK unfolding); `HamiltonIO.abinit.read_wfk` parses the WFK | HamiltonIO >= 0.3.8, netCDF4 |
| `abinit-paw` (manual) | PAW WFK + JTH XML spectral weights | HamiltonIO, pypao, abinao |
| `vasp-paw` (manual) | PAW WAVECAR + licensed POTCAR spectral weights | HamiltonIO, pymatgen, pypao |
| `abipy` | `DDB_unfolder` (ABINIT DDB phonon unfolding) | abipy + a working `anaddb` |
| `dev` | test suite | pytest, sympy |

```bash
pip install unfolding[siesta]      # SIESTA
pip install unfolding[abinit]      # ABINIT WFK
pip install unfolding[abipy]       # ABINIT DDB (needs anaddb)
pip install unfolding[phonopy]     # phonopy force constants
```

PAW backends also need `pypao` and `abinao` (ABINIT route), or
`pymatgen` and `pypao` (VASP route). Install those packages in
the same environment as unfolding; no VASP POTCAR is shipped or fetched.

## Notes

- All backend imports are lazy: the core package works without any extra
  installed, and a missing backend raises an `ImportError` naming the extra
  to install.
- The DDB route shells out to ABINIT's `anaddb`, so a working ABINIT
  installation must be on your `PATH` (or configured in abipy's
  `manager.yml`).
