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
| `abinit` | `read_wfk`, `unfold_abinit` (ABINIT WFK unfolding) | netCDF4 |
| `abipy` | `DDB_unfolder` (ABINIT DDB phonon unfolding) | abipy + a working `anaddb` |
| `dev` | test suite | pytest, sympy |

```bash
pip install unfolding[siesta]      # SIESTA
pip install unfolding[abinit]      # ABINIT WFK
pip install unfolding[abipy]       # ABINIT DDB (needs anaddb)
pip install unfolding[phonopy]     # phonopy force constants
```

## Notes

- All backend imports are lazy: the core package works without any extra
  installed, and a missing backend raises an `ImportError` naming the extra
  to install.
- The DDB route shells out to ABINIT's `anaddb`, so a working ABINIT
  installation must be on your `PATH` (or configured in abipy's
  `manager.yml`).
