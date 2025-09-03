# Unfolding Project Context

## Project Overview

This project, 'unfolding', is a Python utility for unfolding band structures of various Bloch waves, including electrons, phonons, magnons, etc. The primary application demonstrated is phonon band unfolding using methods described in P. B. Allen et al. Phys Rev B 87, 085322 (2013). It provides tools to analyze supercell calculations by mapping the results back to the Brillouin zone of a primitive cell.

The core functionality is implemented in several modules within the `unfolding` package:
- `unfolder.py`: A general class for unfolding Bloch waves.
- `phonon_unfolder.py`: A specialized class for phonon unfolding.
- `phonopy_unfolder.py`: Tools for interfacing with Phonopy to perform phonon unfolding.
- `DDB_unfolder.py`: Tools for interfacing with Abinit DDB files for phonon unfolding.
- `plotphon.py`: Utilities for plotting phonon band structures with weights.

## Installation and Dependencies

The project is a Python package. Installation requires:
- Python
- setuptools

Dependencies specified in `setup.py`:
- numpy
- matplotlib
- ase (Atomic Simulation Environment)

Additional dependencies for specific functionalities:
- phonopy: For using `phonopy_unfolder.py`.
- abinit and abipy: For using `DDB_unfolder.py`.
- Wannier90/PROCAR/pyprocar/minimulti: For electronic/magnon unfolding (mentioned in tutorial).

To install in development mode:
```bash
pip install -e .
```

To install normally:
```bash
python setup.py install
```

## Usage Examples

The `examples` directory contains practical use cases. The `doc/tuturial.md` file provides detailed instructions for several scenarios:

1.  **Phonon unfolding with Phonopy**: Uses `FORCE_CONSTANTS` and `SPOSCAR` files from a Phonopy calculation on a supercell to unfold bands back to the primitive cell's Brillouin zone. The main function is `phonopy_unfold` from `unfolding.phonopy_unfolder`.
2.  **Phonon unfolding with Abinit DDB**: Uses an `out_DDB` file from Abinit to unfold phonon bands. The main class is `DDB_unfolder` from `unfolding.DDB_unfolder`.
3.  **Electron/Magnon unfolding**: Mentions potential workflows with Wannier90, PROCAR (via PyProcar), and minimulti, though specific examples are limited in this repository.

## Development Notes

The project is structured as a standard Python package. The core logic resides in the `unfolding` module. Key classes are `Unfolder` and `phonon_unfolder` for the general and phonon-specific unfolding algorithms, respectively. Plotting is handled by `plotphon.py`. Integration with external codes like Phonopy and Abinit is provided through dedicated modules.

There are no explicit building or testing commands defined in `setup.py` beyond installation. Testing would likely involve running the provided examples or scripts manually. The code appears to be in an alpha development stage based on the `setup.py` classifier.