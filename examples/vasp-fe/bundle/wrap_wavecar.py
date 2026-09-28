#!/usr/bin/env python
"""Rewrite a VASP WAVECAR with wrapped fractional k-points (in place: IN OUT).

Some VASP builds (this one's KPAR writer among them) store the KPOINTS
list verbatim, including unwrapped fractional components >1. Readers that
reconstruct G-vectors from a k-point-centered sphere with loop bounds
sized for the primitive-cell cutoff (pymatgen's Wavecar, and therefore
HamiltonIO.vasp.read_wavecar) then reject the file. The plane-wave sphere
is invariant under k -> k mod 1 (an integer shift of the G list), so
rewriting only the three k-point doubles in each k-record fixes the file
without touching coefficients, occupations or eigenvalues.

Usage: python wrap_wavecar.py WAVECAR_IN WAVECAR_OUT
"""
import sys

import numpy as np


def main(src, dst):
    fin = open(src, "rb")
    recl, spin, rtag = np.fromfile(fin, dtype=np.float64, count=3).astype(int)
    fin.seek(0)
    header = fin.read(2 * recl)  # record 1 (recl, spin, rtag) + record 2
    nk, nb = np.frombuffer(header[recl:recl + 16], dtype=np.float64)
    nk, nb = int(nk), int(nb)
    fout = open(dst, "wb")
    fout.write(header)
    pos = 2 * recl
    for ik in range(nk):
        for _ in range(spin):
            fin.seek(pos)
            raw = fin.read(recl)
            rec = np.frombuffer(raw, dtype=np.float64).copy()
            rec[1:4] %= 1.0
            # nplane double + wrapped kpoint triple + rest of the record
            fout.write(raw[:8] + rec[1:4].tobytes() + raw[32:])
            pos += recl
            for _ in range(nb):
                fout.write(fin.read(recl))
                pos += recl
    fin.close()
    fout.close()
    import os

    if os.path.getsize(dst) != os.path.getsize(src):
        raise RuntimeError(f"{dst}: size differs from {src}; layout unknown")
    print(f"wrapped {nk} x {spin} k-records: {src} -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
