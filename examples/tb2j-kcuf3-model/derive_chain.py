"""Verify the isolated AFM-chain LSWT oracle used by run.py with SymPy."""

import numpy as np
from sympy import Matrix, cos, eye, pi, simplify, sin, symbols

from run import JC_PAIR, JC_EV, build_magnon

J, S, k, energy = symbols("J S k energy", real=True)
A = 2 * J * S
B = A * cos(pi * k)
# Positive bosonic BdG branch of a nearest-neighbor bipartite chain.
dynamical = Matrix([[A, B], [-B, -A]])
characteristic = (energy * eye(2) - dynamical).det()
assert simplify(characteristic - (energy**2 - (2 * J * S * sin(pi * k))**2)) == 0

# Independent numerical seal against the TB2J model, away from the
# Goldstone point where TB2J applies its 1e-6-eV Cholesky shift.
ks = np.array([0.1, 0.25, 0.4])
points = np.column_stack((np.zeros_like(ks), np.zeros_like(ks), ks))
computed = build_magnon(2, jc=JC_EV, jp=0).get_magnon_eigenstates(points).energies[:, -1]
expected = abs(JC_PAIR) * np.abs(np.sin(np.pi * ks))  # S=1/2
np.testing.assert_allclose(computed, expected, atol=5e-6, rtol=0)
print("SymPy AFM-chain identity and TB2J numeric oracle pass")
