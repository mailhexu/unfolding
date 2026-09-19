"""Reference toy LCAO model for tests, derivations, and docs examples.

A 1D chain with lattice constant 1 and one atom per cell carrying two
non-orthogonal orbitals (s-like, p-like). Real-space matrices have
nearest-neighbour range only:

    S(0)  = [[1.0,  0.05], [0.05, 1.0]]
    S(+1) = [[0.20,  0.03], [0.02, 0.15]],   S(-1) = S(+1).T
    H(0)  = diag(-1.0, 0.5)
    H(+1) = -0.3 * [[1.0, 0.1], [0.05, 0.6]], H(-1) = H(+1).T

k-space convention (matches HamiltonIO's R_to_onek convention 2):

    M(k) = sum_R M(R) exp(2 pi i k R)

Both M(k) are real symmetric by construction; S(k) is positive definite for
all k (diagonally dominant), so the generalized eigenproblem H(k) c = e S(k) c
is well posed. Because the matrices are 2x2, the generalized eigenvalues have
the closed form  det(H - e S) = 0  ->  e = (-b -/+ sqrt(b^2 - 4 a c)) / (2 a)
with a = det(S), b = -(h00 s11 + s00 h11 - h01 s10 - s01 h10), c = det(H).
"""
import numpy as np


def toy_lcao_model():
    """Return the toy model as real-space dicts plus k-space callables."""
    S1 = np.array([[0.20, 0.03], [0.02, 0.15]])
    H1 = -0.3 * np.array([[1.0, 0.1], [0.05, 0.6]])
    S = {0: np.array([[1.0, 0.05], [0.05, 1.0]]), 1: S1, -1: S1.T}
    H = {0: np.array([[-1.0, 0.0], [0.0, 0.5]]), 1: H1, -1: H1.T}

    def M_of_k(M, k):
        return M[0] + M[1] * np.exp(2j * np.pi * k) + M[-1] * np.exp(-2j * np.pi * k)

    return {
        "H": H,
        "S": S,
        "H_of_k": lambda k: M_of_k(H, k),
        "S_of_k": lambda k: M_of_k(S, k),
    }


def closed_form_bands(m, k):
    """Analytic generalized eigenvalues of the toy model at scalar k.

    H(k), S(k) are complex Hermitian; a = det(S) and c = det(H) are real, and
    the linear coefficient b is real by Hermiticity
    (h01 s10 + s01 h10 = 2 Re(h01 conj(s01))).
    """
    H = m["H_of_k"](k)
    S = m["S_of_k"](k)
    a = np.linalg.det(S)
    b = -(H[0, 0] * S[1, 1] + S[0, 0] * H[1, 1] - H[0, 1] * S[1, 0] - S[0, 1] * H[1, 0])
    c = np.linalg.det(H)
    b = b.real
    disc = np.sqrt(b * b - 4.0 * a.real * c.real)
    return np.array([(-b - disc) / (2.0 * a.real), (-b + disc) / (2.0 * a.real)])
