"""
Bivariate Bicycle (BB) code construction for the [[144, 12, 12]] Gross code.

Reference: Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum
memory" (Nature 2024). The Gross code uses cyclic groups Z_l x Z_m with l=12, m=6
(so l*m = 72, and n = 2*l*m = 144 physical qubits) and the monomials:

    A = x^3 + y + y^2
    B = y^3 + x + x^2

where x is the cyclic shift on Z_l and y the cyclic shift on Z_m.

This module builds the CSS parity-check matrices H_X, H_Z and a basis of logical
operators using GF(2) linear algebra (no external LDPC libraries required).
"""
import numpy as np

# Gross code group sizes. x has order L, y has order M.  L*M = 72, n = 144.
L = 12
M = 6
N_QUBITS = 2 * L * M  # 144

# Monomial exponents (matching torus_conv_decoder's shift definition).
SHIFTS_A = [(3, 0), (0, 1), (0, 2)]  # x^3, y, y^2
SHIFTS_B = [(0, 3), (1, 0), (2, 0)]  # y^3, x, x^2


# ---------------------------------------------------------------------------
# GF(2) linear algebra helpers
# ---------------------------------------------------------------------------
def gf2_rref(matrix):
    """Reduced row echelon form over GF(2). Returns (rref, pivot_columns)."""
    m = matrix.copy().astype(np.uint8) % 2
    rows, cols = m.shape
    pivots = []
    r = 0
    for c in range(cols):
        # find a pivot row at or below r with a 1 in column c
        pivot = np.where(m[r:, c] == 1)[0]
        if pivot.size == 0:
            continue
        pr = r + pivot[0]
        m[[r, pr]] = m[[pr, r]]
        # eliminate this column from all other rows
        mask = m[:, c] == 1
        mask[r] = False
        m[mask] ^= m[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return m, pivots


def gf2_rank(matrix):
    return len(gf2_rref(matrix)[1])


def gf2_nullspace(matrix):
    """Basis (as row vectors) of the right null space {v : matrix @ v = 0} over GF(2)."""
    m = matrix.astype(np.uint8) % 2
    rref, pivots = gf2_rref(m)
    cols = m.shape[1]
    pivot_set = set(pivots)
    free = [c for c in range(cols) if c not in pivot_set]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        # back-substitute: for each pivot row, set the pivot variable
        for row_i, pc in enumerate(pivots):
            v[pc] = rref[row_i, f]
        basis.append(v)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, cols), np.uint8)


def gf2_row_reduce_basis(vectors):
    """Return an independent generating set (row-reduced) of the given GF(2) vectors."""
    if len(vectors) == 0:
        return np.zeros((0, 0), np.uint8)
    rref, pivots = gf2_rref(np.array(vectors, dtype=np.uint8))
    return rref[: len(pivots)]


def gf2_in_span(vec, basis_rref, pivots):
    """Check whether vec lies in the span described by a RREF basis with given pivots."""
    v = vec.copy().astype(np.uint8) % 2
    for row_i, pc in enumerate(pivots):
        if v[pc] == 1:
            v ^= basis_rref[row_i]
    return not v.any()


# ---------------------------------------------------------------------------
# Build the cyclic monomial matrices over Z_L x Z_M
# ---------------------------------------------------------------------------
def _cyclic_shift(size, power):
    """size x size permutation matrix for a cyclic shift by `power`."""
    s = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        s[i, (i + power) % size] = 1
    return s


def _monomial(px, py):
    """The lm x lm matrix for x^px * y^py acting on Z_L (x) tensor Z_M (y)."""
    sx = _cyclic_shift(L, px)
    sy = _cyclic_shift(M, py)
    return np.kron(sx, sy) % 2  # (L*M) x (L*M) = 72 x 72


def _poly(shifts):
    out = np.zeros((L * M, L * M), dtype=np.uint8)
    for px, py in shifts:
        out ^= _monomial(px, py)
    return out


def monomial_matrices():
    """Return the six individual monomial (circulant) matrices A1,A2,A3,B1,B2,B3,
    each (lm x lm), in the order matching SHIFTS_A then SHIFTS_B."""
    return (
        [_monomial(px, py) for px, py in SHIFTS_A],
        [_monomial(px, py) for px, py in SHIFTS_B],
    )


def build_gross_code():
    """
    Returns a dict with:
      H_X, H_Z : (72, 144) parity-check matrices (CSS)
      L_X      : (12, 144) logical X operators (X-type; detect Z errors)
      L_Z      : (12, 144) logical Z operators (Z-type; detect X errors)
      L, M, n, k
    """
    A = _poly(SHIFTS_A)  # 72 x 72
    B = _poly(SHIFTS_B)  # 72 x 72

    # CSS construction: H_X = [A | B],  H_Z = [B^T | A^T]
    H_X = np.hstack([A, B]).astype(np.uint8)
    H_Z = np.hstack([B.T, A.T]).astype(np.uint8)

    # k = n - rank(H_X) - rank(H_Z)
    rank_x = gf2_rank(H_X)
    rank_z = gf2_rank(H_Z)
    k = N_QUBITS - rank_x - rank_z

    # Logical X operators: in ker(H_Z), independent of rowspace(H_X).
    L_X = _logicals(stab_rows=H_X, commute_with=H_Z, k=k)
    # Logical Z operators: in ker(H_X), independent of rowspace(H_Z).
    L_Z = _logicals(stab_rows=H_Z, commute_with=H_X, k=k)

    return {
        "H_X": H_X, "H_Z": H_Z, "L_X": L_X, "L_Z": L_Z,
        "L": L, "M": M, "n": N_QUBITS, "k": k,
        "rank_x": rank_x, "rank_z": rank_z,
    }


def _logicals(stab_rows, commute_with, k):
    """Find k logical operators: in the null space of `commute_with`, independent of
    the rowspace of `stab_rows`."""
    null = gf2_nullspace(commute_with)            # vectors commuting with the other type
    basis_rref, pivots = gf2_rref(stab_rows)      # span of the same-type stabilizers
    basis_rref = basis_rref[: len(pivots)]
    pivots = list(pivots)

    logicals = []
    for v in null:
        if not gf2_in_span(v, basis_rref, pivots):
            # add v to the running basis so later logicals stay independent of it
            logicals.append(v.copy())
            stacked = np.vstack([basis_rref, np.array(logicals, dtype=np.uint8)])
            basis_rref, pivots = gf2_rref(stacked)
            basis_rref = basis_rref[: len(pivots)]
            pivots = list(pivots)
            if len(logicals) == k:
                break
    return np.array(logicals, dtype=np.uint8)


if __name__ == "__main__":
    code = build_gross_code()
    H_X, H_Z, L_X, L_Z = code["H_X"], code["H_Z"], code["L_X"], code["L_Z"]
    print(f"n (physical qubits): {code['n']}")
    print(f"k (logical qubits) : {code['k']}")
    print(f"rank(H_X)={code['rank_x']}  rank(H_Z)={code['rank_z']}")
    print(f"H_X shape {H_X.shape} | H_Z shape {H_Z.shape}")
    print(f"L_X shape {L_X.shape} | L_Z shape {L_Z.shape}")
    # CSS commutation: H_X H_Z^T = 0
    print("CSS check  H_X·H_Z^T == 0 :", not ((H_X @ H_Z.T) % 2).any())
    # logicals commute with the opposite stabilizers
    print("L_X·H_Z^T == 0           :", not ((L_X @ H_Z.T) % 2).any())
    print("L_Z·H_X^T == 0           :", not ((L_Z @ H_X.T) % 2).any())
    # logical symplectic pairing L_X · L_Z^T should be full rank (k x k invertible)
    pairing = (L_X @ L_Z.T) % 2
    print("rank(L_X·L_Z^T) (==k=12) :", gf2_rank(pairing))
