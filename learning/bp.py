import numpy as np

def bp_decode(H, syndrome, p=0.05, max_iter=50, eps=1e-9):
    H = np.asarray(H)
    mask  = H.astype(bool)                      # which (check, qubit) pairs are edges
    prior = np.full(H.shape[1], np.log((1-p)/p))# prior LLR (positive => probably not flipped)
    signs = 1 - 2*syndrome                       # +1 if detector quiet, -1 if it fired

    M_vc = mask * prior                          # qubit->check messages, init to prior
    M_cv = np.zeros_like(H, dtype=float)         # check->qubit messages

    for _ in range(max_iter):
        # check -> qubit: tanh product rule, excluding the target qubit
        t        = np.where(mask, np.tanh(np.clip(M_vc/2, -30, 30)), 1.0)
        row_prod = t.prod(axis=1, keepdims=True)            # product over each check's qubits
        M_cv     = mask * signs[:,None] * 2*np.arctanh(np.clip(row_prod/t, -1+eps, 1-eps))

        # qubit -> check: prior + all OTHER checks; posterior L falls out for free
        L    = prior + M_cv.sum(axis=0)                      # posterior LLR per qubit
        M_vc = mask * (L - M_cv)                             # subtract self => "all other checks"

        x = (L < 0).astype(int)                             # LLR<0 => flipped
        if np.array_equal((H @ x) % 2, syndrome):
            break
    return x

# --- demo: Hamming(7,4), inject one error, recover it from the syndrome ---
# 7 qubits, 3 detectors
H = np.array([[1,0,1,0,1,0,1],
              [0,1,1,0,0,1,1],
              [0,0,0,1,1,1,1]])
#true_error = np.zeros(7, dtype=int); true_error[0] = 1;true_error[1]=1     # qubit 2 flipped
#print(true_error)
#syndrome = (H @ true_error) % 2                            # what detectors report
syndrome = np.array([0, 0, 1])
print(syndrome)

guess = bp_decode(H, syndrome)
print("detector bits (syndrome):", syndrome)
#print("true error:              ", true_error)
print("BP recovered:            ", guess)
#print("match:", np.array_equal(true_error, guess))





