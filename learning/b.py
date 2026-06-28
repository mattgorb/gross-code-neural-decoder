import numpy as np
'''
p = 0.1
s0, s1 = 1, 1

# 1. Error probability vectors for qubits a, b, c
A = B = C = np.array([1-p, p])

# 2. Syndrome constraints (XOR delta tensors flattened to 2D matrices)
# If s=0, picks matching elements (I). If s=1, picks flipping elements (X).
S0 = np.eye(2) if s0 == 0 else np.eye(2)[::-1]
S1 = np.eye(2) if s1 == 0 else np.eye(2)[::-1]

print(S0)
print(S1)

# 3. Contract the network
# 'a,ab,b,bc,c->abc' multiplies everything along the 1D chain topology.
# The final advanced index expression handles the logical class reduction (a^b^c).
state_probs = np.einsum('a,ab,b,bc,c->abc', A, S0, B, S1, C)
bucket = [state_probs.trace(), state_probs[::-1].trace()]

print("coset probs:", bucket, " -> ML class", np.argmax(bucket))

'''





import numpy as np

p = 0.4
s0, s1, s2 = 1, 1, 0

# 1. Error probability vectors for qubits a through g
A = B = C = D = E = F = G = np.array([1-p, p])

# 2. Multi-qubit Syndrome Tensors (4D delta tensors)
# Each detector checks 4 qubits, so each needs a 4-index tensor.
# The entry is 1.0 only if the XOR sum of the 4 qubits matches the syndrome.

S0_tensor = np.zeros((2, 2, 2, 2))
for a in (0, 1):
    for c in (0, 1):
        for e in (0, 1):
            for g in (0, 1):
                if (a ^ c ^ e ^ g) == s0:
                    S0_tensor[a, c, e, g] = 1.0

S1_tensor = np.zeros((2, 2, 2, 2))
for b in (0, 1):
    for c in (0, 1):
        for f in (0, 1):
            for g in (0, 1):
                if (b ^ c ^ f ^ g) == s1:
                    S1_tensor[b, c, f, g] = 1.0

S2_tensor = np.zeros((2, 2, 2, 2))
for d in (0, 1):
    for e in (0, 1):
        for f in (0, 1):
            for g in (0, 1):
                if (d ^ e ^ f ^ g) == s2:
                    S2_tensor[d, e, f, g] = 1.0



print(S0_tensor)


# 3. The Steane Code Einsum
# We connect the qubits to the exact detector slots specified by your H matrix.
state_probs = np.einsum(
    'a, b, c, d, e, f, g, aceg, bcfg, defg -> abcdefg',
    A, B, C, D, E, F, G, S0_tensor, S1_tensor, S2_tensor
)

print('derp')
print(state_probs)