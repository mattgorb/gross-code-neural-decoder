"""
Circuit-level Stim circuit for the [[144,12,12]] Gross code (X-basis memory),
using the Bravyi et al. depth-7 syndrome-extraction schedule.

Schedule (one round, 7 CNOT steps). At step t, every X-check applies a CNOT to its
neighbour sX[t] and every Z-check to its neighbour sZ[t] ('idle' = no gate that step):

    sX = ['idle', 1, 4, 3, 5, 0, 2]
    sZ = [3, 5, 0, 1, 2, 4, 'idle']

Each check has 6 data neighbours indexed 0..5:
    X-check: 0,1,2 = L-data via A1,A2,A3 ; 3,4,5 = R-data via B1,B2,B3
    Z-check: 0,1,2 = R-data via A1^T,A2^T,A3^T ; 3,4,5 = L-data via B1^T,B2^T,B3^T
(the A/B roles reverse for Z, matching H_Z = [B^T | A^T]).

This interleaving is what gives the good (~0.7%) threshold; the order of the X/Z CNOTs
on shared data qubits is the part that controls hook errors. Correctness is verified
below by checking that the noiseless circuit has fully deterministic detectors.

Qubit indices:
  data  : 0..143   (0..71 = L sector, 72..143 = R sector; matches H_X/H_Z columns)
  X-anc : 144..215
  Z-anc : 216..287
"""
import numpy as np
import stim
from bb_code import build_gross_code, monomial_matrices

_code = build_gross_code()
H_X = _code["H_X"]
H_Z = _code["H_Z"]
L_X = _code["L_X"]
N_DATA = H_X.shape[1]          # 144
N_CHK = H_X.shape[0]           # 72  (X-checks == Z-checks count)
LM = N_DATA // 2               # 72
L_GRID = _code["L"]            # 12  (check torus rows)
M = _code["M"]                 # 6   (check torus cols)
X_ANC0 = N_DATA                # 144
Z_ANC0 = N_DATA + N_CHK        # 216

A_MONS, B_MONS = monomial_matrices()   # each a list of 3 (72x72) matrices

sX = ["idle", 1, 4, 3, 5, 0, 2]
sZ = [3, 5, 0, 1, 2, 4, "idle"]


def _col(mat, i):
    """The single data-qubit index where row i of `mat` is 1 (monomial => one nonzero)."""
    return int(np.flatnonzero(mat[i])[0])


# Precompute each check's 6 data-qubit neighbours, indexed 0..5.
X_NB = [[None] * 6 for _ in range(N_CHK)]
Z_NB = [[None] * 6 for _ in range(N_CHK)]
for i in range(N_CHK):
    # X-check: A on L-data (0..71), B on R-data (72..143)
    X_NB[i][0] = _col(A_MONS[0], i)
    X_NB[i][1] = _col(A_MONS[1], i)
    X_NB[i][2] = _col(A_MONS[2], i)
    X_NB[i][3] = LM + _col(B_MONS[0], i)
    X_NB[i][4] = LM + _col(B_MONS[1], i)
    X_NB[i][5] = LM + _col(B_MONS[2], i)
    # Z-check: slots 0-2 = B^T on L-data, slots 3-5 = A^T on R-data.
    # (This grouping is what makes the Bravyi sX/sZ schedule produce deterministic
    # detectors -- verified by the noiseless determinism check below.)
    Z_NB[i][0] = _col(B_MONS[0].T, i)
    Z_NB[i][1] = _col(B_MONS[1].T, i)
    Z_NB[i][2] = _col(B_MONS[2].T, i)
    Z_NB[i][3] = LM + _col(A_MONS[0].T, i)
    Z_NB[i][4] = LM + _col(A_MONS[1].T, i)
    Z_NB[i][5] = LM + _col(A_MONS[2].T, i)

# Sanity: the neighbour sets must equal the parity-check supports.
for i in range(N_CHK):
    assert set(X_NB[i]) == set(np.flatnonzero(H_X[i])), "X neighbour/support mismatch"
    assert set(Z_NB[i]) == set(np.flatnonzero(H_Z[i])), "Z neighbour/support mismatch"

X_SUPPORT = [np.flatnonzero(H_X[g]).tolist() for g in range(N_CHK)]
L_SUPPORT = [np.flatnonzero(L_X[l]).tolist() for l in range(L_X.shape[0])]


def build_circuit(rounds=12, p=0.005):
    """X-basis memory circuit with uniform circuit-level depolarizing noise.
    p=0 -> noiseless (for determinism validation)."""
    c = stim.Circuit()
    data = list(range(N_DATA))
    x_anc = list(range(X_ANC0, X_ANC0 + N_CHK))
    z_anc = list(range(Z_ANC0, Z_ANC0 + N_CHK))
    all_q = data + x_anc + z_anc

    meas_count = 0
    z_rec = [dict() for _ in range(rounds)]
    x_rec = [dict() for _ in range(rounds)]

    def add_measure(targets, kind):
        nonlocal meas_count
        if p > 0:
            c.append(kind, targets, p)
        else:
            c.append(kind, targets)
        idx = {t: meas_count + j for j, t in enumerate(targets)}
        meas_count += len(targets)
        return idx

    # init data in |+>
    c.append("RX", data)
    if p > 0:
        c.append("Z_ERROR", data, p)

    for t in range(rounds):
        # prepare ancillas: X in |+>, Z in |0>
        c.append("RX", x_anc)
        c.append("R", z_anc)
        if p > 0:
            c.append("Z_ERROR", x_anc, p)
            c.append("X_ERROR", z_anc, p)

        # 7 interleaved CNOT steps
        for step in range(7):
            pairs = []          # (control, target)
            if sX[step] != "idle":
                k = sX[step]
                for i in range(N_CHK):
                    pairs.append((x_anc[i], X_NB[i][k]))   # X-check: anc -> data
            if sZ[step] != "idle":
                k = sZ[step]
                for i in range(N_CHK):
                    pairs.append((Z_NB[i][k], z_anc[i]))   # Z-check: data -> anc
            touched = set()
            for ctrl, tgt in pairs:
                c.append("CX", [ctrl, tgt])
                if p > 0:
                    c.append("DEPOLARIZE2", [ctrl, tgt], p)
                touched.add(ctrl)
                touched.add(tgt)
            if p > 0:                       # idle depolarizing on untouched qubits
                idle = [q for q in all_q if q not in touched]
                if idle:
                    c.append("DEPOLARIZE1", idle, p)

        # measure + reset ancillas
        zi = add_measure(z_anc, "MR")
        xi = add_measure(x_anc, "MRX")
        for i in range(N_CHK):
            z_rec[t][i] = zi[z_anc[i]]
            x_rec[t][i] = xi[x_anc[i]]

    # final data readout in X
    di = add_measure(data, "MX")

    def rec(abs_idx):
        return stim.target_rec(abs_idx - meas_count)

    # Detector coordinates: (channel, round, i, j) with channel 0 = X-check, 1 = Z-check,
    # and check index g = i*M + j on the L x M torus. R dimension spans rounds 0..rounds.
    def ij(g):
        return (g // M, g % M)

    # Z-check detectors: round-to-round change only (random at t=0 in X basis)
    for t in range(1, rounds):
        for i in range(N_CHK):
            r, s = ij(i)
            c.append("DETECTOR", [rec(z_rec[t][i]), rec(z_rec[t - 1][i])], (1, t, r, s))
    # X-check detectors: deterministic from |+> prep
    for i in range(N_CHK):
        r, s = ij(i)
        c.append("DETECTOR", [rec(x_rec[0][i])], (0, 0, r, s))
    for t in range(1, rounds):
        for i in range(N_CHK):
            r, s = ij(i)
            c.append("DETECTOR", [rec(x_rec[t][i]), rec(x_rec[t - 1][i])], (0, t, r, s))
    # final X-check detectors from data readout (round index = rounds)
    for g in range(N_CHK):
        r, s = ij(g)
        targets = [rec(x_rec[rounds - 1][g])] + [rec(di[d]) for d in X_SUPPORT[g]]
        c.append("DETECTOR", targets, (0, rounds, r, s))

    # 12 logical-X observables on final data readout
    for l, support in enumerate(L_SUPPORT):
        c.append("OBSERVABLE_INCLUDE", [rec(di[d]) for d in support], l)

    return c


def _detector_coord_index(circuit):
    """Per-detector (channel, round, i, j) arrays for scattering a flat detector vector
    onto the [2, R, L_GRID, M] grid."""
    n_det = circuit.num_detectors
    coords = circuit.get_detector_coordinates()
    ch = np.zeros(n_det, dtype=np.int64)
    rr = np.zeros(n_det, dtype=np.int64)
    ii = np.zeros(n_det, dtype=np.int64)
    jj = np.zeros(n_det, dtype=np.int64)
    for d, co in coords.items():
        ch[d], rr[d], ii[d], jj[d] = int(co[0]), int(co[1]), int(co[2]), int(co[3])
    return ch, rr, ii, jj


def _scatter(dets, ch, rr, ii, jj, R):
    b = dets.shape[0]
    grid = np.zeros((b, 2, R, L_GRID, M), dtype=np.float32)
    grid[:, ch, rr, ii, jj] = dets.astype(np.float32)
    return grid


def circuit_to_arrays(rounds=12, p=0.005, num_samples=20000, chunk=20000, seed=0):
    """Sample the circuit-level Gross code and return decoder-ready arrays.

    Returns:
      syndromes  : float32 [N, 2, R, L_GRID, M]  (channel 0 = X-checks, 1 = Z-checks)
      observables: float32 [N, 12]
    where R = rounds + 1 (detector round indices 0..rounds).
    """
    circuit = build_circuit(rounds=rounds, p=p)
    R = rounds + 1
    ch, rr, ii, jj = _detector_coord_index(circuit)
    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes = np.zeros((num_samples, 2, R, L_GRID, M), dtype=np.float32)
    observables = np.zeros((num_samples, 12), dtype=np.float32)
    for start in range(0, num_samples, chunk):
        stop = min(start + chunk, num_samples)
        dets, obs = sampler.sample(shots=stop - start, separate_observables=True)
        syndromes[start:stop] = _scatter(dets, ch, rr, ii, jj, R)
        observables[start:stop] = obs.astype(np.float32)
    return syndromes, observables


# Cache compiled samplers so on-the-fly streaming doesn't rebuild the circuit each step.
_STREAM_CACHE = {}


def sample_circuit_batch(rounds, p, batch, seed=0):
    """Stream one fresh circuit-level batch (for on-the-fly training).
    Returns x [batch, 2, R, L_GRID, M] float32, y [batch, 12] float32."""
    key = (rounds, p, seed)
    if key not in _STREAM_CACHE:
        circuit = build_circuit(rounds=rounds, p=p)
        ch, rr, ii, jj = _detector_coord_index(circuit)
        sampler = circuit.compile_detector_sampler(seed=seed)
        _STREAM_CACHE[key] = (sampler, ch, rr, ii, jj, rounds + 1)
    sampler, ch, rr, ii, jj, R = _STREAM_CACHE[key]
    dets, obs = sampler.sample(shots=batch, separate_observables=True)
    return _scatter(dets, ch, rr, ii, jj, R), obs.astype(np.float32)


if __name__ == "__main__":
    noiseless = build_circuit(rounds=12, p=0.0)
    print(f"circuit: {noiseless.num_detectors} detectors, {noiseless.num_observables} observables")
    dets, obs = noiseless.compile_detector_sampler().sample(shots=2000, separate_observables=True)
    print("noiseless detectors all zero :", not dets.any(), "(must be True)")
    print("noiseless observables all zero:", not obs.any(), "(must be True)")

    print("\nlogical error rate vs p (depth-7 schedule):")
    for p in [0.001, 0.002, 0.003, 0.005, 0.007]:
        c = build_circuit(rounds=12, p=p)
        d, o = c.compile_detector_sampler().sample(shots=8000, separate_observables=True)
        print(f"  p={p:<6} dets/shot {d.sum(1).mean():6.1f} | mean obs flip {o.mean():.4f} "
              f"| any-logical {(o.sum(1) > 0).mean():.4f}")
