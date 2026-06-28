"""
Data generation for the [[144,12,12]] Gross code (X-basis memory, 12 logical observables).

Two noise models (set NOISE_MODEL below or pass --noise):
  - "phenomenological": data + measurement errors over rounds, X-checks only.
      Fast, fully verifiable. Output channels C=1.
  - "circuit": Bravyi depth-7 syndrome-extraction circuit with uniform circuit-level
      depolarizing noise (gate/reset/measure/idle). Paper-faithful noise. Output C=2
      (X-check and Z-check detector channels). See bb_circuit.py.

Both save:
  syndromes.npy   : float32 [N, C, R, 12, 6]   (spacetime detectors on the check torus)
  observables.npy : float32 [N, 12]            (logical-X observable flips)

OOD testing: raise PHYSICAL_ERROR_RATE at test time, or change ROUNDS.
"""
import argparse
import numpy as np
from bb_code import build_gross_code

PHYSICAL_ERROR_RATE = 0.005
ROUNDS = 12
NUM_SAMPLES = 200000
CHUNK = 20000
SEED = 12
NOISE_MODEL = "phenomenological"   # or "circuit"


# Cache the code so repeated (on-the-fly) calls don't rebuild it.
_CODE = None


def _code_cached():
    global _CODE
    if _CODE is None:
        _CODE = build_gross_code()
    return _CODE


def sample_phenomenological_batch(p, rounds, batch, rng):
    """Generate a single fresh batch (for on-the-fly training).
    Returns x [batch,1,R,L,M] float32, y [batch,12] float32."""
    code = _code_cached()
    H_X = code["H_X"].astype(np.int32)
    L_X = code["L_X"].astype(np.int32)
    L, M = code["L"], code["M"]
    n_checks, n_qubits = H_X.shape
    R = rounds + 1
    accum = np.zeros((batch, n_qubits), dtype=np.int32)
    frames = np.zeros((batch, R, n_checks), dtype=np.int32)
    for t in range(rounds):
        accum ^= (rng.random((batch, n_qubits)) < p).astype(np.int32)
        s = (accum @ H_X.T) % 2
        frames[:, t] = s ^ (rng.random((batch, n_checks)) < p).astype(np.int32)
    frames[:, rounds] = (accum @ H_X.T) % 2
    detectors = frames.copy()
    detectors[:, 1:] ^= frames[:, :-1]
    x = detectors.reshape(batch, 1, R, L, M).astype(np.float32)
    y = ((accum @ L_X.T) % 2).astype(np.float32)
    return x, y


def gen_phenomenological(p, rounds, num_samples, seed):
    code = _code_cached()
    H_X = code["H_X"].astype(np.int32)
    L_X = code["L_X"].astype(np.int32)
    L, M = code["L"], code["M"]
    n_checks, n_qubits, n_logical = H_X.shape[0], H_X.shape[1], L_X.shape[0]
    R = rounds + 1
    rng = np.random.default_rng(seed)

    syndromes = np.zeros((num_samples, 1, R, L, M), dtype=np.float32)
    observables = np.zeros((num_samples, n_logical), dtype=np.float32)

    for start in range(0, num_samples, CHUNK):
        stop = min(start + CHUNK, num_samples)
        b = stop - start
        accum = np.zeros((b, n_qubits), dtype=np.int32)
        frames = np.zeros((b, R, n_checks), dtype=np.int32)
        for t in range(rounds):
            accum ^= (rng.random((b, n_qubits)) < p).astype(np.int32)
            s = (accum @ H_X.T) % 2
            frames[:, t] = s ^ (rng.random((b, n_checks)) < p).astype(np.int32)
        frames[:, rounds] = (accum @ H_X.T) % 2
        detectors = frames.copy()
        detectors[:, 1:] ^= frames[:, :-1]
        syndromes[start:stop, 0] = detectors.reshape(b, R, L, M).astype(np.float32)
        observables[start:stop] = ((accum @ L_X.T) % 2).astype(np.float32)
    return syndromes, observables


def main(noise_model, p, rounds, num_samples, seed, out_prefix=""):
    print(f"=== GROSS CODE [[144,12,12]] DATA GEN (noise={noise_model}, p={p}, "
          f"rounds={rounds}, N={num_samples}) ===")
    if noise_model == "phenomenological":
        syndromes, observables = gen_phenomenological(p, rounds, num_samples, seed)
    elif noise_model == "circuit":
        from bb_circuit import circuit_to_arrays
        syndromes, observables = circuit_to_arrays(
            rounds=rounds, p=p, num_samples=num_samples, chunk=CHUNK, seed=seed)
    else:
        raise ValueError(noise_model)

    print("\n--- Verification ---")
    det_per_row = syndromes.reshape(num_samples, -1).sum(axis=1)
    print(f"channels={syndromes.shape[1]} shape={syndromes.shape}")
    print(f"detectors/shot: min={det_per_row.min():.0f} max={det_per_row.max():.0f} "
          f"mean={det_per_row.mean():.2f}")
    print(f"per-observable flip rate: {observables.mean(axis=0).round(3)}")
    print(f"shots with >=1 logical flip: {(observables.sum(1) > 0).mean():.3f}")

    sfile = f"{out_prefix}syndromes.npy"
    ofile = f"{out_prefix}observables.npy"
    np.save(sfile, syndromes)
    np.save(ofile, observables)
    print(f"\nSaved {sfile} {syndromes.shape} and {ofile} {observables.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", choices=["phenomenological", "circuit"], default=NOISE_MODEL)
    ap.add_argument("-p", type=float, default=PHYSICAL_ERROR_RATE)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("-n", "--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="", help="filename prefix, e.g. 'ood_' for OOD sets")
    args = ap.parse_args()
    main(args.noise, args.p, args.rounds, args.num_samples, args.seed, args.out)
