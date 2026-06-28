# Neural Decoder for the [[144, 12, 12]] Gross Code

A neural decoder for the **Gross code** (a bivariate-bicycle qLDPC code), reproducing the
approach of *"Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation"*
([arXiv:2604.08358](https://arxiv.org/abs/2604.08358)) at a scale that runs on a laptop.

The network ingests the spacetime syndrome (detector pattern over rounds) and predicts,
for each of the **12 logical qubits**, whether a logical error occurred — trained with
binary cross-entropy, exactly as in the paper.

## What this is (and isn't)

- ✅ Real `[[144, 12, 12]]` Gross code, built from scratch and verified (CSS commutation,
  k = 12, full-rank logical pairing).
- ✅ Paper-style architecture: detector embedding → bottleneck residual blocks → torus
  convolution (Gross-code monomial offsets) → 12-logit head, with SiLU + BatchNorm.
- ✅ Two noise models: fast **phenomenological** and paper-faithful **circuit-level**
  (Bravyi et al. depth-7 syndrome-extraction schedule, validated for deterministic
  detectors).
- ✅ Paper training recipe available (Muon + Lion optimizers, EMA) and three size presets.
- ⚠️ Scaled down for local hardware. The paper trains H=256/L=14 on ~3×10⁸ examples on
  GPUs; the `tiny`/`small` presets train on a Mac (MPS). The default noise model is
  phenomenological (circuit-level is available via a flag).

## The code, briefly

`[[n, k, d]]` = `[[144, 12, 12]]`: **144 physical qubits** encode **12 logical qubits**
with **distance 12**. Compared to a distance-12 surface code (`[[144, 1, 12]]`, 1 logical
qubit), the Gross code protects 12× more logical qubits with the same physical count —
that better encoding rate is why bivariate-bicycle codes are of practical interest.

## Install

```bash
pip install -r requirements.txt   # torch, numpy, stim
```

## Quickstart

```bash
# 1. Generate data (phenomenological, ~10s)
python data_gen.py --noise phenomenological

# 2. Train a small conv decoder
python train.py --model conv --size small --epochs 50

# 3. ...or stream fresh data on the fly (no dataset files, no overfitting)
python train.py --model conv --size small --on-the-fly --steps 15000
```

## Data generation (`data_gen.py`)

```bash
python data_gen.py --noise phenomenological              # 1-channel detectors (X-checks)
python data_gen.py --noise circuit -p 0.002              # 2-channel detectors (X & Z), depth-7 circuit
python data_gen.py -p 0.01 -n 50000 --out ood_           # an OOD test set at a shifted error rate
```

Outputs `syndromes.npy` `[N, C, R, 12, 6]` and `observables.npy` `[N, 12]`.
- **phenomenological**: data + measurement errors over rounds. Fast, fully verifiable.
- **circuit**: full gate-level Stim circuit with depolarizing noise (`bb_circuit.py`).

## Models and sizes

Two architectures, three presets each (param counts, phenomenological input):

| size  | hidden / layers | conv   | topo   |
|-------|-----------------|--------|--------|
| tiny  | H=24,  L=3      | 23k    | 31k    |
| small | H=64,  L=8      | 92k    | 100k   |
| paper | H=256, L=14     | 1.21M  | 1.22M  |

- **conv** (`torus_conv_decoder.py`): the torus-convolution decoder.
- **topo** (`torus_conv_topo_decoder.py`): same backbone plus a differentiable
  multi-scale structural branch fused at the head.

## Training (`train.py`)

```bash
# disk data, fixed dataset
python train.py --model conv --size small --epochs 50 --log-csv run.csv

# on-the-fly streaming (recommended — effectively unlimited data)
python train.py --model topo --size small --on-the-fly --steps 15000

# paper recipe (needs a GPU)
python train.py --model conv --size paper --on-the-fly --steps 50000 \
    --batch-size 256 --optimizer muon-lion --ema
```

Logging shows wall-time, throughput, train/val BCE, per-observable error, and logical
error rate, with best-weights saving (`<out>_best.pth`) and an optional CSV (`--log-csv`).

**Metrics**: *per-observable error* (fraction of the 12 logits wrong) and *logical error
rate* (fraction of shots with any of the 12 wrong). The validation set is an
in-distribution held-out split (`--val-frac`), or a fixed generated set in on-the-fly mode.

## Out-of-distribution evaluation

Train at one error rate, test at another (or different rounds):

```bash
python train.py --model topo --size small --on-the-fly -p 0.005 --steps 15000
python data_gen.py -p 0.01 -n 50000 --out ood_
python train.py --eval-only --model topo --size small \
    --weights gross_topo_small_best.pth --data ood_
```

`--size`/`--model` must match training, and the noise model (channel count) must match.

## Project structure

```
bb_code.py                  # [[144,12,12]] construction + logical operators (GF(2))
bb_circuit.py               # Bravyi depth-7 circuit-level Stim circuit
data_gen.py                 # phenomenological / circuit data generation
torus_conv_decoder.py       # conv decoder (12 logits)
torus_conv_topo_decoder.py  # hybrid conv + structural decoder (12 logits)
optim.py                    # Muon + Lion optimizers, EMA
train.py                    # training, on-the-fly streaming, logging, OOD eval
```

Each module runs standalone (`python bb_code.py`, `python bb_circuit.py`, etc.) to
self-verify its construction.

## Hardware notes

- `tiny` / `small` train on a Mac (MPS) or any GPU.
- `paper` (1.2M params) wants a GPU. Memory is activation-bound by batch size: ~16 GB
  (e.g. AWS `g4dn.xlarge`, T4) fits `--batch-size 128`; a 24 GB card fits ~256.
- These are block decoders evaluated for **accuracy**; real-time QPU decoding (µs-scale,
  superconducting) is a separate FPGA/ASIC problem this repo does not address.

## Differences from the paper

- Phenomenological noise by default (circuit-level available).
- Smaller models / fewer training examples than the paper's reference run.
- Fused linear readout head instead of per-observable heads.
- Distance check for the circuit relies on noiseless-determinism + matching the published
  depth-7 schedule (full BB-code distance search is expensive).

## References

- Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation —
  [arXiv:2604.08358](https://arxiv.org/abs/2604.08358)
- Bravyi et al., High-threshold and low-overhead fault-tolerant quantum memory —
  [arXiv:2308.07915](https://arxiv.org/abs/2308.07915) ·
  [reference code](https://github.com/sbravyi/BivariateBicycleCodes)
