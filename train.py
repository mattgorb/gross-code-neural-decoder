import argparse
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PRESETS = {
    "tiny":  dict(hidden=24, layers=3),
    "small": dict(hidden=64, layers=8),
    "paper": dict(hidden=256, layers=14),
}


def get_model(which, in_channels, hidden, layers, num_logical=12):
    if which == "conv":
        from torus_conv_decoder import GrossCodeDecoder
    elif which == "topo":
        from torus_conv_topo_decoder import GrossCodeDecoder
    else:
        raise ValueError(f"unknown model '{which}'")
    return GrossCodeDecoder(in_channels=in_channels, hidden=hidden,
                            layers=layers, num_logical=num_logical)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_disk(prefix):
    x = np.load(f"{prefix}syndromes.npy")
    y = np.load(f"{prefix}observables.npy")
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def build_and_materialize(which, size, in_channels, num_logical, sample_x, device):
    cfg = PRESETS[size]
    model = get_model(which, in_channels, cfg["hidden"], cfg["layers"], num_logical).to(device)
    with torch.no_grad():                       # materialize LazyLinear head
        model(sample_x[:2].to(device))
    return model, cfg


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """Return (val_loss, per_observable_error, logical_error_rate)."""
    model.eval()
    loss_sum = n = n_obs_wrong = n_obs = n_fail = n_shots = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        loss_sum += criterion(logits, by).item() * bx.size(0)
        pred = (torch.sigmoid(logits) > 0.5).float()
        wrong = (pred != by)
        n_obs_wrong += wrong.sum().item(); n_obs += wrong.numel()
        n_fail += wrong.any(dim=1).sum().item(); n_shots += wrong.shape[0]
        n += bx.size(0)
    return loss_sum / n, n_obs_wrong / n_obs, n_fail / n_shots


def baseline(y):
    return float(y.mean()), float((y.sum(1) > 0).float().mean())


# ---------------------------------------------------------------------------
# Evaluate a saved model on a dataset (for OOD testing, conv or topo)
# ---------------------------------------------------------------------------
def eval_only(which, size, weights, data_prefix):
    device = pick_device()
    x, y = load_disk(data_prefix)
    in_channels, num_logical = x.shape[1], y.shape[1]
    model, cfg = build_and_materialize(which, size, in_channels, num_logical, x, device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    loader = DataLoader(TensorDataset(x, y), batch_size=512, shuffle=False)
    crit = nn.BCEWithLogitsLoss()
    b_obs, b_ler = baseline(y)
    vloss, per_obs, ler = evaluate(model, loader, device, crit)
    print("=" * 60)
    print(f" EVAL-ONLY  model={which} size={size} weights={weights}")
    print(f" data={data_prefix or '<default>'}syndromes.npy  N={len(x)} channels={in_channels}")
    print("=" * 60)
    print(f" baseline (predict zeros): per-obs {b_obs:.4f} | LER {b_ler:.4f}")
    print(f" model                   : per-obs {per_obs:.4f} | LER {ler:.4f} | val BCE {vloss:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_optimizer(model, optimizer, lr):
    if optimizer == "muon-lion":
        from optim import MuonLion
        return MuonLion(model)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def log_header(**kw):
    print("=" * 72)
    print(" Gross-code decoder training")
    for k, v in kw.items():
        print(f"   {k:14s}: {v}")
    print("=" * 72)


def train(which, size, epochs, batch_size, lr, val_frac, optimizer, use_ema,
          data_prefix, on_the_fly, p, rounds, steps, val_size, log_csv, out):
    device = pick_device()
    crit = nn.BCEWithLogitsLoss()
    csv = open(log_csv, "w") if log_csv else None
    if csv:
        csv.write("step,wall_s,train_bce,val_bce,per_obs,ler,best_ler\n")

    # ---- assemble data ----
    if on_the_fly:
        if data_prefix:                            # fixed val set from disk
            vx, vy = load_disk(data_prefix)
        else:                                      # generate a fixed val set once
            from data_gen import gen_phenomenological
            sx, sy = gen_phenomenological(p, rounds, val_size, seed=999)
            vx, vy = torch.from_numpy(sx).float(), torch.from_numpy(sy).float()
        in_channels, num_logical = vx.shape[1], vy.shape[1]
        val_loader = DataLoader(TensorDataset(vx, vy), batch_size=512, shuffle=False)
        sample_x = vx
        data_desc = f"on-the-fly phenomenological (p={p}, rounds={rounds}), val={len(vx)}"
    else:
        x, y = load_disk(data_prefix)
        in_channels, num_logical = x.shape[1], y.shape[1]
        n = x.shape[0]; n_val = int(n * val_frac)
        perm = torch.randperm(n); vi, ti = perm[:n_val], perm[n_val:]
        train_loader = DataLoader(TensorDataset(x[ti], y[ti]), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x[vi], y[vi]), batch_size=512, shuffle=False)
        vy = y[vi]; sample_x = x
        data_desc = f"disk '{data_prefix or ''}' N={n} (train {len(ti)} / val {len(vi)})"

    model, cfg = build_and_materialize(which, size, in_channels, num_logical, sample_x, device)
    opt = make_optimizer(model, optimizer, lr)
    ema = None
    if use_ema:
        from optim import EMA
        ema = EMA(model, decay=0.9998)

    b_obs, b_ler = baseline(vy)
    n_params = sum(pp.numel() for pp in model.parameters())
    log_header(model=which, size=f"{size} (H={cfg['hidden']} L={cfg['layers']}, {n_params:,} params)",
               in_channels=in_channels, device=str(device), optimizer=optimizer, ema=use_ema,
               data=data_desc, baseline=f"per-obs {b_obs:.4f} | LER {b_ler:.4f}")

    best_ler = 1.0
    t0 = time.perf_counter()

    def checkpoint(tag, train_bce, seen):
        nonlocal best_ler
        eval_model = model
        if ema is not None:
            eval_model = copy.deepcopy(model); ema.copy_to(eval_model)
        vloss, per_obs, ler = evaluate(eval_model, val_loader, device, crit)
        elapsed = time.perf_counter() - t0
        star = ""
        if ler < best_ler:
            best_ler = ler; star = " *best"
            torch.save(eval_model.state_dict(), f"{out}_best.pth")
        rate = seen / elapsed if elapsed > 0 else 0
        print(f"{tag} | {elapsed:6.1f}s {rate/1000:5.1f}k ex/s | train {train_bce:.4f} "
              f"| val {vloss:.4f} per-obs {per_obs:.4f} LER {ler:.4f} | best {best_ler:.4f}{star}")
        if csv:
            csv.write(f"{seen},{elapsed:.2f},{train_bce:.5f},{vloss:.5f},"
                      f"{per_obs:.5f},{ler:.5f},{best_ler:.5f}\n"); csv.flush()

    seen = 0
    if on_the_fly:
        from data_gen import sample_phenomenological_batch
        rng = np.random.default_rng(0)
        log_every = max(1, steps // 50)
        run = 0.0; run_n = 0
        for step in range(1, steps + 1):
            bx_np, by_np = sample_phenomenological_batch(p, rounds, batch_size, rng)
            bx = torch.from_numpy(bx_np).to(device); by = torch.from_numpy(by_np).to(device)
            model.train()
            loss = crit(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()
            if ema is not None: ema.update(model)
            seen += batch_size; run += loss.item() * batch_size; run_n += batch_size
            if step % log_every == 0:
                checkpoint(f"step {step:6d}/{steps}", run / run_n, seen)
                run = 0.0; run_n = 0
    else:
        for epoch in range(1, epochs + 1):
            model.train(); run = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                loss = crit(model(bx), by)
                opt.zero_grad(); loss.backward(); opt.step()
                if ema is not None: ema.update(model)
                run += loss.item() * bx.size(0); seen += bx.size(0)
            checkpoint(f"ep {epoch:3d}/{epochs}", run / len(train_loader.dataset), seen)

    final = f"{out}_final.pth"
    final_model = model
    if ema is not None:
        final_model = copy.deepcopy(model); ema.copy_to(final_model)
    torch.save(final_model.state_dict(), final)
    if csv: csv.close()
    print(f"\nDone. best LER {best_ler:.4f}. weights: {out}_best.pth (best), {final} (final)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["conv", "topo"], default="conv")
    ap.add_argument("--size", choices=list(PRESETS), default="small")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--optimizer", choices=["adamw", "muon-lion"], default="adamw")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--data", default="", help="data filename prefix to load")
    ap.add_argument("--out", default=None, help="output weights prefix (default: gross_<model>_<size>)")
    ap.add_argument("--log-csv", default=None, help="optional CSV metrics log path")
    # on-the-fly streaming (phenomenological)
    ap.add_argument("--on-the-fly", action="store_true", help="stream fresh phenomenological data")
    ap.add_argument("-p", type=float, default=0.005)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--val-size", type=int, default=20000)
    # eval-only (OOD testing)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--weights", default=None, help="weights file to evaluate (eval-only)")
    args = ap.parse_args()

    out = args.out or f"gross_{args.model}_{args.size}"
    if args.eval_only:
        assert args.weights, "--eval-only requires --weights"
        eval_only(args.model, args.size, args.weights, args.data)
    else:
        train(args.model, args.size, args.epochs, args.batch_size, args.lr, args.val_frac,
              args.optimizer, args.ema, args.data, args.on_the_fly, args.p, args.rounds,
              args.steps, args.val_size, args.log_csv, out)
