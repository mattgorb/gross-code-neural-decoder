"""
Optimizers for the paper-faithful training recipe: Muon (for matrix/conv weights) +
Lion (for scalars/biases/norms), plus an exponential moving average of weights.

These match the paper's setup (Muon peak ~3e-3 for conv weights, Lion ~2e-4 for scalars,
EMA decay 0.9998). They are optional -- AdamW works fine for the tiny local models.
"""
import torch


# ---------------------------------------------------------------------------
# Muon: momentum + Newton-Schulz orthogonalization of the (matricized) update.
# ---------------------------------------------------------------------------
def _newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for parameters with ndim >= 2 (weights are reshaped to 2D)."""

    def __init__(self, params, lr=3e-3, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(list(params), dict(lr=lr, momentum=momentum,
                                            nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mom, nest, ns = group["momentum"], group["nesterov"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nest else buf
                shape = g.shape
                u = _newton_schulz5(g.reshape(shape[0], -1), ns).reshape(shape)
                # scale so the update RMS is roughly lr regardless of matrix shape
                scale = max(1.0, shape[0] / (u.numel() / shape[0])) ** 0.5
                p.add_(u.to(p.dtype), alpha=-group["lr"] * scale)


class Lion(torch.optim.Optimizer):
    """Lion: sign-of-momentum updates. Good for scalars/biases/norm params here."""

    def __init__(self, params, lr=2e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(list(params), dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(p)
                ea = st["exp_avg"]
                update = ea.mul(b1).add(g, alpha=1 - b1).sign_()
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                ea.mul_(b2).add_(g, alpha=1 - b2)


class MuonLion:
    """Convenience wrapper: route ndim>=2 params to Muon, the rest to Lion.
    Exposes zero_grad()/step() like a single optimizer."""

    def __init__(self, model, muon_lr=3e-3, lion_lr=2e-4, weight_decay=3e-3):
        matrix, scalar = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (matrix if p.ndim >= 2 else scalar).append(p)
        self.muon = Muon(matrix, lr=muon_lr)
        self.lion = Lion(scalar, lr=lion_lr, weight_decay=weight_decay)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.lion.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.lion.step()


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model, step=None):
        # Warmup: early on, use a smaller effective decay so the EMA tracks the model
        # instead of staying stuck near the random initialization.
        d = self.decay
        if step is not None:
            d = min(d, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
