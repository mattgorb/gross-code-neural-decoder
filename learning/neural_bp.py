"""
Learnable ("neural") Belief-Propagation decoder — annotated skeleton.

WHAT THIS IS: the ordinary BP loop, but the message-scaling weights are
nn.Parameters trained by gradient descent on simulated (syndrome -> error)
data. This is the lane-3 STARTING POINT/baseline, not a finished paper.

CAVEATS:
  * The code here is a GENERIC small LDPC code, NOT the real surface/gross
    code — just enough graph structure to demonstrate BP.
  * It is SINGLE-SHOT: one syndrome -> one decode. There is no time/rounds
    dimension here (a real QEC decoder would stack many measurement rounds).
"""
import torch, numpy as np

# ---------------------------------------------------------------------------
# Build the code (its parity-check matrix H).
#   n      = number of QUBITS         (bits being protected) -> COLUMNS of H
#   m      = number of CHECKS/DETECTORS (parity checks)      -> ROWS of H
#   col_w  = checks-per-qubit (graph degree). This is the "Low-Density" in
#            LDPC: each qubit is watched by only a few checks (sparse graph).
#            NOTE: col_w is NOT "rounds" — there are no rounds in this toy.
# H[a, i] == 1  means "detector a watches qubit i"  (one edge of the graph).
# ---------------------------------------------------------------------------
def make_ldpc(n=16, m=12, col_w=3, seed=1):
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), int)
    for j in range(n):                       # for each qubit (column)...
        H[rng.choice(m, col_w, replace=False), j] = 1   # wire it to col_w checks
    return H

H_np = make_ldpc(); m, n = H_np.shape        # m detectors, n qubits

# The Tanner graph, in three equivalent forms:
checks = [np.where(H_np[a])[0] for a in range(m)]   # checks[a]  = qubits detector a watches
varsof = [np.where(H_np[:, i])[0] for i in range(n)] # varsof[i] = detectors watching qubit i
edges  = [(a, i) for a in range(m) for i in checks[a]]  # every (detector, qubit) edge
eidx   = {e: k for k, e in enumerate(edges)}; E = len(edges)   # index each edge




class NeuralBP(torch.nn.Module):
    """BP message-passing with LEARNABLE weights. Plain BP = all weights 1.0."""
    def __init__(self, T=4):                 # T = number of BP iterations (unrolled "layers")
        super().__init__()
        self.T  = T
        # THE LEARNED PARAMETERS (this is the entire "neural" part):
        self.w  = torch.nn.Parameter(torch.ones(E))        # one weight per graph edge
        self.wp = torch.nn.Parameter(torch.tensor(1.0))    # one weight on the prior


    def forward(self, syndrome, p=0.07):
        # prior belief: every qubit's base log-odds of being error-free.
        # p = assumed physical error rate; high prior = "errors are rare".
        prior = torch.log(torch.tensor((1 - p) / p)) * self.wp
        M_vc = prior * torch.ones(E)         # qubit->detector messages (init at prior)
        M_cv = torch.zeros(E)                # detector->qubit messages

        for _ in range(self.T):              # each iteration = one round of "gossip"
            # ---- detector -> qubit: enforce each parity constraint ----
            new = torch.zeros(E)
            for a in range(m):
                eks = [eidx[(a, i)] for i in checks[a]]
                t = torch.tanh(M_vc[torch.tensor(eks)] / 2.0)
                sgn = (1 - 2 * syndrome[a])  # flips the message if this detector fired (s=1)
                for kk, ek in enumerate(eks):
                    others = torch.cat([t[:kk], t[kk+1:]])           # exclude self (no echo)
                    prod = torch.clamp(torch.prod(others), -0.999, 0.999)
                    new[ek] = self.w[ek] * sgn * 2.0 * torch.atanh(prod)  # <-- learned weight w
            M_cv = new
            # ---- qubit -> detector: combine prior + all OTHER detectors ----
            new = torch.zeros(E)
            for i in range(n):
                eks = [eidx[(a, i)] for a in varsof[i]]
                tot = prior + M_cv[torch.tensor(eks)].sum()
                for ek in eks:
                    new[ek] = tot - M_cv[ek]                          # exclude self again
            M_vc = new

        # ---- final belief per qubit -> decision ----
        L = torch.stack([prior + M_cv[torch.tensor([eidx[(a, i)] for a in varsof[i]])].sum()
                         for i in range(n)])
        return L                              # L < 0  =>  "this qubit flipped"

# ---------------------------------------------------------------------------
# Simulate training/eval data: flip each qubit w.p. p, compute the syndrome
# the detectors would report. (Real version: generate this with Stim.)
# ---------------------------------------------------------------------------
def sample(p=0.07, seed=None):
    rng = np.random.default_rng(seed)
    e = (rng.random(n) < p).astype(np.float32)   # true error pattern (the LABEL)
    s = (H_np @ e) % 2                            # syndrome = detector bits (the INPUT)
    return torch.tensor(s, dtype=torch.float32), torch.tensor(e)

# ---- TRAIN: gradient descent finds the weights that decode best ----
model = NeuralBP(T=4)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
lossf = torch.nn.BCEWithLogitsLoss()             # predict flip/no-flip per qubit
for step in range(250):
    s, e = sample(seed=step)
    loss = lossf(-model(s), e)      
    print(s)
    print(e)   
    sys.exit()          # -L is the per-qubit flip logit
    opt.zero_grad(); loss.backward(); opt.step()  # update the learned weights

# ---- EVAL: learned weights vs the plain-BP default (all weights = 1) ----
def block_success(weights_one, N=300):
    succ = 0
    with torch.no_grad():
        ws, wps = model.w.data.clone(), model.wp.data.clone()
        if weights_one:                           # temporarily reset to plain BP
            model.w.data = torch.ones(E); model.wp.data = torch.tensor(1.0)
        for k in range(N):
            s, e = sample(seed=10000 + k)
            succ += int(torch.equal((model(s) < 0).float(), e))  # whole pattern correct?
        model.w.data, model.wp.data = ws, wps
    return succ / N

print("plain BP   (weights = 1):", block_success(True))
print("neural BP  (learned w)  :", block_success(False))
print("sample learned weights  :", model.w.data[:6].numpy().round(2))