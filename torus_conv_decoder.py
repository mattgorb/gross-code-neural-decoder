"""
Gross-code [[144,12,12]] neural decoder (paper-style), torus-convolution variant.

Architecture (following "Scalable Neural Decoders for Practical Fault-Tolerant
Quantum Computation"):
  - Binary detection events embedded into H-dim features at each (round, check) site.
  - L bottleneck residual blocks: BatchNorm -> SiLU -> (H -> H/4) -> torus conv
    -> (H/4 -> H) -> + residual.
  - Convolution = generalized convolution on the Gross-code torus, with weights
    indexed by the monomial offsets (A = x^3+y+y^2, B = y^3+x+x^2) plus a temporal
    neighbour, so features mix across both space (the check torus) and rounds.
  - Prediction head -> one logit per logical observable (k = 12).

Defaults here (H=64, L=8) are sized to train on a laptop; the paper uses H=256, L=14.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Gross-code monomial offsets on the L x M check torus (x on L-axis, y on M-axis).
SHIFTS_A = [(3, 0), (0, 1), (0, 2)]  # x^3, y, y^2
SHIFTS_B = [(0, 3), (1, 0), (2, 0)]  # y^3, x, x^2


class TorusBivariateConv(nn.Module):
    """Generalized convolution on the Gross-code spacetime torus.

    Input/Output: [B, C, R, X, Y] where (X, Y) is the check torus and R is rounds.
    Each output site aggregates: itself, the 3 A-offset neighbours, the 3 B-offset
    neighbours (all spatial, periodic), and the two temporal neighbours (+/-1 round).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_taps = 1 + len(SHIFTS_A) + len(SHIFTS_B) + 2  # = 9
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.num_taps)
            / (in_channels * self.num_taps) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        taps = [x]  # self
        for sx, sy in SHIFTS_A:
            taps.append(torch.roll(x, shifts=(sx, sy), dims=(-2, -1)))
        for sx, sy in SHIFTS_B:
            taps.append(torch.roll(x, shifts=(sx, sy), dims=(-2, -1)))
        taps.append(torch.roll(x, shifts=1, dims=-3))   # previous round
        taps.append(torch.roll(x, shifts=-1, dims=-3))  # next round

        stacked = torch.stack(taps, dim=-1)  # [B, C, R, X, Y, taps]
        out = torch.einsum("bcrxyn,ocn->borxy", stacked, self.weight)
        return out + self.bias.view(1, -1, 1, 1, 1)


class BottleneckBlock(nn.Module):
    """Bottleneck residual block: BN -> SiLU -> 1x1 down -> torus conv -> SiLU -> 1x1 up."""

    def __init__(self, hidden):
        super().__init__()
        bottleneck = max(hidden // 4, 1)
        self.bn = nn.BatchNorm3d(hidden)
        self.proj_in = nn.Conv3d(hidden, bottleneck, kernel_size=1)
        self.conv = TorusBivariateConv(bottleneck, bottleneck)
        self.proj_out = nn.Conv3d(bottleneck, hidden, kernel_size=1)

    def forward(self, x):
        h = F.silu(self.bn(x))
        h = self.proj_in(h)
        h = F.silu(self.conv(h))
        h = self.proj_out(h)
        return x + h


class GrossCodeDecoder(nn.Module):
    def __init__(self, in_channels=1, hidden=64, layers=8, num_logical=12):
        super().__init__()
        self.embed = nn.Conv3d(in_channels, hidden, kernel_size=1)  # embed detections
        self.blocks = nn.ModuleList(BottleneckBlock(hidden) for _ in range(layers))
        self.head_bn = nn.BatchNorm3d(hidden)
        # Pool over rounds only, keep the check torus, then read out per logical qubit.
        # LazyLinear infers the flattened (hidden * L * M) size on the first forward.
        self.head = nn.LazyLinear(num_logical)

    def forward(self, syndrome):
        # syndrome: [B, 1, R, X, Y]
        h = self.embed(syndrome)
        for block in self.blocks:
            h = block(h)
        h = F.silu(self.head_bn(h))
        h = h.mean(dim=2)           # average over rounds -> [B, H, X, Y]
        h = h.flatten(1)            # keep spatial layout -> [B, H*X*Y]
        return self.head(h)         # [B, num_logical]


if __name__ == "__main__":
    for C in (1, 2):
        model = GrossCodeDecoder(in_channels=C, hidden=64, layers=8)
        x = torch.randint(0, 2, (4, C, 13, 12, 6)).float()
        logits = model(x)
        n = sum(p.numel() for p in model.parameters())
        print(f"C={C}: input {list(x.shape)} -> logits {list(logits.shape)} | params {n:,}")
