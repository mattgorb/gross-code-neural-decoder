"""
Gross-code [[144,12,12]] neural decoder -- hybrid variant.

Same torus-convolution backbone as torus_conv_decoder.GrossCodeDecoder, plus a second
"structural" branch that summarizes the detector pattern at multiple spatial scales and
fuses it in before the readout. The branches are combined and mapped to 12 logits.

This replaces the original gudhi/witness-complex (persistent-homology) branch with a
fully differentiable, dependency-free multi-scale density summary so the model trains
locally with no extra packages. The two-branch "coordinate conv + global structure"
design is preserved; swap the structural branch for a true persistence computation if
gudhi is available and you want exact Betti features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torus_conv_decoder import TorusBivariateConv, BottleneckBlock  # shared backbone


class StructuralBranch(nn.Module):
    """Multi-scale, differentiable summary of the detector spacetime volume.

    Adaptive-average-pools the raw detectors to a few fixed grid sizes (capturing
    density/clustering structure at different scales, a differentiable stand-in for
    Betti-0-style features) and maps them to a feature vector with a small MLP."""

    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.scales = [(4, 3, 3), (2, 2, 2), (1, 1, 1)]
        feat_in = in_channels * sum(t * h * w for t, h, w in self.scales)
        self.mlp = nn.Sequential(
            nn.Linear(feat_in, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):  # x: [B, C, R, H, W]
        feats = [F.adaptive_avg_pool3d(x, s).flatten(1) for s in self.scales]
        return self.mlp(torch.cat(feats, dim=1))


class GrossCodeDecoder(nn.Module):
    def __init__(self, in_channels=1, hidden=64, layers=8, num_logical=12, topo_dim=64):
        super().__init__()
        self.embed = nn.Conv3d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(BottleneckBlock(hidden) for _ in range(layers))
        self.head_bn = nn.BatchNorm3d(hidden)
        self.structural = StructuralBranch(in_channels, out_dim=topo_dim)
        self.head = nn.LazyLinear(num_logical)   # fuses conv + structural features

    def forward(self, syndrome):
        h = self.embed(syndrome)
        for block in self.blocks:
            h = block(h)
        h = F.silu(self.head_bn(h))
        h_conv = h.mean(dim=2).flatten(1)            # [B, hidden*H*W]
        h_struct = self.structural(syndrome)          # [B, topo_dim]
        return self.head(torch.cat([h_conv, h_struct], dim=1))


if __name__ == "__main__":
    for C in (1, 2):
        model = GrossCodeDecoder(in_channels=C, hidden=64, layers=8)
        x = torch.randint(0, 2, (4, C, 13, 12, 6)).float()
        logits = model(x)
        n = sum(p.numel() for p in model.parameters())
        print(f"C={C}: input {list(x.shape)} -> logits {list(logits.shape)} | params {n:,}")
