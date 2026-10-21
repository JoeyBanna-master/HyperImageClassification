"""HSI center-pixel classification baseline.

Input:  [B, C, P, P]  (bands-first HSI patches, odd P)
Output: [B, K]        (class logits, not probabilities)

Three swappable modules:
    SpectralMamba -> SpatialTransformer -> GatedFusion
The spectral and spatial branches run in parallel; the arrow above
only lists modules, not execution order.
"""

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class SpectralMamba(nn.Module):
    """Read the center pixel as a sequence of C scalar band values."""

    def __init__(self, num_bands: int, dim: int = 64):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "Missing mamba_ssm. Install a GPU-compatible mamba-ssm "
                "or activate your existing S2Mamba environment."
            )
        self.num_bands = num_bands
        self.embedding = nn.Linear(1, dim)
        self.band_pos = nn.Parameter(torch.zeros(1, num_bands, dim))
        nn.init.trunc_normal_(self.band_pos, std=0.02)
        self.pre_norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, center_spectrum: torch.Tensor) -> torch.Tensor:
        # center_spectrum: [B, C]
        if center_spectrum.ndim != 2 or center_spectrum.size(1) != self.num_bands:
            raise ValueError(f"Expected [B, {self.num_bands}] center spectra")
        tokens = center_spectrum.unsqueeze(-1)       # [B, C, 1]
        tokens = self.embedding(tokens) + self.band_pos  # [B, C, D]
        tokens = tokens + self.mamba(self.pre_norm(tokens))
        return self.out_norm(tokens.mean(dim=1))      # [B, D]


class SpatialTransformer(nn.Module):
    """Center-query attention over every pixel in a P x P HSI patch."""

    def __init__(self, num_bands: int, patch_size: int = 9,
                 dim: int = 64, num_heads: int = 4):
        super().__init__()
        if patch_size % 2 == 0 or patch_size <= 0:
            raise ValueError("patch_size must be a positive odd integer")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.proj = nn.Linear(num_bands, dim)
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, patch_size * patch_size, dim)
        )
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, patch: torch.Tensor, return_attention: bool = False):
        # patch: [B, C, P, P]
        b, c, h, w = patch.shape
        p = self.patch_size
        if (c, h, w) != (self.num_bands, p, p):
            raise ValueError(f"Expected [B, {self.num_bands}, {p}, {p}]")
        pixels = patch.flatten(2).transpose(1, 2)   # [B, P*P, C]
        tokens = self.input_norm(self.proj(pixels) + self.spatial_pos)
        center_index = (p * p) // 2
        query = tokens[:, center_index:center_index + 1, :]  # [B, 1, D]
        # The center pixel is included in keys/values.
        context, attn_weights = self.attention(
            query, tokens, tokens, need_weights=return_attention
        )  # context: [B, 1, D]
        out = self.norm1(query + context)
        out = self.norm2(out + self.mlp(out))
        feature = out.squeeze(1)                   # [B, D]
        return (feature, attn_weights) if return_attention else feature


class GatedFusion(nn.Module):
    """Dimension-wise adaptive weighting of spectral and spatial features."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, spectral: torch.Tensor, spatial: torch.Tensor):
        g = self.gate(torch.cat([spectral, spatial], dim=-1))  # [B, D]
        fused = g * spectral + (1.0 - g) * spatial
        return fused, g


class HSIHybridBaseline(nn.Module):
    """Independent spectral/spatial extraction followed by late gated fusion."""

    def __init__(self, num_bands: int, num_classes: int, patch_size: int = 9,
                 dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.spectral = SpectralMamba(num_bands=num_bands, dim=dim)
        self.spatial = SpatialTransformer(
            num_bands=num_bands, patch_size=patch_size,
            dim=dim, num_heads=num_heads
        )
        self.fusion = GatedFusion(dim=dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        if x.ndim != 4:
            raise ValueError("Input must be [B, C, P, P], not [B, P, P, C]")
        b, c, h, w = x.shape
        if (c, h, w) != (self.num_bands, self.patch_size, self.patch_size):
            raise ValueError(
                f"Expected [B, {self.num_bands}, {self.patch_size}, {self.patch_size}], "
                f"got {tuple(x.shape)}"
            )
        center_spectrum = x[:, :, h // 2, w // 2]       # [B, C]
        spectral_feat = self.spectral(center_spectrum)  # [B, D]
        if return_details:
            spatial_feat, attention = self.spatial(x, return_attention=True)
        else:
            spatial_feat = self.spatial(x)
            attention = None
        fused, gate = self.fusion(spectral_feat, spatial_feat)
        logits = self.classifier(fused)                  # [B, K]
        if return_details:
            return logits, {
                "spectral": spectral_feat,
                "spatial": spatial_feat,
                "fused": fused,
                "gate": gate,
                "spatial_attention": attention,        # [B, 1, P*P]
            }
        return logits