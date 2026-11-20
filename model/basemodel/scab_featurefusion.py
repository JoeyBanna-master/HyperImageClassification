import torch
import torch.nn as nn


import math
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import functional



class SpikingSpatialAttention(nn.Module):
    def __init__(self, dim, T=4):
        super().__init__()
        self.T = T

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)

        self.q_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend="torch"
        )
        self.k_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend="torch",v_threshold=0.75
        )
        self.v_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend="torch",v_threshold=0.75
        )
        self.gate_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5,
            detach_reset=True, backend="torch"
        )

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # 不同 batch 独立，清除上次调用的膜电位。
        functional.reset_net(self)

        # x: [B, N, D]
        # 将静态特征重复到 T 个脉冲时间步。
        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_norm(self.v_proj(x))

        q = self.q_lif(q.unsqueeze(0).repeat(self.T, 1, 1, 1))
        k = self.k_lif(k.unsqueeze(0).repeat(self.T, 1, 1, 1))
        v = self.v_lif(v.unsqueeze(0).repeat(self.T, 1, 1, 1))
        # q/k/v: [T, B, N, D]

        # K、V 的脉冲重合，随后沿空间位置聚合。
        context = (k * v).sum(dim=2, keepdim=True)
        # context: [T, B, 1, D]

        gate = self.gate_lif(context)

        # 所有位置共享全局上下文，由各位置的 Q 决定输出。
        update = q * gate                     # [T, B, N, D]
        update = update.mean(dim=0)           # [B, N, D]
        update = self.out_proj(update)

        return update, None
# 光谱空间都使用 spiking 注意力
class SpatialAttention(nn.Module):
    def __init__(self, dim, T=4):
        super().__init__()

        # 全部 dim 个通道一起计算注意力 basic 注意力
        # self.attn = BasicAttention(dim)
        self.attn = SpikingSpatialAttention(dim,T=T)

    def forward(self, x):
        B, C, H, W = x.shape

        # 每个空间位置作为一个 token
        tokens = x.flatten(2).transpose(1, 2)  # [B,H*W,C]

        update, _ = self.attn(tokens)
        tokens = tokens + update             # 保留短残差

        return tokens.transpose(1, 2).reshape(B, C, H, W)
class PairwiseCrossFusion(nn.Module):
    """两路 [B,dim,H,W] → 一路 [B,dim,H,W]。"""

    def __init__(
        self,
        dim=128,
        spectral_attention_factory=SpatialAttention,
        spatial_attention_factory=SpatialAttention,
    ):
        super().__init__()
        self.dim = dim

        # 工厂每次创建新实例，避免不同分支意外共享参数。
        self.spectral_attention = spectral_attention_factory(dim)
        self.spatial_attention = spatial_attention_factory(dim)

        self.cross_spectral = nn.Conv2d(2 * dim, dim, 1)
        self.cross_spatial = nn.Conv2d(2 * dim, dim, 1)

        self.fusion = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

    def forward(self, fs, fp):
        if fs.ndim != 4 or fs.shape[1] != self.dim or fp.shape != fs.shape:
            raise ValueError("Expected two matching [B,dim,H,W] tensors")

        fs_attn = self.spectral_attention(fs)
        fp_attn = self.spatial_attention(fp)

        rs = fs + self.cross_spectral(
            torch.cat([fs_attn, fp], dim=1)
        )
        rp = fp + self.cross_spatial(
            torch.cat([fp_attn, fs], dim=1)
        )

        return self.fusion(torch.cat([rs, rp], dim=1))
# class PairwiseCrossFusion(nn.Module):
#     """用双向门控融合两张 [B, dim, H, W] 特征图。"""

#     def __init__(
#         self,
#         dim=128,
#         spectral_attention_factory=SpatialAttention,
#         spatial_attention_factory=SpatialAttention,
#     ):
#         super().__init__()
#         self.dim = dim

#         self.spectral_attention = spectral_attention_factory(dim)
#         self.spatial_attention = spatial_attention_factory(dim)

#         # 根据本路的注意力特征与对侧的原始特征计算更新量。
#         self.cross_spectral = nn.Conv2d(2 * dim, dim, kernel_size=1)
#         self.cross_spatial = nn.Conv2d(2 * dim, dim, kernel_size=1)

#         # 用对侧的注意力特征生成门控，控制更新量。
#         self.gate_from_spatial = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1),
#             nn.Sigmoid(),
#         )
#         self.gate_from_spectral = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1),
#             nn.Sigmoid(),
#         )

#         self.fusion = nn.Sequential(
#             nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(),
#         )

#     def forward(self, fs, fp):
#         if fs.ndim != 4 or fs.shape[1] != self.dim or fp.shape != fs.shape:
#             raise ValueError(
#                 f"Expected two matching [B, {self.dim}, H, W] tensors, "
#                 f"got {tuple(fs.shape)} and {tuple(fp.shape)}"
#             )

#         fs_attn = self.spectral_attention(fs)
#         fp_attn = self.spatial_attention(fp)

#         delta_s = self.cross_spectral(torch.cat([fs_attn, fp], dim=1))
#         delta_p = self.cross_spatial(torch.cat([fp_attn, fs], dim=1))

#         # 光谱路加回原始光谱，空间路加回原始空间。
#         rs = fs + self.gate_from_spatial(fp_attn) * delta_s
#         rp = fp + self.gate_from_spectral(fs_attn) * delta_p

#         return self.fusion(torch.cat([rs, rp], dim=1))
class SoftmaxAttention2D(nn.Module):
    """普通注意力示例"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        update, _ = self.attn(
            tokens, tokens, tokens, need_weights=False
        )
        return (tokens + update).transpose(1, 2).reshape(b, c, h, w)
class MHLA2D(nn.Module):
    """Map [B, dim, H, W] to the same shape, including one residual.

    For input_resolution=(11, 11) and piece_size=4, nine spatial blocks
    contain sixteen tokens each after padding. Padding is applied after
    Q/K activation and V projection, so padded keys and values stay zero.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        input_resolution=(11, 11),
        piece_size=4,
        dropout=0.0,
    ):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by num_heads")
        height, width = input_resolution
        if min(height, width, piece_size) < 1:
            raise ValueError("input dimensions and piece_size must be positive")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.input_resolution = (height, width)
        self.piece_size = piece_size
        self.block_rows = (height + piece_size - 1) // piece_size
        self.block_cols = (width + piece_size - 1) // piece_size
        self.num_blocks = self.block_rows * self.block_cols
        self.pad_height = self.block_rows * piece_size - height
        self.pad_width = self.block_cols * piece_size - width
        self.eps = 1e-6

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.RMSNorm(dim)
        self.k_norm = nn.RMSNorm(dim)
        self.lepe = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        rows, cols = torch.meshgrid(
            torch.arange(self.block_rows, dtype=torch.float32),
            torch.arange(self.block_cols, dtype=torch.float32),
            indexing="ij",
        )
        centers = torch.stack((rows, cols), dim=-1).reshape(-1, 2)
        distances = torch.cdist(centers, centers)
        initial_weights = 1 - distances / distances.max().clamp_min(1)
        initial_weights = initial_weights.clamp_min(1e-4)
        initial_weights = initial_weights / initial_weights.sum(dim=-1, keepdim=True)
        self.mix_logits = nn.Parameter(initial_weights.log())

    def _split_heads(self, features):
        """Convert valid [B,H,W,C] features to [B,heads,blocks,tokens,d]."""
        batch_size = features.shape[0]
        features = F.pad(
            features, (0, 0, 0, self.pad_width, 0, self.pad_height)
        )
        blocks = features.reshape(
            batch_size,
            self.block_rows,
            self.piece_size,
            self.block_cols,
            self.piece_size,
            self.num_heads,
            self.head_dim,
        )
        return blocks.permute(0, 5, 1, 3, 2, 4, 6).reshape(
            batch_size,
            self.num_heads,
            self.num_blocks,
            self.piece_size ** 2,
            self.head_dim,
        )

    def _restore(self, blocks):
        """Restore valid [B,H,W,C] features from the block/head layout."""
        batch_size = blocks.shape[0]
        features = blocks.reshape(
            batch_size,
            self.num_heads,
            self.block_rows,
            self.block_cols,
            self.piece_size,
            self.piece_size,
            self.head_dim,
        )
        features = features.permute(0, 2, 4, 3, 5, 1, 6).reshape(
            batch_size,
            self.block_rows * self.piece_size,
            self.block_cols * self.piece_size,
            self.dim,
        )
        height, width = self.input_resolution
        return features[:, :height, :width, :]

    def forward(self, x):
        if (
            x.ndim != 4
            or x.shape[1] != self.dim
            or tuple(x.shape[-2:]) != self.input_resolution
        ):
            raise ValueError(
                f"Expected [B,{self.dim},{self.input_resolution[0]},"
                f"{self.input_resolution[1]}], got {tuple(x.shape)}"
            )

        tokens = self.norm(x.permute(0, 2, 3, 1))
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        q = F.relu(self.q_norm(q)) + self.eps
        k = F.relu(self.k_norm(k)) + self.eps

        # Compute LePE on valid V. The convolution uses zero boundary padding.
        local_position = self.lepe(v.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        q, k, v = (self._split_heads(features) for features in (q, k, v))
        output_dtype = v.dtype
        accumulation_dtype = torch.float32 if v.dtype in (
            torch.float16, torch.bfloat16
        ) else v.dtype

        # Keep reductions and normalization in float32 under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = (features.to(accumulation_dtype) for features in (q, k, v))
            mixing = self.mix_logits.to(accumulation_dtype).softmax(dim=-1)
            kv = k.transpose(-2, -1) @ v
            k_sum = k.sum(dim=-2).unsqueeze(-1)
            mixed_kv = torch.einsum("ij,bhjde->bhide", mixing, kv)
            mixed_k_sum = torch.einsum("ij,bhjdv->bhidv", mixing, k_sum)
            numerator = q @ mixed_kv
            denominator = (q @ mixed_k_sum).clamp_min(self.eps)
            update = numerator / denominator

        update = self._restore(update.to(output_dtype)) + local_position
        update = self.proj_drop(self.proj(update)).permute(0, 3, 1, 2)
        return x + update