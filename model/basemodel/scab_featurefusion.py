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
            tau=2.0, detach_reset=True, backend="torch",v_threshold=1.0
        )
        self.v_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend="torch",v_threshold=1.0
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
class BiASpatialAttention2D(nn.Module):
    """
    空间注意力适配器，包含短残差。

    输入、输出：[B, dim, H, W]
    使用时固定 input_resolution，与实际特征尺寸一致。
    """

    def __init__(
        self,
        dim,
        input_resolution=(11, 11),
        num_heads=4,
        attention_groups=2,
        stride=2,
        dynamic_factor=2,
        postype="rel",
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        # 仅在使用 BiA 时加载原模块。
        from model.CBiAFormer.models.CBiAFormer import BiA_Attention

        h, w = input_resolution

        if h != w or h < 2:
            raise ValueError("当前适配器要求尺寸不小于 2 的方形特征图")

        if num_heads <= 0 or attention_groups <= 0:
            raise ValueError("num_heads 和 attention_groups 必须大于 0")

        # 原 BiA 局部分支使用 dim/4 通道，再进行多头划分。
        if dim <= 0 or dim % (4 * num_heads) != 0:
            raise ValueError("dim 必须是 4 * num_heads 的正整数倍")

        if num_heads % attention_groups != 0:
            raise ValueError("num_heads 必须能被 attention_groups 整除")

        if stride < 1 or postype not in ("rel", "conv"):
            raise ValueError("stride 必须为正，postype 必须是 rel 或 conv")

        self.dim = dim
        self.input_resolution = (h, w)

        self.attn = BiA_Attention(
            dim=dim,
            input_resolution=(h, w),
            num_heads=num_heads,
            stride=stride,
            dynamic_factor=dynamic_factor,
            # 整个 H×W patch 作为一个查询窗口。
            window_size=h,
            Attention_Group=attention_groups,
            postype=postype,
            # 启用局部细粒度分支，每个位置采样 9 个邻域点。
            k=3,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # 原实现的固定步幅卷积在 11×11 下输出 2×2 偏移图，
        # 但当前查询窗口网格是 1×1，需要将两者对齐。
        # 这里只池化偏移预测图，输入特征保持原尺寸。
        self.attn.offset.add_module(
            "window_pool",
            nn.AdaptiveAvgPool2d(self.attn.window_num),
        )

    def forward(self, x):
        if (
            x.ndim != 4
            or x.shape[1] != self.dim
            or tuple(x.shape[-2:]) != self.input_resolution
        ):
            raise ValueError(
                f"Expected [B,{self.dim},"
                f"{self.input_resolution[0]},"
                f"{self.input_resolution[1]}], "
                f"got {tuple(x.shape)}"
            )

        # 原 BiA 返回更新特征与采样坐标。
        update, _ = self.attn(x)

        # 与现有 SpatialAttention 保持一致：返回含短残差的特征。
        return x + update