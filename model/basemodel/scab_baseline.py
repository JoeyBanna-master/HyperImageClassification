import torch
import torch.nn as nn


import math
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import functional
#卷积模块 负责提取  普通的混合空间–光谱特征
class ConvStem(nn.Module):
    # 
    def __init__(self,in_channels, embed_dim):
        super().__init__()

        # self.block = nn.Sequential(
        #     nn.Conv3d(1, channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
        #     nn.BatchNorm3d(channels),
        #     nn.ReLU(),
        # )

        self.conv = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
#池化和分类头 
class CNNBaseline(nn.Module):
    def __init__(self, num_bands=144, num_classes=15, dim=128):
        super().__init__()
        self.stem = ConvStem(num_bands, dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.stem(x)             # [B, 32, 11, 11]
        x = self.pool(x).flatten(1)  # [B, 32]
        logits = self.head(x)        # [B, 15]
        return logits
class DualBranchBaseline(nn.Module):
    def __init__(self, num_bands=144, num_classes=15, dim=32):
        super().__init__()

        # 共享特征提取：144 个波段 → 32 个特征通道
        self.stem = ConvStem(num_bands, dim)

        # 通道混合分支
        self.spectral = nn.Sequential(
                   nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                   nn.BatchNorm2d(dim),
                   nn.ReLU(),
               )
       
        # 空间分支：每个通道独立提取空间邻域特征
        self.spatial = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        # 两路拼接后：64 通道 → 32 通道
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        feature = self.stem(x)                    # [B,32,11,11]

        fs = self.spectral(feature)                # [B,32,11,11]
        fp = self.spatial(feature)                 # [B,32,11,11]

        fused = torch.cat([fs, fp], dim=1)         # [B,64,11,11]
        fused = self.fusion(fused)                 # [B,32,11,11]

        vector = self.pool(fused).flatten(1)       # [B,32]
        return self.head(vector)                  # [B,15]

class BasicAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k_proj  = k.transpose(1, 2)

        scores = torch.bmm(q, k_proj) / math.sqrt(q.size(-1))        
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        return out, attn
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
            tau=2.0, detach_reset=True, backend="torch"
        )
        self.v_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=True, backend="torch"
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

# 分组
class GroupSpatialAttention(nn.Module):
    def __init__(self, dim, num_groups=4):
        super().__init__()
        assert num_groups > 0 and dim % num_groups == 0

        self.num_groups = num_groups
        # 每组有多少个特征通道
        self.group_dim = dim // num_groups

        self.attn = nn.ModuleList([
            SpikingSpatialAttention(self.group_dim)
            for _ in range(num_groups)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.num_groups * self.group_dim

        groups = x.chunk(self.num_groups, dim=1)
        outputs = []

        for group, attention in zip(groups, self.attn):
            # 每次处理一组：[B, 8, 11, 11]
            tokens = group.flatten(2).transpose(1, 2)
            # tokens：[B, 121, 8]

            update, _ = attention(tokens)
            tokens = tokens + update  # 短残差

            feature = tokens.transpose(1, 2).reshape(
                B, self.group_dim, H, W
            )
            outputs.append(feature)

        return torch.cat(outputs, dim=1)  # [B, 32, 11, 11]
# 不分组
class SpatialAttention(nn.Module):
    def __init__(self, dim, T=4):
        super().__init__()

        # 全部 dim 个通道一起计算注意力 basic 注意力
        # self.attn = BasicAttention(dim)
        self.attn = SpikingSpatialAttention(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 每个空间位置作为一个 token
        tokens = x.flatten(2).transpose(1, 2)  # [B,H*W,C]

        update, _ = self.attn(tokens)
        tokens = tokens + update             # 保留短残差

        return tokens.transpose(1, 2).reshape(B, C, H, W)
class GroupedAttentionBaseline(DualBranchBaseline):
    def __init__(
        self,
        num_bands=144,
        num_classes=15,
        dim=32,
        num_groups=4,
    ):
        super().__init__(num_bands, num_classes, dim)
# 分组注意力
        # self.spectral_attention = GroupSpatialAttention(
        #     dim, num_groups=num_groups
        # )
        # self.spatial_attention = GroupSpatialAttention(
        #     dim, num_groups=num_groups
        # )
        # 不分组注意力
        self.spectral_attention = SpatialAttention(
                    dim
                )
        self.spatial_attention = SpatialAttention(
            dim,
        )

    def forward(self, x):
        feature = self.stem(x)                 # [B,32,11,11]

        fs = self.spectral(feature)             # [B,32,11,11]
        fp = self.spatial(feature)              # [B,32,11,11]

        fs = self.spectral_attention(fs)        # [B,32,11,11]
        fp = self.spatial_attention(fp)         # [B,32,11,11]

        fused = torch.cat([fs, fp], dim=1)      # [B,64,11,11]
        fused = self.fusion(fused)              # [B,32,11,11]

        vector = self.pool(fused).flatten(1)    # [B,32]
        return self.head(vector)               # [B,15]
class CrossFusionBaseline(GroupedAttentionBaseline):
    def __init__(
        self, num_bands=144, num_classes=15, dim=128, num_groups=4, channels=16,
    ):
        super().__init__(
            num_bands, num_classes, dim, num_groups, 
        )

        # 跨分支拼接后，将 64 通道投影回 32 通道
        self.cross_spectral = nn.Conv2d(
            2 * dim, dim, kernel_size=1
        )
        self.cross_spatial = nn.Conv2d(
            2 * dim, dim, kernel_size=1
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        # x = x.permute(0, 1, 3, 4, 2)    # [b,n,h,w,c]
        feature = self.stem(x)

        # 保留两路原始特征
        fs = self.spectral(feature)           # [B,32,H,W]
        fp = self.spatial(feature)            # [B,32,H,W]

        # 注意力输出，已经包含组内短残差
        fs_attn = self.spectral_attention(fs)
        fp_attn = self.spatial_attention(fp)

        # 本路注意力输出 + 对侧原始特征，沿通道拼接
        cross_s = torch.cat([fs_attn, fp], dim=1)
        cross_p = torch.cat([fp_attn, fs], dim=1)
        # 两者都是 [B,64,H,W]

        # 投影回 32 通道，再加本路长残差
        rs = fs + self.cross_spectral(cross_s)
        rp = fp + self.cross_spatial(cross_p)
        # 两者都是 [B,32,H,W]

        # 最终融合与分类
        fused = self.fusion(torch.cat([rs, rp], dim=1))
        vector = self.pool(fused).flatten(1)
        return self.head(vector)
class SpatialOnlyBaseline(nn.Module):
    def __init__(
        self,
        num_bands=144,
        num_classes=15,
        dim=32,
        num_groups=4
    ):
        super().__init__()

        # 1. 共享初始特征提取
        self.stem = ConvStem(num_bands, dim)

        # 2. 空间特征提取
        self.spatial = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        # 3. 分组空间注意力
        self.spatial_attention = GroupSpatialAttention(
            dim, num_groups
        )

        # 4. 分类
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # [B,144,11,11] -> [B,32,11,11]
        feature = self.stem(x)

        # 空间特征提取
        fp = self.spatial(feature)

        # 空间注意力
        fp = self.spatial_attention(fp)

        # 全局池化
        vector = self.pool(fp).flatten(1)

        # 分类
        return self.head(vector)
if __name__ == "__main__":
    model = CrossFusionBaseline()

    x = torch.randn(8, 144, 11, 11)
    labels = torch.randint(0, 15, (8,))

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    print("输出形状：", logits.shape)
    print("loss：", loss.item())

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    print("全部参数的梯度检查通过")