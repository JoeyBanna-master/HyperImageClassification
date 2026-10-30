import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# class BasicAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#     def forward(self, x):
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)
#
#         k_proj  = k.transpose(1, 2)
#         scores = torch.bmm(q, k_proj) / math.sqrt(q.size(-1))
#         attn = F.softmax(scores, dim=-1)
#         out = torch.bmm(attn, v)
#         return out, attn
#
# class GroupSpatialAttention(nn.Module):
#     def __init__(self, dim, num_groups=4):
#         super().__init__()
#         assert num_groups > 0 and dim % num_groups == 0
#
#         self.num_groups = num_groups
#         # 每组有多少个特征通道
#         self.group_dim = dim // num_groups
#
#         self.attn = nn.ModuleList([
#             BasicAttention(self.group_dim)
#             for _ in range(num_groups)
#         ])
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert C == self.num_groups * self.group_dim
#
#         groups = x.chunk(self.num_groups, dim=1)
#         outputs = []
#
#         for group, attention in zip(groups, self.attn):
#             # 每次处理一组：[B, 8, 11, 11]
#             tokens = group.flatten(2).transpose(1, 2)
#             # tokens：[B, 121, 8]
#
#             update, _ = attention(tokens)
#             tokens = tokens + update  # 短残差
#
#             feature = tokens.transpose(1, 2).reshape(
#                 B, self.group_dim, H, W
#             )
#             outputs.append(feature)
#
#         return torch.cat(outputs, dim=1)  # [B, 32, 11, 11]

class GroupAxisAttention(nn.Module):
    """
    五维分组轴向自注意力

    输入/输出: [B, C, H, W, D]

    axis='spectral':
        对每个空间位置的 D 个波段做自注意力

    axis='spatial':
        对每个波段的 H*W 个空间位置做自注意力
    """
    def __init__(self, dim, num_groups=4, axis="spectral"):
        super().__init__()

        assert num_groups > 0
        assert dim % num_groups == 0
        assert axis in ("spectral", "spatial")

        self.num_groups = num_groups
        self.group_dim = dim // num_groups
        self.axis = axis

        # 每个组使用独立的 QKV 投影参数
        self.qkv = nn.ModuleList([
            nn.Linear(self.group_dim, 3 * self.group_dim)
            for _ in range(num_groups)
        ])

        self.proj = nn.ModuleList([
            nn.Linear(self.group_dim, self.group_dim)
            for _ in range(num_groups)
        ])

    def forward(self, x):
        B, C, H, W, D = x.shape
        assert C == self.num_groups * self.group_dim

        outputs = []

        for g, (qkv_layer, proj_layer) in enumerate(
            zip(self.qkv, self.proj)
        ):
            start = g * self.group_dim
            end = (g + 1) * self.group_dim

            # [B, group_dim, H, W, D]
            group = x[:, start:end]

            if self.axis == "spectral":
                # 每个空间位置视为一个独立光谱序列
                # [B*H*W, D, group_dim]
                tokens = group.permute(
                    0, 2, 3, 4, 1
                ).reshape(
                    B * H * W, D, self.group_dim
                )

            else:
                # 每个光谱位置对应一个空间序列
                # [B*D, H*W, group_dim]
                tokens = group.permute(
                    0, 4, 2, 3, 1
                ).reshape(
                    B * D, H * W, self.group_dim
                )

            # Q, K, V: [N, L, group_dim]
            q, k, v = qkv_layer(tokens).chunk(3, dim=-1)

            # 缩放点积注意力
            update = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0
            )

            update = proj_layer(update)

            # 组内残差
            tokens = tokens + update

            if self.axis == "spectral":
                # [B, group_dim, H, W, D]
                feature = tokens.reshape(
                    B, H, W, D, self.group_dim
                ).permute(0, 4, 1, 2, 3)

            else:
                # [B, group_dim, H, W, D]
                feature = tokens.reshape(
                    B, D, H, W, self.group_dim
                ).permute(0, 4, 2, 3, 1)

            outputs.append(feature)

        return torch.cat(outputs, dim=1)

class CrossFusionBaseline(nn.Module):
    def __init__(self, num_bands=144, num_classes=15, dim=32, num_groups=4, channels=16,patch = 11):
        super().__init__()

        self.num_bands = num_bands
        self.block = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(),)

        # 通道混合分支
        self.spectral = nn.Sequential(
            nn.Conv3d(channels, dim, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            nn.BatchNorm3d(dim),
            nn.ReLU(),)

        # 空间分支：每个通道独立提取空间邻域特征
        self.spatial = nn.Sequential(
            nn.Conv3d(channels, dim, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
        )
        # self.spectral_attention = GroupSpatialAttention(dim, num_groups)
        # self.spatial_attention = GroupSpatialAttention(dim, num_groups)

        # 分组光谱注意力
        self.spectral_attention = GroupAxisAttention(
            dim, num_groups, axis="spectral"
        )

        # 分组空间注意力
        self.spatial_attention = GroupAxisAttention(
            dim, num_groups, axis="spatial"
        )

        # 两个方向分别采用独立的跨分支投影
        self.cross_s = nn.Conv3d(2 * dim, dim, 1)
        self.cross_p = nn.Conv3d(2 * dim, dim, 1)

        # 最终特征融合
        self.fusion = nn.Sequential(
            nn.Conv3d(2 * dim, dim, 1),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )

        # 分类器：仅在最后聚合所有空间与光谱特征
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        assert x.ndim == 4
        assert x.shape[1] == self.num_bands

        # [B,144,11,11] -> [B,1,11,11,144]
        x = x.unsqueeze(1).permute(0, 1, 3, 4, 2)

        # [B,16,11,11,144]
        feature = self.block(x)

        # 两个分支均为 [B,32,11,11,144]
        fs = self.spectral(feature)
        fp = self.spatial(feature)

        # 分组轴向注意力
        fs_attn = self.spectral_attention(fs)
        fp_attn = self.spatial_attention(fp)

        # 跨分支特征拼接
        cross_s = torch.cat([fs_attn, fp], dim=1)
        cross_p = torch.cat([fp_attn, fs], dim=1)

        # 独立卷积投影 + 本路长残差
        rs = fs + self.cross_s(cross_s)
        rp = fp + self.cross_p(cross_p)

        # [B,64,11,11,144] -> [B,32,11,11,144]
        fused = self.fusion(torch.cat([rs, rp], dim=1))

        # [B,15]
        return self.fc(fused)


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