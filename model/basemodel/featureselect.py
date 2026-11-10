
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExpert(nn.Module):
    """
    特征专家网络

    输入: [B, C, H, W]
    输出: [B, C, H, W]

    spatial=True:
        加入 Depthwise Conv，提取局部空间信息。

    spatial=False:
        使用 1x1 Conv 进行通道特征变换。
    """

    def __init__(self, dim, spatial=False):
        super().__init__()

        hidden_dim = max(dim // 2, 1)

        layers = []

        if spatial:
            layers.extend([
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=1,
                    groups=dim,
                    bias=False
                ),
                nn.GELU()
            ])

        layers.extend([
            nn.Conv2d(
                dim, hidden_dim,
                kernel_size=1,
                bias=False
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_dim, dim,
                kernel_size=1,
                bias=False
            )
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RegionAwareMixtureExperts(nn.Module):
    """
    ReMoE-inspired Feature Selection Module.

    Input:
        x: [B, C, H, W]

    Optional:
        saliency: [B, H, W] or [B, 1, H, W]

    Output:
        out: [B, C, H, W]

    Components:
        1. Shared Expert
        2. Center Expert
        3. Regional Experts
        4. Region Router
        5. Feature Integration
    """

    def __init__(
        self,
        dim=128,
        center_size=3,
        temperature=1.0
    ):
        super().__init__()

        self.dim = dim
        self.center_size = center_size
        self.temperature = temperature

        # ==========================================
        # 1. Shared Expert
        # ==========================================
        # 所有空间位置共享同一套特征变换参数

        self.shared_expert = FeatureExpert(
            dim=dim,
            spatial=False
        )

        # ==========================================
        # 2. Center Expert
        # ==========================================
        # 专门处理中心区域

        self.center_expert = FeatureExpert(
            dim=dim,
            spatial=True
        )

        # ==========================================
        # 3. Regional Experts
        # ==========================================
        # 四个区域对应四个独立专家

        self.region_experts = nn.ModuleList([
            FeatureExpert(dim, spatial=True)
            for _ in range(4)
        ])

        # ==========================================
        # 4. Region Router
        # ==========================================
        # 根据区域特征生成区域重要性分数

        hidden_dim = max(dim // 4, 16)

        self.region_router = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # 如果未来接入 SCAB 的 saliency，
        # 可以通过该参数调节其对路由的影响。
        self.saliency_scale = nn.Parameter(
            torch.tensor(1.0)
        )

        # ==========================================
        # 5. Feature Integration
        # ==========================================

        self.gamma = nn.Parameter(
            torch.tensor(0.1)
        )

        self.center_scale = nn.Parameter(
            torch.tensor(0.5)
        )

        self.region_scale = nn.Parameter(
            torch.tensor(1.0)
        )

        # 保存区域权重，方便后续可视化
        self.last_region_weights = None

    def split_regions(self, x):
        """
        将特征图划分为四个区域。

        支持奇数尺寸，例如 11x11。

        Returns:
            regions: 四个区域特征
            boxes: 四个区域的坐标
        """

        B, C, H, W = x.shape

        h_mid = (H + 1) // 2
        w_mid = (W + 1) // 2

        boxes = [
            (0, h_mid, 0, w_mid),
            (0, h_mid, w_mid, W),
            (h_mid, H, 0, w_mid),
            (h_mid, H, w_mid, W)
        ]

        regions = []

        for y1, y2, x1, x2 in boxes:
            regions.append(
                x[:, :, y1:y2, x1:x2]
            )

        return regions, boxes

    def calculate_region_weights(
        self,
        regions,
        saliency=None,
        boxes=None
    ):
        """
        根据区域特征计算动态区域权重。

        regions:
            四个区域特征

        saliency:
            可选的外部空间重要性图

        Returns:
            weights: [B, 4]
        """

        region_scores = []

        for region in regions:

            # 区域全局平均池化
            descriptor = region.mean(
                dim=(2, 3)
            )

            # 学习区域重要性
            score = self.region_router(
                descriptor
            )

            region_scores.append(score.squeeze(-1))

        # [B, 4]
        scores = torch.stack(
            region_scores,
            dim=1
        )

        # 如果提供 SCAB 的重要性图，
        # 将其作为额外的区域引导信息。
        if saliency is not None:

            if saliency.ndim == 3:
                saliency = saliency.unsqueeze(1)

            if saliency.shape[1] != 1:
                raise ValueError(
                    "saliency must have one channel"
                )

            if boxes is None:
                raise ValueError("boxes is required")

            saliency = saliency / (
                saliency.mean(
                    dim=(2, 3),
                    keepdim=True
                ) + 1e-6
            )

            saliency_scores = []

            for y1, y2, x1, x2 in boxes:

                score = saliency[
                    :, :, y1:y2, x1:x2
                ].mean(dim=(1, 2, 3))

                saliency_scores.append(score)

            saliency_scores = torch.stack(
                saliency_scores,
                dim=1
            )

            scores = (
                scores
                + self.saliency_scale * saliency_scores
            )

        weights = F.softmax(
            scores / self.temperature,
            dim=1
        )

        return weights

    def forward(self, x, saliency=None):

        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(
                f"Expected [B,{self.dim},H,W], "
                f"got {tuple(x.shape)}"
            )

        B, C, H, W = x.shape

        # ==========================================
        # 1. Shared Expert
        # ==========================================

        shared_out = self.shared_expert(x)

        # ==========================================
        # 2. Region Partition
        # ==========================================

        regions, boxes = self.split_regions(x)

        # ==========================================
        # 3. Region Routing
        # ==========================================

        region_weights = self.calculate_region_weights(
            regions,
            saliency=saliency,
            boxes=boxes
        )

        self.last_region_weights = (
            region_weights.detach()
        )

        # ==========================================
        # 4. Regional Experts
        # ==========================================

        region_outputs = []

        for i, expert in enumerate(
            self.region_experts
        ):

            region = regions[i]

            # 区域重要性转化为专家输出增益
            gain = (
                4.0 * region_weights[:, i]
            ).view(B, 1, 1, 1)

            expert_out = expert(region)

            expert_out = expert_out * gain

            region_outputs.append(expert_out)

        # ==========================================
        # 5. Region Restoration
        # ==========================================

        top_out = torch.cat(
            [
                region_outputs[0],
                region_outputs[1]
            ],
            dim=3
        )

        bottom_out = torch.cat(
            [
                region_outputs[2],
                region_outputs[3]
            ],
            dim=3
        )

        region_out = torch.cat(
            [top_out, bottom_out],
            dim=2
        )

        # ==========================================
        # 6. Center Expert
        # ==========================================

        ch = min(self.center_size, H)
        cw = min(self.center_size, W)

        top = (H - ch) // 2
        left = (W - cw) // 2

        center_x = x[
            :,
            :,
            top:top + ch,
            left:left + cw
        ]

        center_out = self.center_expert(
            center_x
        )

        # 将中心区域恢复到原始特征图
        center_out = F.pad(
            center_out,
            (
                left,
                W - left - cw,
                top,
                H - top - ch
            )
        )

        # ==========================================
        # 7. Feature Integration
        # ==========================================

        refinement = (
            shared_out
            + self.center_scale * center_out
            + self.region_scale * region_out
        )

        # 残差更新，保持原始特征信息
        out = x + self.gamma * refinement

        return out


if __name__ == "__main__":

    x = torch.randn(2, 128, 11, 11)

    model = RegionAwareMixtureExperts(
        dim=128,
        center_size=3
    )

    out = model(x)

    print("Input:", x.shape)
    print("Output:", out.shape)

    print(
        "Region Weights:",
        model.last_region_weights
    )
