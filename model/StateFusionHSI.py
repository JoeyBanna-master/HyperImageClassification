"""HSI 中心像素分类：光谱提取、空间提取、可更新状态融合。

Reference topology: model/s2mamba.py (parallel branches, fusion, residual).
The extractors here are deliberately simple PyTorch convolutions, not Mamba.
After input projection, 'spectral' means learned-channel mixing, not a scan
along the original wavelength axis. Spatial coordinates are never downsampled.

Houston defaults: [B, 144, 11, 11] -> [B, 15] unnormalized logits.
All intermediate feature maps: [B, feature_dim, H, W].
content_features = x; fusion_state = z. State is local to each input patch,
updated without detach, and passed between blocks (never between batches).
"""

from typing import Optional

import torch
from torch import nn


class PixelChannelNorm(nn.Module):
    """在每个像素内部沿通道归一化；不混合空间位置或 batch 样本。"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.channel_norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channels_last = features.permute(0, 2, 3, 1)
        normalized = self.channel_norm(channels_last)
        return normalized.permute(0, 3, 1, 2).contiguous()


class InputFeatureProjection(nn.Module):
    """原始 patch [B,C,H,W] -> 初始主特征 x0 和融合状态 z0。"""

    def __init__(self, in_channels: int, feature_dim: int):
        super().__init__()
        self.content_state_projection = nn.Conv2d(in_channels, 2 * feature_dim, 1)
        self.activation = nn.GELU()

    def forward(self, hyperspectral_patch: torch.Tensor):
        projected_features = self.activation(
            self.content_state_projection(hyperspectral_patch)
        )
        content_features, fusion_state = projected_features.chunk(2, dim=1)
        return content_features, fusion_state


class SpectralFeatureExtractor(nn.Module):
    """光谱分支：每个像素独立的通道 MLP，D -> 2D -> D。

    1x1 卷积只混合通道，不访问邻居。这里处理投影后的隐特征通道，
    不声称学习原始波段顺序。可整体替换为同尺寸输出的光谱提取器。
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.input_norm = PixelChannelNorm(feature_dim)
        self.channel_expansion = nn.Conv2d(feature_dim, 2 * feature_dim, 1)
        self.activation = nn.GELU()
        self.channel_projection = nn.Conv2d(2 * feature_dim, feature_dim, 1)

    def forward(self, content_features: torch.Tensor) -> torch.Tensor:
        normalized_features = self.input_norm(content_features)
        expanded_features = self.activation(self.channel_expansion(normalized_features))
        return self.channel_projection(expanded_features)


class SpatialFeatureExtractor(nn.Module):
    """空间分支：逐通道 3x3 邻域卷积，再用 1x1 投影；保持 H、W。"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.input_norm = PixelChannelNorm(feature_dim)
        self.spatial_convolution = nn.Conv2d(
            feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(feature_dim, feature_dim, 1)

    def forward(self, content_features: torch.Tensor) -> torch.Tensor:
        normalized_features = self.input_norm(content_features)
        neighborhood_features = self.activation(self.spatial_convolution(normalized_features))
        return self.output_projection(neighborhood_features)


class StatefulFeatureFusion(nn.Module):
    """更新 z，再由新 z 决定光谱、空间分支的融合比例。

    z_next = z + Update(concat(z, spectral, spatial))
    weights = softmax(Gate(z_next), branch_axis)
    x_next = x + Project(weights_spectral * spectral + weights_spatial * spatial)

    权重为 [B,2,H,W]，每个像素两支权重之和为 1，同位置通道共享权重。
    返回 (x_next, z_next)，两者均为 [B,D,H,W]。无跨 batch 状态缓存。
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.state_updater = nn.Sequential(
            PixelChannelNorm(3 * feature_dim),
            nn.Conv2d(3 * feature_dim, feature_dim, 1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        self.branch_gate = nn.Sequential(
            PixelChannelNorm(feature_dim),
            nn.Conv2d(feature_dim, 2, 1),
            nn.Softmax(dim=1),
        )
        self.fused_projection = nn.Conv2d(feature_dim, feature_dim, 1)

    def forward(self, content_features, fusion_state, spectral_features, spatial_features):
        update_input = torch.cat(
            (fusion_state, spectral_features, spatial_features), dim=1
        )
        updated_fusion_state = fusion_state + self.state_updater(update_input)
        branch_weights = self.branch_gate(updated_fusion_state)
        spectral_weight = branch_weights[:, 0:1]
        spatial_weight = branch_weights[:, 1:2]
        fused_features = (
            spectral_weight * spectral_features + spatial_weight * spatial_features
        )
        updated_content_features = content_features + self.fused_projection(fused_features)
        return updated_content_features, updated_fusion_state


class SpectralSpatialBlock(nn.Module):
    """三个独立模块组成一个块；可通过构造参数或属性替换任一模块。

    Extractor interface: [B,D,H,W] -> [B,D,H,W].
    Fusion interface: (content, state, spectral, spatial) -> (content, state).
    """

    def __init__(
        self, feature_dim: int, *,
        spectral_extractor: Optional[nn.Module] = None,
        spatial_extractor: Optional[nn.Module] = None,
        feature_fusion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.spectral_extractor = (
            spectral_extractor if spectral_extractor is not None
            else SpectralFeatureExtractor(feature_dim)
        )
        self.spatial_extractor = (
            spatial_extractor if spatial_extractor is not None
            else SpatialFeatureExtractor(feature_dim)
        )
        self.feature_fusion = (
            feature_fusion if feature_fusion is not None
            else StatefulFeatureFusion(feature_dim)
        )

    def forward(self, content_features, fusion_state):
        spectral_features = self.spectral_extractor(content_features)
        spatial_features = self.spatial_extractor(content_features)
        return self.feature_fusion(
            content_features, fusion_state, spectral_features, spatial_features
        )


class CenterPixelClassifier(nn.Module):
    """取中心位置 [B,D]，归一化并输出 [B,K] logits。"""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.class_projection = nn.Linear(feature_dim, num_classes)

    def forward(self, content_features: torch.Tensor) -> torch.Tensor:
        height, width = content_features.shape[-2:]
        center_features = content_features[:, :, height // 2, width // 2]
        return self.class_projection(self.feature_norm(center_features))


class StateFusionHSI(nn.Module):
    """两个独立参数的块依次更新 (x,z)，默认适配 Houston 2013。

    forward(patch): classification logits, compatible with CrossEntropyLoss.
    forward_features(patch): final (content_features, fusion_state), useful
    for inspecting features without changing the training interface.
    """

    def __init__(
        self, in_channels: int = 144, num_classes: int = 15,
        feature_dim: int = 64, num_blocks: int = 2,
    ):
        super().__init__()
        for name, value in (
            ('in_channels', in_channels), ('num_classes', num_classes),
            ('feature_dim', feature_dim), ('num_blocks', num_blocks),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.input_projection = InputFeatureProjection(in_channels, feature_dim)
        self.blocks = nn.ModuleList([
            SpectralSpatialBlock(feature_dim) for _ in range(num_blocks)
        ])
        self.classifier = CenterPixelClassifier(feature_dim, num_classes)

    def forward_features(self, hyperspectral_patch: torch.Tensor):
        if hyperspectral_patch.ndim != 4:
            raise ValueError('Expected a patch tensor with shape [B,C,H,W]')
        if hyperspectral_patch.shape[1] != self.in_channels:
            raise ValueError(f'Expected {self.in_channels} input bands')
        height, width = hyperspectral_patch.shape[-2:]
        if height < 1 or width < 1 or height % 2 == 0 or width % 2 == 0:
            raise ValueError('Patch height and width must be positive odd integers')
        content_features, fusion_state = self.input_projection(hyperspectral_patch)
        for feature_block in self.blocks:
            content_features, fusion_state = feature_block(content_features, fusion_state)
        return content_features, fusion_state

    def forward(self, hyperspectral_patch: torch.Tensor) -> torch.Tensor:
        content_features, _ = self.forward_features(hyperspectral_patch)
        return self.classifier(content_features)
