import torch
import torch.nn as nn

from model.basemodel.AdaptiveConv import MultiScaleFeatureExtractorWithAdaDR
from model.basemodel.scab_featurefusion import PairwiseCrossFusion


class MultiScaleSCABBaseline(nn.Module):
    def __init__(
        self,
        num_bands=144,
        embed_dim=32,
        dim=128,
        num_classes=15,
        initial_dilation=2,
        extractor=None,
        fusion_factory=PairwiseCrossFusion,
    ):
        super().__init__()
        self.num_bands = num_bands

        # 可传入其他提取器实例，要求保持四路输出接口。
        self.extractor = (
            MultiScaleFeatureExtractorWithAdaDR(
                embed_dim=embed_dim,
                initial_dilation=initial_dilation,
            )
            if extractor is None else extractor
        )

        def make_projection():
            return nn.Sequential(
                # [B,E,L,H,W] → [B,E*L,H,W] → [B,D,H,W]
                nn.Flatten(1, 2),
                nn.Conv2d(embed_dim * num_bands, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )

        self.spectral_project = make_projection()
        self.small_project = make_projection()
        self.medium_project = make_projection()
        self.large_project = make_projection()

        # 每次调用工厂，创建独立的融合模块。
        self.cross_small = fusion_factory(dim=dim)
        self.cross_medium = fusion_factory(dim=dim)
        self.cross_large = fusion_factory(dim=dim)

        self.merge = nn.Sequential(
            nn.Conv2d(3 * dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, num_classes)

    def forward_features(self, x):
        if x.ndim != 4 or x.shape[1] != self.num_bands:
            raise ValueError(f"Expected [B,{self.num_bands},H,W]")

        # 1. 四路特征提取。
        fs, fp_small, fp_medium, fp_large = self.extractor(x)

        # 2. 适配交叉融合模块的输入尺寸。
        fs = self.spectral_project(fs)
        fp_small = self.small_project(fp_small)
        fp_medium = self.medium_project(fp_medium)
        fp_large = self.large_project(fp_large)

        # 3. 同一份光谱特征，分别参与三组独立融合。
        return (
            self.cross_small(fs, fp_small),
            self.cross_medium(fs, fp_medium),
            self.cross_large(fs, fp_large),
        )

    def forward(self, x):
        features = self.forward_features(x)
        fused = self.merge(torch.cat(features, dim=1))
        return self.head(self.pool(fused).flatten(1))