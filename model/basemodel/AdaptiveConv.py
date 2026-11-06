import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDilatedConv2d(nn.Module):
    """Pure PyTorch implementation of the AdaDR configuration used here.

    Reference: https://github.com/Linwei-Chen/FADC/blob/main/FADC_only/conv_custom.py
    Implements a 3x3, stride-1 convolution with one deformable group,
    replication padding, a spatial dilation map and nine modulation masks.
    AdaKern and FreqSelect are disabled, as in the original medium branch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        initial_dilation: int = 2,
        bias: bool = True,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError('in_channels and out_channels must be positive')
        if initial_dilation < 1:
            raise ValueError('initial_dilation must be at least 1')
        if epsilon < 0:
            raise ValueError('epsilon must be nonnegative')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_dilation = initial_dilation
        self.epsilon = epsilon
        self.PAD = nn.ReplicationPad2d(1)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.conv_offset = nn.Conv2d(in_channels, 1, 3)
        self.conv_mask = nn.Conv2d(in_channels, 9, 3)

        # Row-major (dy, dx), matching the reference deformable offsets.
        axis = torch.arange(-1, 2, dtype=torch.float32)
        dy, dx = torch.meshgrid(axis, axis, indexing='ij')
        offsets = torch.stack((dy, dx), dim=-1)
        self.register_buffer('dilated_offset', offsets.reshape(1, 1, 18, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        bound = (self.in_channels * 9) ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.constant_(
            self.conv_offset.bias,
            (self.initial_dilation - 1) / self.initial_dilation + self.epsilon,
        )
        nn.init.zeros_(self.conv_mask.weight)
        nn.init.zeros_(self.conv_mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f'Expected [N,{self.in_channels},H,W], got {tuple(x.shape)}'
            )
        n, _, h, w = x.shape
        padded = self.PAD(x)
        dilation = 1 + self.conv_offset(padded).abs() * self.initial_dilation
        mask = self.conv_mask(padded).sigmoid()

        # A deformable offset is added to the ordinary 3x3 sampling grid:
        # r + abs(raw) * initial_dilation * r = dilation * r.
        y, x_coord = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype) + 1,
            torch.arange(w, device=x.device, dtype=x.dtype) + 1,
            indexing='ij',
        )
        center = torch.stack((y, x_coord), dim=-1).reshape(1, h, w, 1, 2)
        offsets = self.dilated_offset.to(dtype=x.dtype).reshape(1, 1, 1, 9, 2)
        coordinates = center + dilation.permute(0, 2, 3, 1).unsqueeze(-1) * offsets
        sample_y, sample_x = coordinates.unbind(dim=-1)

        # grid_sample expects (x, y), normalized to the padded image.
        grid = torch.stack((
            2 * (sample_x + 0.5) / (w + 2) - 1,
            2 * (sample_y + 0.5) / (h + 2) - 1,
        ), dim=-1)
        sampled = F.grid_sample(
            padded,
            grid.reshape(n, h, w * 9, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        ).reshape(n, self.in_channels, h, w, 9)

        # Keep FADC's sigmoid mask (initial value 0.5); do not rescale it.
        sampled = sampled * mask.permute(0, 2, 3, 1).unsqueeze(1)
        output = torch.einsum('nchwk,ock->nohw', sampled, self.weight.flatten(2))
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        return output


class HSIAdaptiveSpatialConv(nn.Module):
    """Run the local 2-D AdaDR independently on each spectral plane.

    Input:  [B, 1, L, H, W]
    Output: [B, out_channels, L, H, W]

    No cross-band mixing occurs in this spatial path. The learned dilation
    map is predicted *separately for each band*; it is not shared across L.
    """

    def __init__(self, out_channels: int = 32, initial_dilation: int = 2):
        super().__init__()
        self.out_channels = out_channels
        self.adadr = AdaptiveDilatedConv2d(
            in_channels=1,
            out_channels=out_channels,
            initial_dilation=initial_dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f'Expected [B,1,L,H,W], got {tuple(x.shape)}')
        b, _, bands, h, w = x.shape
        planes = x.permute(0, 2, 1, 3, 4).reshape(b * bands, 1, h, w)
        planes = self.adadr(planes)  # [B*L, embed_dim, H, W]
        return planes.reshape(b, bands, self.out_channels, h, w).permute(0, 2, 1, 3, 4).contiguous()


class MultiScaleFeatureExtractorWithAdaDR(nn.Module):
    """Return fixed spectral + three spatial-scale features, all 5-D."""

    def __init__(self, embed_dim: int = 32, initial_dilation: int = 2):
        super().__init__()

        self.spectral = nn.Sequential(
            nn.Conv3d(1, embed_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_small = nn.Sequential(
            nn.Conv3d(1, embed_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_medium = nn.Sequential(
            HSIAdaptiveSpatialConv(embed_dim, initial_dilation),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_large = nn.Sequential(
            nn.Conv3d(1, embed_dim, kernel_size=(1, 7, 7), padding=(0, 3, 3)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f'Expected [B,L,H,W], got {tuple(x.shape)}')
        x3d = x.unsqueeze(1)  # [B, 1, L, H, W]
        return (
            self.spectral(x3d),
            self.spatial_small(x3d),
            self.spatial_medium(x3d),
            self.spatial_large(x3d),
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    example = torch.randn(2, 144, 11, 11, device=device)
    extractor = MultiScaleFeatureExtractorWithAdaDR(embed_dim=32).to(device)
    fs, fp1, fp2, fp3 = extractor(example)
    print('spectral :', fs.shape)
    print('small    :', fp1.shape)
    print('AdaDR    :', fp2.shape)
    print('large    :', fp3.shape)
