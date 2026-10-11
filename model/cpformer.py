import torch.nn as nn
from model.CPFormer import CPFormer as OriginalCPFormer


class CPFormer(nn.Module):
    def __init__(self, in_channels, num_classes, dim=64):
        super().__init__()
        self.net = OriginalCPFormer({
            "net": {
                "depth": 2,
                "heads": 8,
                "mlp_dim": 128,
                "kernal": 3,  # 原代码的参数名就是 kernal
                "padding": 1,
                "dropout": 0.1,
                "dim": dim,
            },
            "data": {
                "spectral_size": in_channels,
                "num_classes": num_classes,
            },
        })

    def forward(self, x):
        # [B,C,H,W] -> [B,H,W,C]
        x = x.permute(0, 2, 3, 1).contiguous()
        logits, _, _ = self.net(x)
        return logits