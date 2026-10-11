#补充两个消融实验  1.完全解耦实部虚部效果大于联合复数值处理  2.补充加性融合优于乘性融合消融实验
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class SpectralBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

class SpectralLocal(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, 1)  # 1x1 投影
        self.block1 = SpectralBlock(out_channels)
        self.block2 = SpectralBlock(out_channels)
        self.block3 = SpectralBlock(out_channels)
        self.block4 = SpectralBlock(out_channels)
        self.block5 = SpectralBlock(out_channels)


    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        return x.view(b, -1, h, w).permute(0, 2, 3, 1)

# class DSResBlock(nn.Module):#95.97、95.73 、95.64
#     """
#  .000000000000000     Dual-Scale Residual Block (DS-ResBlock)
#     既保留了ResNet的稳定性，又引入了多尺度膨胀卷积来增强光谱特征提取。
#     """
#     def __init__(self, channels):
#         super().__init__()
#         # 1. 第一层：标准的特征提取 (保持不变)
#         self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(channels)
#
#         # 2. 第二层：创新点 - 分组双尺度处理
#         self.split_channels = channels // 2
#
#         # 分支A: 保持局部特征 (Local) - 普通卷积
#         self.conv2_local = nn.Conv1d(self.split_channels, self.split_channels, 3, padding=1, bias=False)
#
#         # 分支B: 捕获长程趋势 (Context) - 膨胀卷积 (Dilation=2)
#         # 膨胀卷积允许模型“跨越”观察波段，捕捉更宽的光谱形状
#         self.conv2_context = nn.Conv1d(self.split_channels, self.split_channels, 3, padding=2, dilation=2, bias=False)
#
#         self.bn2 = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU(inplace=True)
#
#         # 3. 可选：极轻量的通道注意力 (SE-Lite)，用于校准拼接后的特征
#         # 这有助于进一步稳定训练，特别是对于只有 144/103 波段的数据
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(channels, channels // 4, 1),
#             nn.ReLU(),
#             nn.Conv1d(channels // 4, channels, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         residual = x
#
#         # 第一阶段：标准卷积
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         # 第二阶段：Split -> Transform -> Concat
#         x1 = out[:, :self.split_channels, :]
#         x2 = out[:, self.split_channels:, :]
#
#         x1 = self.conv2_local(x1)
#         x2 = self.conv2_context(x2)
#
#         out = torch.cat([x1, x2], dim=1)
#         out = self.bn2(out)
#
#         # 第三阶段：Attention + Residual
#         # 加入 SE 注意力权重
#         out = out * self.se(out)
#
#         out += residual
#         return self.relu(out)
#
#
# class SpectralLocal(nn.Module):
#     def __init__(self, in_channels, out_channels=64):
#         super().__init__()
#         # 1x1 投影，将原始波段映射到特征维度
#         self.proj = nn.Conv1d(in_channels, out_channels, 1, bias=False)
#         self.bn_proj = nn.BatchNorm1d(out_channels)  # 加个BN更稳
#         self.relu = nn.ReLU(inplace=True)
#
#         # 使用 4 个 DSResBlock (原来是5个，稍微减少一个以加快速度，因为这个Block略复杂一点点)
#         # 如果显存允许，也可以保持 5 个
#         self.block1 = DSResBlock(out_channels)
#         self.block2 = DSResBlock(out_channels)
#         self.block3 = DSResBlock(out_channels)
#         self.block4 = DSResBlock(out_channels)
#
#         self.final_norm = nn.BatchNorm1d(out_channels)
#
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         # [B, C, H, W] -> [B, C, HW]
#         x = x.view(b, c, h * w)
#
#         # 投影
#         x = self.proj(x)
#         x = self.bn_proj(x)
#         x = self.relu(x)
#
#         # 残差堆叠
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#
#         x = self.final_norm(x)
#
#         # [B, C, HW] -> [B, H, W, C]
#         return x.view(b, -1, h, w).permute(0, 2, 3, 1)

# import torch
# import torch.nn as nn
#
#
# class DSResBlock(nn.Module):
#     """
#     DS-ResBlock (Optimized Version)
#     基于 Option B (OA~95.97%) 的微调版。
#     改进点:
#     1. ReLU -> GELU: 更平滑的激活函数，提升光谱细节保留能力。
#     2. SE Ratio 优化: 减少通道信息的压缩损失 (Ratio 4->2)。
#     """
#
#     def __init__(self, channels):
#         super().__init__()
#         # 1. 第一层：标准特征提取
#         self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(channels)
#
#         # 2. 第二层：分组双尺度处理 (核心创新点)
#         self.split_channels = channels // 2
#
#         # 分支A: 局部特征 (Local) - 普通卷积
#         self.conv2_local = nn.Conv1d(self.split_channels, self.split_channels, 3, padding=1, bias=False)
#
#         # 分支B: 长程趋势 (Context) - 膨胀卷积 (Dilation=2)
#         # 保持不变，因为这个结构已经被证明有效
#         self.conv2_context = nn.Conv1d(self.split_channels, self.split_channels, 3, padding=2, dilation=2, bias=False)
#
#         self.bn2 = nn.BatchNorm1d(channels)
#
#         # 3. 激活函数升级: ReLU -> GELU
#         self.act = nn.GELU()
#
#         # 4. SE Attention 优化: 压缩比改为 2 (原来是4)
#         # 减少信息损失，特别是对于 Houston/PU 这种波段数不是特别多的数据
#         reduction = 2
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(channels, channels // reduction, 1),
#             nn.GELU(),  # 这里也同步用 GELU
#             nn.Conv1d(channels // reduction, channels, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         residual = x
#
#         # 第一阶段
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.act(out)
#
#         # 第二阶段：Split -> Multi-Scale
#         x1 = out[:, :self.split_channels, :]
#         x2 = out[:, self.split_channels:, :]
#
#         x1 = self.conv2_local(x1)
#         x2 = self.conv2_context(x2)
#
#         # 拼接
#         out = torch.cat([x1, x2], dim=1)
#         out = self.bn2(out)
#
#         # 第三阶段：Attention + Residual
#         # SE 加权
#         out = out * self.se(out)
#
#         out += residual
#         return self.act(out)
#
#
# class SpectralLocal(nn.Module):
#     def __init__(self, in_channels, out_channels=64):
#         super().__init__()
#         # 投影层
#         self.proj = nn.Conv1d(in_channels, out_channels, 1, bias=False)
#         self.bn_proj = nn.BatchNorm1d(out_channels)
#         self.act = nn.GELU()  # 同步升级
#
#         # 堆叠 4 个 DSResBlock
#         # 保持 4 层结构，兼顾深度和效率
#         self.block1 = DSResBlock(out_channels)
#         self.block2 = DSResBlock(out_channels)
#         self.block3 = DSResBlock(out_channels)
#         self.block4 = DSResBlock(out_channels)
#
#         self.final_norm = nn.BatchNorm1d(out_channels)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         # [B, C, H, W] -> [B, C, HW]
#         x = x.view(b, c, h * w)
#
#         x = self.proj(x)
#         x = self.bn_proj(x)
#         x = self.act(x)
#
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#
#         x = self.final_norm(x)
#
#         # [B, C, HW] -> [B, H, W, C]
#         return x.view(b, -1, h, w).permute(0, 2, 3, 1)
# class SpectralBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         # 【改1】添加 bias=False（与 BN 配合）
#         self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
#         self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.bn2 = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU(inplace=True)  # inplace 节省内存
#
#     def forward(self, x):
#         residual = x
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += residual
#         return self.relu(x)
#
#
# class SpectralLocal(nn.Module):
#     def __init__(self, in_channels, out_channels=64):
#         super().__init__()
#         # 【改1】bias=False
#         self.proj = nn.Conv1d(in_channels, out_channels, 1, bias=False)
#         self.block1 = SpectralBlock(out_channels)
#         self.block2 = SpectralBlock(out_channels)
#         self.block3 = SpectralBlock(out_channels)
#         self.block4 = SpectralBlock(out_channels)
#
#         # 【改3】新增：轻量输出归一化（稳定后续融合）
#         self.final_norm = nn.BatchNorm1d(out_channels)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.view(b, c, h * w)  # [B, C_in, HW]
#         x = self.proj(x)  # [B, C_out, HW]
#
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#
#         # 【改2 & 3】先归一化，再移除最后 ReLU
#         x = self.final_norm(x)  # 稳定分布
#
#         x = x.view(b, -1, h, w).permute(0, 2, 3, 1)  # [B, H, W, C_out]
#         return x  # ← 不再加 ReLU！

class SpectralGlobal(nn.Module):
    def __init__(self, in_channels, dim=64,num_heads=4,patch_size=11):
        super().__init__()
        self.proj1 = nn.Linear(in_channels, dim)
        self.spectral_pe = nn.Parameter(torch.randn(dim) * 0.02)  # [dim]
        # self.attn1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.freq = SpectralGFEM(dim)   # ← 光谱频域增强（沿通道）
        # self.freq = CSFPR_SpectralFrequency(dim)#原模块消融
        self.proj2 = nn.Linear(dim, dim)
        self.attn = BCGA(in_dims=dim, token_dim=dim, num_heads=num_heads,H=patch_size,W=patch_size)
        # self.attn = EfficientViMBlock(dim=out_channels, mlp_ratio=4.0, ssd_expand=1, state_dim=64)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        # [B, C, H, W] → [B*H*W, C]
        x = x.permute(0, 2, 3, 1).reshape(-1, c)
        x = self.proj1(x)  # [N, dim]
        x = x + self.spectral_pe  # broadcast add

        # reshape to [B, dim, H, W] for SpectralFrequency
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2)  # [B, dim, H, W]
        x = self.freq(x)  # [B, dim, H, W] —— 沿 dim（光谱）做频域增强
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1)  # [B, N, dim]

        x = self.proj2(x)
        # x = x.transpose(1, 2)
        # res = x
        # attn_out, _ = self.attn1(x, x, x)
        # attn_out =res + attn_out
        attn_out = self.attn(x)
        # attn_out = attn_out.transpose(1, 2)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.view(b, h, w, -1)

# class SpectralGlobal(nn.Module):
#     def __init__(self, in_channels, dim=64, num_heads=4):
#         super().__init__()
#         self.proj = nn.Linear(in_channels, dim)
#         self.spectral_pe = nn.Parameter(torch.randn(dim) * 0.02)  # [dim]
#         self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.ReLU(),
#             nn.Linear(dim * 2, dim)
#         )
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.permute(0, 2, 3, 1).reshape(-1, c)  # [N, C]
#         x = self.proj(x)  # [N, dim]
#         x = x + self.spectral_pe  # broadcast add
#         attn_out, _ = self.attn(x, x, x)
#         x = self.norm1(x + attn_out)
#         x = self.norm2(x + self.ffn(x))
#         return x.view(b, h, w, -1)



class SpatialLocal(nn.Module):
    """空间局部特征提取：3个HFP模块，每层带残差连接（如图所示）"""
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # 3个HFP模块
        self.hfp1 = DFIM(
            in_channels=out_channels,
            ratio=(0.25, 0.25),
            patch=(8, 8),
            isdct=True
        )
        self.hfp2 = DFIM(
            in_channels=out_channels,
            ratio=(0.25, 0.25),
            patch=(8, 8),
            isdct=True
        )
        self.hfp3 = DFIM(
            in_channels=out_channels,
            ratio=(0.25, 0.25),
            patch=(8, 8),
            isdct=True
        )
        # self.hfp4 = DFIM(
        #     in_channels=out_channels,
        #     ratio=(0.25, 0.25),
        #     patch=(8, 8),
        #     isdct=True
        # )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W]
        residual = x  # 初始残差

        # 第一层 HFP + 残差
        x1 = self.hfp1(x)
        x = residual + x1  # ← 第一次加和
        residual = x  # 更新残差

        # 第二层 HFP + 残差
        x2 = self.hfp2(x)
        x = residual + x2  # ← 第二次加和
        residual = x

        # 第三层 HFP + 残差
        x3 = self.hfp3(x)
        x = residual + x3  # ← 第三次加和

        # # 第三层 HFP + 残差
        # x4 = self.hfp4(x)
        # x = residual + x4  # ← 第四次加和

        x = self.relu(x)
        return x

# class SpatialGlobal(nn.Module): #---base
#     """空间全局建模：使用Transformer"""
#     def __init__(self, in_channels, dim=64, num_heads=4):
#         super().__init__()
#         self.proj = nn.Linear(in_channels, dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.ReLU(),
#             nn.Linear(dim * 2, dim)
#         )
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, C]
#         x = self.proj(x)  # [B, HW, dim]
#         attn_out, _ = self.attn(x, x, x)
#         x = self.norm1(x + attn_out)
#         x = self.norm2(x + self.ffn(x))
#         x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)  # [B, dim, H, W]
#         return x

class SpatialGlobal(nn.Module): #---best1
    """空间全局建模：使用 EfficientAdditiveAttention"""
    def __init__(self, in_channels, dim=64, num_heads=4,patch_size=11):
        super().__init__()
        total_dim = dim * num_heads
        self.proj_in = nn.Linear(in_channels, total_dim)
        self.freq = SpatialGFEM(total_dim)
        # self.freq = CSFPR_Frequency(total_dim)
        # self.attn1 = nn.MultiheadAttention(total_dim, num_heads, batch_first=True) #多头
        self.attn = BCGA(in_dims=total_dim,token_dim=dim,num_heads=num_heads,H=patch_size,W=patch_size)
        # 注意：不再投影回 in_channels，而是保持为 dim
        self.dim = dim

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, C]
        x = self.proj_in(x)
        # res = x
        # attn_out, _ = self.attn1(x, x, x)# [B, N, total_dim] #多头
        # attn_out =res + attn_out
        # reshape to [B, dim, H, W] for SpectralFrequency
        # ===== 在 attn 前插入 Frequency =====
        x_4d = x.permute(0, 2, 1).view(B, -1, H, W)  # [B, total_dim, H, W]
        x_4d = self.freq(x_4d)  # [B, total_dim, H, W]
        x = x_4d.view(B, -1, N).permute(0, 2, 1)  # [B, N, total_dim]
        # ===================================
        x = self.attn(x)                               # [B, N, dim]
        x = x.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)  # [B, dim, H, W]
        return x

# class SpectralFrequency(nn.Module): #95.69 95.45 95.34
#     """
#     微调版：沿光谱波段维度（C）进行 1D 频域增强
#     """
#     def __init__(self, in_channels, reduction=2):  # 降低 reduction
#         super().__init__()
#         self.in_channels = in_channels
#         # 移除 LayerNorm（或可选加 BN）
#
#         self.shared_conv1 = nn.Conv1d(1, 1, kernel_size=1)
#         self.shared_dw = nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1)
#         self.act = nn.SiLU()
#
#         # 轻量频域交互 MLP
#         self.freq_mlp = nn.Sequential(
#             nn.Conv1d(2, 4, 1),
#             nn.SiLU(),
#             nn.Conv1d(4, 2, 1)
#         )
#
#         # 简化注意力
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(2, 1, 1),
#             nn.Sigmoid()
#         )
#         self.sa = nn.Sequential(
#             nn.Conv1d(2, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#         self.alpha = nn.Parameter(torch.zeros(1))  # 初始关闭
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         identity = x
#
#         x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
#         # x_norm = self.norm(x_flat)  # ← 注释掉 LayerNorm
#         x_norm = x_flat
#
#         x_fft = torch.fft.rfft(x_norm, dim=-1, norm='ortho')  # [N, F]
#
#         real = torch.real(x_fft).unsqueeze(1)  # [N, 1, F]
#         imag = torch.imag(x_fft).unsqueeze(1)  # [N, 1, F]
#
#         real = self.act(self.shared_dw(self.act(self.shared_conv1(real))))
#         imag = self.act(self.shared_dw(self.act(self.shared_conv1(imag))))
#
#         freq_cat = torch.cat([real, imag], dim=1)  # [N, 2, F]
#         freq_cat = self.freq_mlp(freq_cat)        # ← 新增交互
#
#         ca_w = self.ca(freq_cat)   # [N, 1, 1]
#         sa_w = self.sa(freq_cat)   # [N, 1, F]
#         freq_att = freq_cat * ca_w * sa_w
#
#         real_out, imag_out = freq_att.chunk(2, dim=1)
#         real_out = real_out.squeeze(1)
#         imag_out = imag_out.squeeze(1)
#         fft_recon = torch.complex(real_out, imag_out)
#
#         x_recon = torch.fft.irfft(fft_recon, n=C, dim=-1, norm='ortho')
#         x_recon = x_recon.view(B, H, W, C).permute(0, 3, 1, 2)
#
#         return identity + self.alpha * x_recon

# class SpectralFrequency(nn.Module):  #95.65 95.16 95.29
#     def __init__(self, in_channels, reduction=2):
#         super().__init__()
#         self.in_channels = in_channels
#         freq_len = in_channels // 2 + 1
#
#         self.freq_filter = nn.Sequential(
#             nn.Linear(freq_len, freq_len),
#             nn.SiLU(),
#             nn.Linear(freq_len, freq_len)
#         )
#         # ✅ 修复1: pos_embed 改为 [1, F]
#         self.pos_embed = nn.Parameter(torch.randn(1, freq_len) * 0.02)
#
#         # ✅ 修复2: 使用 InstanceNorm1d
#         self.freq_norm = nn.InstanceNorm1d(2)
#
#         self.freq_mlp = nn.Sequential(
#             nn.Conv1d(2, 4, 1),
#             nn.SiLU(),
#             nn.Conv1d(4, 2, 1)
#         )
#
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(2, 1, 1),
#             nn.Sigmoid()
#         )
#         self.sa = nn.Sequential(
#             nn.Conv1d(2, 1, 3, padding=1),
#             nn.Sigmoid()
#         )
#         self.alpha = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         identity = x
#         N = B * H * W
#
#         x_flat = x.permute(0, 2, 3, 1).reshape(N, C)  # [N, C]
#         x_fft = torch.fft.rfft(x_flat, dim=-1, norm='ortho')  # [N, F]
#
#         real = torch.real(x_fft)  # [N, F]
#         imag = torch.imag(x_fft)  # [N, F]
#
#         # ✅ 修复3: 确保广播正确
#         real = self.freq_filter(real) + self.pos_embed  # [N, F] + [1, F] → [N, F]
#         imag = self.freq_filter(imag) + self.pos_embed
#
#         real = real.unsqueeze(1)  # [N, 1, F]
#         imag = imag.unsqueeze(1)  # [N, 1, F]
#         freq_cat = torch.cat([real, imag], dim=1)  # [N, 2, F]
#
#         # ✅ 修复4: 直接归一化，无需 transpose
#         freq_cat = self.freq_norm(freq_cat)  # [N, 2, F]
#
#         freq_cat = self.freq_mlp(freq_cat)
#
#         ca_w = self.ca(freq_cat)
#         sa_w = self.sa(freq_cat)
#         freq_att = freq_cat * ca_w * sa_w
#
#         real_out, imag_out = freq_att.chunk(2, dim=1)
#         real_out = real_out.squeeze(1)
#         imag_out = imag_out.squeeze(1)
#         fft_recon = torch.complex(real_out, imag_out)
#
#         x_recon = torch.fft.irfft(fft_recon, n=C, dim=-1, norm='ortho')
#         x_recon = x_recon.view(B, H, W, C).permute(0, 3, 1, 2)
#
#         alpha = torch.clamp(self.alpha, min=-1.0, max=1.0)
#         return identity + alpha * x_recon

class SpectralGFEM(nn.Module):  # v2: 独立卷积 + MLP 残差
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.in_channels = in_channels

        # ✅ 独立卷积（核心改动）
        self.real_conv1 = nn.Conv1d(1, 1, kernel_size=1)
        self.real_dw = nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1)
        self.imag_conv1 = nn.Conv1d(1, 1, kernel_size=1)
        self.imag_dw = nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1)
        self.act = nn.SiLU()

        self.freq_mlp = nn.Sequential(
            nn.Conv1d(2, 4, 1),
            nn.SiLU(),
            nn.Conv1d(4, 2, 1)
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(2, 1, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        x_fft = torch.fft.rfft(x_flat, dim=-1, norm='ortho')

        real = torch.real(x_fft).unsqueeze(1)  # [N, 1, F]
        imag = torch.imag(x_fft).unsqueeze(1)

        # ✅ 独立处理
        real = self.act(self.real_dw(self.act(self.real_conv1(real))))
        imag = self.act(self.imag_dw(self.act(self.imag_conv1(imag))))

        freq_cat = torch.cat([real, imag], dim=1)  # [N, 2, F]

        # ✅ MLP 残差
        freq_cat = self.freq_mlp(freq_cat) + freq_cat

        ca_w = self.ca(freq_cat)
        sa_w = self.sa(freq_cat)
        freq_att = freq_cat * ca_w * sa_w

        real_out, imag_out = freq_att.chunk(2, dim=1)
        real_out = real_out.squeeze(1)
        imag_out = imag_out.squeeze(1)
        fft_recon = torch.complex(real_out, imag_out)

        x_recon = torch.fft.irfft(fft_recon, n=C, dim=-1, norm='ortho')
        x_recon = x_recon.view(B, H, W, C).permute(0, 3, 1, 2)

        return identity + self.alpha * x_recon  # 保留原 alpha（若掉点再加 sigmoid）

class SpatialGFEM(nn.Module):  #暂时最好  微调frequency 95.73 95.47 95.38
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.freq_norm = nn.BatchNorm2d(dim)

        self.shared_conv1 = nn.Conv2d(dim, dim, 1)
        self.shared_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()  # 升级激活

        # 增强版频域注意力（通道 + 空间）
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 4, dim * 2, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(dim * 2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.proj = nn.Conv2d(dim, dim, 1)
        # self.alpha = nn.Parameter(torch.ones(1))  # 可学习残差权重
        self.alpha = nn.Parameter(torch.ones(1))  # 可学习残差权重 1

    def forward(self, x):
        identity = x
        x_freq = self.freq_norm(x)

        ffted = torch.fft.rfft2(x_freq, norm='ortho')
        real = torch.real(ffted)
        imag = torch.imag(ffted)

        real = self.act(self.shared_dw(self.act(self.shared_conv1(real))))
        imag = self.act(self.shared_dw(self.act(self.shared_conv1(imag))))

        freq_cat = torch.cat([real, imag], dim=1)

        # 通道 + 空间注意力
        ca_w = self.ca(freq_cat)
        sa_w = self.sa(freq_cat)
        freq_att = freq_cat * ca_w * sa_w

        real_out, imag_out = freq_att.chunk(2, dim=1)
        fft_recon = torch.complex(real_out, imag_out)
        x_recon = torch.fft.irfft2(fft_recon, s=x.shape[-2:], norm='ortho')
        x_recon = self.proj(x_recon)

        # 可学习残差
        return identity + self.alpha * x_recon




# class LGA(nn.Module): #nowbest
#     def __init__(self, in_dims=512, token_dim=256, num_heads=4):
#         super().__init__()
#
#         # 线性变换层，将输入维度映射到token维度乘以头数
#         self.to_query = nn.Linear(in_dims, token_dim * num_heads)
#         self.to_key = nn.Linear(in_dims, token_dim * num_heads)
#
#         # 可学习的权重矩阵，用于计算query的重要性
#         self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
#         # 缩放因子，通常为根号下的token维度的倒数
#         self.scale_factor = token_dim ** -0.5
#         # 投影层，保持维度不变
#         self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
#         # 最终线性变换层，将结果压缩回原始token维度
#         self.final = nn.Linear(token_dim * num_heads, token_dim)
#
#     # 前向传播函数
#     def forward(self, x):
#
#         # torch.Size([3, 4096, 64])
#         # 通过线性变换层得到 Q K
#         Q = self.to_query(x) # torch.Size([3, 4096, 64])
#         K = self.to_key(x) # torch.Size([3, 4096, 64])
#         # 对Q、K进行归一化处理
#         Q = torch.nn.functional.normalize(Q, dim=-1) # 归一化最后一维 torch.Size([3, 4096, 64])
#         K = torch.nn.functional.normalize(K, dim=-1) # 归一化最后一维 torch.Size([3, 4096, 64])
#
#         # ----------------------------------------------------------------------------
#         # 图中左侧蓝色Q的全部计算
#         # 计算查询与可学习权重w_g的点积，得到每个查询的重要性
#         """
#             Q : torch.Size([3, 4096, 64])
#             self.w_g : torch.Size([64, 1])
#         """
#         Q_weight = Q @ self.w_g # BxNx1 (BxNxD @ Dx1) torch.Size([3, 4096, 1])
#         # 应用缩放因子，并且在行向量上归一化
#         A = Q_weight * self.scale_factor # BxNx1 torch.Size([3, 4096, 1]) * float
#         A = torch.nn.functional.normalize(A, dim=1) # BxNx1 torch.Size([3, 4096, 1])
#         # 计算全局描述符G，它是A和query的逐元素相乘后的总和
#         G = torch.sum(A * Q, dim=1) # BxD torch.Size([3, 4096, 1]) * torch.Size([3, 4096, 64]) 求和
#         # 重复G张量，使其形状匹配key，以便于后续运算
#         G = einops.repeat(
#             G, "b d -> b repeat d", repeat=K.shape[1]
#         ) # BxNxD torch.Size([3, 4096, 64])
#         # ----------------------------------------------------------------------------
#
#         # 图中右侧 相乘
#         # 将G与key相乘后通过投影层，并加上原始查询
#         """
#             G :torch.Size([3, 4096, 64])
#             K :torch.Size([3, 4096, 64])
#             Q :torch.Size([3, 4096, 64])
#         """
#         out = self.Proj(G * K) + Q # BxNxD torch.Size([3, 4096, 64])
#
#         # 通过最终线性层输出
#         out = self.final(out) # BxNxD torch.Size([3, 4096, 32])
#         return out #用于替换光谱和空间全局的多头注意力机制


import torch
import torch.nn as nn
import einops


class BCGA(nn.Module): #95.88 95.66 95.54/95.95 95.74 95.62/95.76 95.59 95.41
    def __init__(self, in_dims=512, token_dim=256, num_heads=4, H=11, W=11):
        super().__init__()
        self.H, self.W = H, W
        self.N = H * W

        # LGA core
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        # ✅ 小方差初始化（关键！）
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1) * 0.01)

        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.norm_out = nn.LayerNorm(token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

        # Spatial & Channel Attention (no sigmoid!)
        self.spatial_conv = nn.Conv2d(in_dims, in_dims, 3, 1, 1)
        self.spatial_norm = nn.BatchNorm2d(in_dims)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv2d(in_dims, in_dims, 1)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.N, f"Token number {N} != H*W={self.N}"

        # Step 1: LGA core
        Q = self.to_query(x)
        K = self.to_key(x)
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # ✅ L2 normalization for sharp attention
        A = Q @ self.w_g  # [B, N, 1]
        A = F.normalize(A, dim=1)  # L2 over tokens

        G = torch.sum(A * Q, dim=1)
        G = einops.repeat(G, "b d -> b repeat d", repeat=N)

        # Step 2-3: Spatial & Channel (ReLU instead of Sigmoid)
        x_reshaped = x.transpose(1, 2).reshape(B, C, self.H, self.W)

        # Spatial
        spatial_feat = self.spatial_norm(x_reshaped)
        spatial_feat = self.spatial_conv(spatial_feat)
        spatial_feat = F.relu(spatial_feat)
        spatial_weight_flat = spatial_feat.reshape(B, C, -1).transpose(1, 2)

        # Channel
        channel_feat = self.avgpool(x_reshaped)
        channel_feat = self.channel_conv(channel_feat)
        channel_feat = F.relu(channel_feat)
        # channel_feat = self.channel_conv(channel_feat)#1
        # channel_feat = F.relu(channel_feat)#1

        channel_weight_flat = channel_feat.squeeze(-1).squeeze(-1)
        channel_weight_expanded = einops.repeat(channel_weight_flat, "b c -> b n c", n=N)

        # Step 4: Additive attention bias
        attention_bias = torch.sigmoid(spatial_weight_flat) + torch.sigmoid(channel_weight_expanded)

        # Step 5: Fusion + Non-linearity
        out = self.Proj(G * K) + Q + attention_bias

        # out = self.Proj(G * K) + Q
        out = self.norm_out(out)
        out = F.gelu(out)
        out = self.final(out)
        return out

import torch_dct as DCT
# 定义DCT空间交互模块
class DctSpatialInteraction(nn.Module):#the raw
    def __init__(self,
                 in_channels, # 输入特征图的通道数
                 ratio, # 用于计算高频保留比例的参数
                 isdct=True): # 标记是否使用DCT变换，True时在p1&p2中使用，False时在p3&p4中使用
        # 调用父类nn.Module的初始化方法
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct

        # 如果不使用DCT，创建1x1卷积用于空间注意力
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                # 1x1卷积将输入通道数转为1，用于生成空间注意力图
                *[nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)]
            )

    # 计算权重矩阵的方法
    def _compute_weight(self, h, w, ratio):
        # 根据比例计算低频区域的高度和宽度
        h0 = int(h * ratio[0]) # 高度方向的低频区域比例
        w0 = int(w * ratio[1]) # 宽度方向的低频区域比例
        # 创建全为1的权重矩阵，大小与输入特征图的空间维度相同
        weight = torch.ones((h, w), requires_grad=False) # 不需要计算梯度
        # 将低频区域（左上角）的权重设为0，实现过滤低频特征的效果
        weight[:h0, :w0] = 0
        return weight
    def forward(self, x): # x是输入特征图
        # 获取输入特征图的形状：batch_size, channels, height, width
        _, _, h0, w0 = x.size()
        # 如果不使用DCT，直接通过1x1卷积生成空间注意力并与输入相乘
        if not self.isdct:
            # 用sigmoid将卷积结果归一化到0-1，作为注意力权重
            return x * torch.sigmoid(self.spatial1x1(x))
        # 对输入特征图进行二维DCT变换
        idct = DCT.dct_2d(x, norm='ortho') # 使用正交归一化 二维离散余弦变换
        # 计算权重矩阵并移动到与输入相同的设备（CPU/GPU）
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        # 调整权重形状并扩展到与DCT结果相同的形状
        weight = weight.view(1, h0, w0).expand_as(idct)
        # 用权重过滤低频特征（保留高频特征）
        dct = idct * weight # 过滤掉低频特征
        # 对处理后的DCT结果进行逆DCT变换，生成空间掩码
        dct_ = DCT.idct_2d(dct, norm='ortho')
        # 将输入特征图与生成的空间掩码相乘
        return x * dct_

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
#
#
# class CSFPR_FourierUnit_2D(nn.Module):
#     """提取自 CSFPR：用于 2D 空间全局的频域处理"""
#
#     def __init__(self, channels, groups=4):
#         super().__init__()
#         self.groups = groups
#         self.bn = nn.BatchNorm2d(channels * 2)
#
#         self.fdc = nn.Conv2d(channels * 2, channels * 2 * self.groups, kernel_size=1, groups=self.groups)
#         self.weight = nn.Sequential(
#             nn.Conv2d(channels * 2, self.groups, kernel_size=1),
#             nn.Softmax(dim=1)
#         )
#         self.fpe = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # 2D FFT
#         ffted = torch.fft.rfft2(x, norm='ortho')
#         real = torch.real(ffted).unsqueeze(-1)
#         imag = torch.imag(ffted).unsqueeze(-1)
#         ffted = torch.cat((real, imag), dim=-1)
#         ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
#
#         # 核心 CSFPR 动态加权机制
#         ffted = self.bn(ffted)
#         ffted = self.fpe(ffted) + ffted
#         dy_weight = self.weight(ffted)
#
#         ffted = self.fdc(ffted).view(b, self.groups, 2 * c, h, -1)
#         ffted = torch.einsum('b g c h w, b g h w -> b c h w', ffted, dy_weight)
#         ffted = F.gelu(ffted)
#
#         # 2D IFFT
#         ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
#         ffted = torch.view_as_complex(ffted)
#         return torch.fft.irfft2(ffted, s=(h, w), norm='ortho')


# class CSFPR_Frequency(nn.Module):
#     """消融替换目标：Frequency"""
#
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.freq_norm = nn.BatchNorm2d(dim)
#         self.FFC = CSFPR_FourierUnit_2D(dim)
#         self.proj = nn.Conv2d(dim, dim, 1)
#         self.alpha = nn.Parameter(torch.ones(1))
#
#     def forward(self, x):
#         identity = x
#         x_freq = self.freq_norm(x)
#         x_recon = self.FFC(x_freq)
#         x_recon = self.proj(x_recon)
#         return identity + self.alpha * x_recon
#
#
# class CSFPR_FourierUnit_1D(nn.Module):
#     """衍生自 CSFPR：用于 1D 光谱全局的频域处理 (通道即频段)"""
#
#     def __init__(self, groups=2):
#         super().__init__()
#         self.groups = groups
#         # 光谱 1D FFT 只有实部和虚部 2 个通道
#         self.bn = nn.BatchNorm1d(2)
#
#         self.fdc = nn.Conv1d(2, 2 * self.groups, kernel_size=1, groups=1)
#         self.weight = nn.Sequential(
#             nn.Conv1d(2, self.groups, kernel_size=1),
#             nn.Softmax(dim=1)
#         )
#         self.fpe = nn.Conv1d(2, 2, kernel_size=3, padding=1, groups=2)
#
#     def forward(self, x):
#         B, C, H, W = x.size()
#         # [B, C, H, W] -> [B*H*W, C]，沿光谱维度 C 展开
#         x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
#
#         # 1D FFT 沿光谱维度
#         ffted = torch.fft.rfft(x_flat, dim=-1, norm='ortho')  # [N, F]
#         real = torch.real(ffted).unsqueeze(1)  # [N, 1, F]
#         imag = torch.imag(ffted).unsqueeze(1)  # [N, 1, F]
#         ffted_cat = torch.cat((real, imag), dim=1)  # [N, 2, F]
#
#         # 核心 CSFPR 动态加权机制 (1D 版)
#         ffted_cat = self.bn(ffted_cat)
#         ffted_cat = self.fpe(ffted_cat) + ffted_cat
#         dy_weight = self.weight(ffted_cat)  # [N, groups, F]
#
#         ffted_cat = self.fdc(ffted_cat).view(-1, self.groups, 2, ffted_cat.size(-1))
#         ffted_cat = torch.einsum('n g c f, n g f -> n c f', ffted_cat, dy_weight)
#         ffted_cat = F.gelu(ffted_cat)
#
#         # 1D IFFT 返回光谱维度
#         real_out, imag_out = ffted_cat.chunk(2, dim=1)
#         fft_recon = torch.complex(real_out.squeeze(1), imag_out.squeeze(1))
#         x_recon = torch.fft.irfft(fft_recon, n=C, dim=-1, norm='ortho')
#
#         return x_recon.view(B, H, W, C).permute(0, 3, 1, 2)
#
#
# class CSFPR_SpectralFrequency(nn.Module):
#     """消融替换目标：SpectralFrequency"""
#
#     def __init__(self, dim):
#         super().__init__()
#         # 在这里 dim 不需要传给 FFC，因为 1D FFT 的输入通道固定为 2 (实/虚)
#         self.FFC = CSFPR_FourierUnit_1D()
#         self.alpha = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         identity = x
#         x_recon = self.FFC(x)
#         return identity + self.alpha * x_recon
#
def _high_frequency(module, x, ratio):
    """Cache only the constant mask, never features or autograd graphs."""
    h, w = x.shape[-2:]
    key = (h, w, tuple(ratio), x.device, x.dtype)
    if getattr(module, '_frequency_mask_key', None) != key:
        mask = x.new_ones((1, 1, h, w))
        mask[..., :int(h * ratio[0]), :int(w * ratio[1])] = 0
        # Non-persistent buffer keeps legacy state_dict keys compatible.
        module.register_buffer('_frequency_mask', mask, persistent=False)
        module._frequency_mask_key = key
    coefficients = DCT.dct_2d(x, norm='ortho')
    return DCT.idct_2d(coefficients * module._frequency_mask, norm='ortho')


class DSI(nn.Module):#95.91 95.64 95.58
    def __init__(self,
                 in_channels,
                 ratio,
                 isdct=True):
        super(DSI, self).__init__()
        self.ratio = ratio
        self.isdct = isdct

        if not self.isdct:
            # 非DCT分支保持轻量
            self.spatial1x1 = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        else:
            # === 核心改进区域 ===
            # 1. 输入标准化：防止高频幅值波动
            num_groups = max(1, in_channels // 8)
            self.dct_norm = nn.GroupNorm(num_groups, in_channels)

            # 2. 空间门控机制 (Spatial Gating)
            # 这是一个"三明治"结构：降维 -> 激活 -> 升维
            # 作用：过滤掉高频噪声，只保留显著的边缘纹理
            reduction = 4  # 压缩比，减少参数量同时强制提取核心特征
            self.gate_network = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // reduction),  # BN在这里很关键，加速收敛
                nn.GELU(),  # 引入非线性
                nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
                nn.Sigmoid()  # 生成 [0,1] 的门控权重
            )

            # 3. 原始高频特征的变换层 (Depthwise)
            self.source_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1,
                                         groups=in_channels, bias=False)

            # 4. 可学习的缩放系数
            self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为0，零初始化残差分支

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

    def forward(self, x, high_frequency=None):
        _, _, h0, w0 = x.size()

        if not self.isdct:
            return x * self.spatial1x1(x)

        dct_ = high_frequency if high_frequency is not None else _high_frequency(self, x, self.ratio)

        # 2. 归一化
        high_freq = self.dct_norm(dct_)

        # 3. 生成门控权重 (Gating)
        # 这一步决定了高频特征图中，哪些位置是"真边缘"，哪些是"噪声"
        gate = self.gate_network(high_freq)

        # 4. 对高频特征进行加权过滤
        # clean_high_freq 只保留了通过门控的重要高频信息
        source = self.source_conv(high_freq)
        clean_high_freq = source * gate

        # 5. 残差注入
        # 初始时 alpha=0，clean_high_freq 不起作用，模型退化为 Identity
        # 随着训练，alpha 逐渐增大，注入"干净的"高频细节
        return x + self.alpha * clean_high_freq

# ==========================================
# 最终增强版 v7: 语义引导的空间交互模块
# ==========================================
# class DctSpatialInteraction(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  ratio,
#                  isdct=True):
#         super(DctSpatialInteraction, self).__init__()
#         self.ratio = ratio
#         self.isdct = isdct
#
#         if not self.isdct:
#             # 非DCT分支：保持极简
#             self.spatial1x1 = nn.Sequential(
#                 nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
#                 nn.Sigmoid()
#             )
#         else:
#             # === 核心微调区域 (v7: Semantic-Guided Joint Gating) ===
#
#             # 1. 稳定性基石：GroupNorm
#             # 仍然对高频特征做归一化，防止数值爆炸
#             num_groups = max(1, in_channels // 8)
#             self.dct_norm = nn.GroupNorm(num_groups, in_channels)
#
#             # 2. 联合门控网络 (Joint Gate Network)
#             # 输入通道是 2 * in_channels (原始特征 + 高频特征)
#             # 逻辑：利用 x 的语义稳定性来"监督" dct 的高频噪声
#             reduction = 8
#             self.joint_gate = nn.Sequential(
#                 nn.Conv2d(in_channels * 2, in_channels // reduction, kernel_size=1, bias=False),
#                 nn.GroupNorm(1, in_channels // reduction),  # 使用 GN 替代 LayerNorm/BN
#                 nn.GELU(),
#                 nn.Conv2d(in_channels // reduction, 1, kernel_size=1, bias=False),
#                 nn.Sigmoid()  # 生成 [B, 1, H, W] 的空间掩码
#             )
#
#             # 3. 特征对齐层
#             # 1x1 卷积，调整高频特征的通道分布以匹配 x
#             self.proj_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
#
#             # 4. 可学习系数 (Zero Init)
#             # 初始为 0，保证训练起始阶段完全等同于无 DCT 分支，杜绝性能下降风险
#             self.alpha = nn.Parameter(torch.zeros(1))
#
#     def _compute_weight(self, h, w, ratio):
#         h0 = int(h * ratio[0])
#         w0 = int(w * ratio[1])
#         weight = torch.ones((h, w), requires_grad=False)
#         weight[:h0, :w0] = 0
#         return weight
#
#     def forward(self, x):
#         _, _, h0, w0 = x.size()
#
#         if not self.isdct:
#             return x * self.spatial1x1(x)
#
#         # 1. DCT 提取高频
#         idct = DCT.dct_2d(x, norm='ortho')
#         weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
#         weight = weight.view(1, h0, w0).expand_as(idct)
#         dct = idct * weight
#         dct_ = DCT.idct_2d(dct, norm='ortho')  # [B, C, H, W]
#
#         # 2. 归一化高频特征
#         dct_feat = self.dct_norm(dct_)
#
#         # 3. 联合特征生成门控 (关键改进)
#         # 将原始语义 x 和 高频细节 dct_feat 拼接
#         # 让网络自己判断：哪些高频细节是符合语义结构的？
#         combined = torch.cat([x, dct_feat], dim=1)  # [B, 2C, H, W]
#         gate = self.joint_gate(combined)  # [B, 1, H, W]
#
#         # 4. 过滤与投影
#         # 只保留通过校验的高频特征
#         refined_high_freq = self.proj_conv(dct_feat) * gate
#
#         # 5. 稳定注入
#         # x + alpha * refined_noise
#         return x + self.alpha * refined_high_freq
# 定义DCT通道交互模块
class DCI(nn.Module):#the raw
    def __init__(self,
                 in_channels, # 输入特征图的通道数
                 patch, # 用于池化的补丁大小
                 ratio, # 用于计算高频保留比例的参数
                 isdct=True # 标记是否使用DCT变换
                 ):
        super(DCI, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0] # 补丁的高度
        self.w = patch[1] # 补丁的宽度
        self.ratio = ratio
        self.isdct = isdct

        # 1x1卷积，用于通道注意力计算，使用分组卷积（32组）
        self.channel1x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.channel2x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.relu = nn.ReLU() # ReLU激活函数
    # 计算权重矩阵的方法（与空间交互模块中的实现相同）
    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0# 将低频区域权重设为0
        return weight
    # 前向传播方法
    def forward(self, x, high_frequency=None): # x是输入特征图
        # 获取输入特征图的形状：batch_size, channels, height, width
        n, c, h, w = x.size()

        # 如果不使用DCT，使用普通的通道注意力机制
        if not self.isdct:# true时在p1&p2中使用，false时在p3&p4中使用
            # 对输入进行自适应最大池化，得到1x1的特征图
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            # 对输入进行自适应平均池化，得到1x1的特征图
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))

            # 将最大池化和平均池化的结果通过ReLU激活后，再通过1x1卷积，最后相加
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))

            # 生成通道注意力权重并与输入相乘
            return x * torch.sigmoid(self.channel2x1(channel))
        dct_ = high_frequency if high_frequency is not None else _high_frequency(self, x, self.ratio)
        # 对逆DCT结果进行自适应最大池化和平均池化，输出大小为patch大小
        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        # 对池化结果应用ReLU激活，然后在空间维度上求和，调整形状为(batch_size, channels, 1, 1)
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)
        # 计算通道注意力
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        # 生成通道注意力权重并与输入相乘
        return x * torch.sigmoid(self.channel2x1(channel))

#定义高频感知模块
class DFIM(nn.Module): #the raw
    def __init__(self,
                 in_channels, # 输入特征图的通道数
                 ratio=(0.25, 0.25), # 高频保留比例，默认保留75%的高频区域
                 patch=(8, 8), # 池化补丁大小
                 isdct=True): # 是否使用DCT变换
        super(DFIM, self).__init__()

        # 创建空间交互子模块
        self.spatial = DSI(in_channels, ratio=ratio, isdct=isdct)
        # 创建通道交互子模块
        self.channel = DCI(in_channels, patch=patch, ratio=ratio, isdct=isdct)

        # 输出处理模块：3x3卷积（保持空间大小）+ 分组归一化
        self.out = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
              nn.GroupNorm(32, in_channels)] # 使用32组的分组归一化
        )
    def forward(self, x): # x是输入特征图
        # 通过空间交互模块得到空间注意力加权后的特征
        shared = None
        if (self.spatial.isdct and self.channel.isdct
                and self.spatial.ratio == self.channel.ratio):
            shared = _high_frequency(self, x, self.spatial.ratio)
        spatial = self.spatial(x, high_frequency=shared)
        # 通过通道交互模块得到通道注意力加权后的特征
        channel = self.channel(x, high_frequency=shared)
        # 将空间和通道注意力的结果相加，再通过输出处理模块
        return self.out(spatial + channel)




class DoubleBranchFusion(nn.Module): #THE BEST
    """二分支融合模块：仅融合，不分类"""
    def __init__(self, in_channels, out_channels=None, reduction=16):
        """
        Args:
            in_channels: 输入特征的通道数 (e.g., 64)
            out_channels: 融合后输出的通道数。如果为None，则等于in_channels。
        """
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels

        def make_branch():
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
                nn.Sigmoid()
            )

        self.branch_spatial = make_branch()
        self.branch_spectral = make_branch()

        # 可选：一个额外的卷积层来调整输出通道数（如果需要）
        if self.out_channels != in_channels:
            self.proj_out = nn.Conv2d(in_channels, self.out_channels, 1)
        else:
            self.proj_out = nn.Identity()

    def forward(self, spatial, spectral):
        # 【关键修复】将输入从 [B, H, W, C] 转换为 [B, C, H, W]
        spatial = spatial.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        spectral = spectral.permute(0, 3, 1, 2)

        # 计算注意力权重
        w_s = self.branch_spatial(spatial)  # [B, C, 1, 1]
        w_p = self.branch_spectral(spectral)  # [B, C, 1, 1]
        # 加权融合
        fused = spatial * torch.sigmoid(w_s) + spectral * torch.sigmoid(w_p)  # [B, C, H, W]
        # fused = spatial * torch.sigmoid(w_p) + spectral * torch.sigmoid(w_s)  # [B, C, H, W]
        # fused = spatial * w_s + spectral * w_p  # [B, C, H, W]
        # 调整输出通道 (如果需要)
        fused = self.proj_out(fused)  # [B, out_channels, H, W]

        # 转回 [B, H, W, out_channels] 格式以保持一致性
        return fused.permute(0, 2, 3, 1)  # [B, H, W, out_channels]



class Classifier(nn.Module): #增加非线性层
    def __init__(self, in_dim, num_classes, hidden_dim=None, dropout=0.5):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim  # 或者 in_dim // 2

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 如果输入是 [B, H, W, D]，先转为4D
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [B, H, W, D]
        x = x.permute(0, 3, 1, 2)  # [B, D, H, W] for AdaptiveAvgPool2d
        return self.classifier(x)

class FBCANet(nn.Module):
    """高光谱图像分类主网络（Patch-level）"""
    def __init__(self, in_channels, num_classes, dim=64,reduction=16,patch_size=11):
        super().__init__()
        self.spectral_local = SpectralLocal(in_channels, dim)
        self.spectral_global = SpectralGlobal(in_channels, dim,num_heads=1,patch_size=patch_size)

        self.spatial_local = SpatialLocal(in_channels, dim)
        self.spatial_global = SpatialGlobal(in_channels, dim,num_heads=4,patch_size=patch_size) #仅用于LGA
        self.freq_proj = nn.Conv2d(in_channels, dim, kernel_size=1, bias=False)
        # self.freq = Frequency(dim)
        # self.freq = CSFPR_Frequency(dim)
        self.fusion2 = DoubleBranchFusion(dim, dim, reduction)

        self.classifier = Classifier(dim, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        spec_local = self.spectral_local(x)    # [B, H, W, dim]
        spec_global = self.spectral_global(x)  # [B, H, W, dim]

        spa_local = self.spatial_local(x)      # [B, dim, H, W]
        spa_global = self.spatial_global(x)    # [B, dim, H, W]
        # spec_global = spec_global.permute(0, 2, 3, 1)
        spa_local = spa_local.permute(0, 2, 3, 1)  # [B, H, W, dim]
        spa_global = spa_global.permute(0, 2, 3, 1)

        combined_spatial = spa_local + spa_global
        combined_spectral = spec_local + spec_global

        fused_feat = self.fusion2(combined_spatial, combined_spectral)
        pred = self.classifier(fused_feat)  # [B, num_classes]
        return pred