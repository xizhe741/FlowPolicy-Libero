"""
模型基础模块.

1D U-Net 子模块 (C2 复现, 与 Diffusion Policy 官方 ConditionalUnet1D 对齐):
- sinusoidal_embedding: 标量时间 t -> 正弦/余弦位置编码, max_period=10000
- FiLMResBlock1D:       两层 Conv1d-GroupNorm-Mish 之间注入 FiLM scale/bias
- Downsample1d:         stride=2 Conv1d
- Upsample1d:           ConvTranspose1d (kernel=4, stride=2, padding=1)
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(embedded_dim: int, t: torch.Tensor) -> torch.Tensor:
    """t: (B,) -> (B, embedded_dim).
    max_period=10000, 分母 (half - 1) 与 DP positional_embedding.SinusoidalPosEmb 一致.
    """
    half = embedded_dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1)).to(t.device)
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class FiLMResBlock1D(nn.Module):
    """DP ConditionalResidualBlock1D (cond_predict_scale=True).

    流程 (DP 官方顺序, Conv1dBlock 是 Conv -> GroupNorm -> Mish 而非 Norm -> Activate -> Conv):
        out = conv1 -> norm1 -> activate                    # blocks[0]
        scale, bias = cond_linear(activate(cond)).chunk(2)   # FiLM 系数
        out = scale * out + bias
        out = conv2 -> norm2 -> activate                    # blocks[1]
        return out + residual_conv(x)

    cond_linear 用 PyTorch 默认 Kaiming uniform 初始化 (锁定决策).
    FiLM 形式是 scale * out + bias (无 1 + scale 偏置), 与 DP 一致.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activate = nn.Mish()
        self.cond_linear = nn.Linear(cond_dim, 2 * out_channels)
        self.out_channels = out_channels
        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H);  cond: (B, cond_dim)
        res = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activate(x)

        scale, bias = self.cond_linear(self.activate(cond)).chunk(2, dim=-1)
        x = scale.unsqueeze(-1) * x + bias.unsqueeze(-1)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activate(x)

        return x + res


class Downsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
