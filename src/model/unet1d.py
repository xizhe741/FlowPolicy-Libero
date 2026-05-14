"""
ConditionalUnet1D (C2 共享 U-Net), CFM 与 DP 共用同一 backbone.

输入 / 输出形状: (B, H, action_dim).
forward(a_tau, t, obs):
    a_tau: (B, H, action_dim) 含噪动作 chunk
    t:     (B,) 归一化时间; CFM 直传 τ ∈ [0, 1], DP wrapper 内传 t/T ∈ [0, 1)
    obs:   (B, obs_dim) 全局条件
    返回   (B, H, action_dim); CFM 解释为 velocity, DP 解释为 noise.

与 DP 官方 conditional_unet1d.py 的两点差异:
1. condition = concat[sinusoidal(t) 经 time_mlp, obs]; obs 端的压维 (如有) 放在外部 obs encoder.
2. t 一律传归一化值; raw t / scale 由各自 wrapper 处理.

注意: encoder 推 3 个 skip, decoder 只 pop 2 个; 最浅 skip[0] 永不使用. 与 DP 一致.
"""

import torch
import torch.nn as nn

from src.model.modules import (
    Downsample1d,
    FiLMResBlock1D,
    Upsample1d,
    sinusoidal_embedding,
)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int = 2048,
        embedded_dim: int = 256,
        down_dims=(128, 256, 512),
        kernel_size: int = 5,
    ):
        super().__init__()
        self.embedded_dim = embedded_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(embedded_dim, embedded_dim * 4),
            nn.Mish(),
            nn.Linear(embedded_dim * 4, embedded_dim),
        )
        cond_dim = embedded_dim + obs_dim

        all_dims = [action_dim] + list(down_dims)           # [a_d, 128, 256, 512]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))     # [(a_d,128),(128,256),(256,512)]
        start_dim = down_dims[0]
        mid_dim = down_dims[-1]
        num_res_blocks = 2

        # encoder: 3 levels; 前 2 个有 Downsample1d, 最后一个不下采样 (DP 一致)
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(in_out):
            level = nn.ModuleList()
            level.append(FiLMResBlock1D(in_ch, out_ch, cond_dim, kernel_size))
            for _ in range(num_res_blocks - 1):
                level.append(FiLMResBlock1D(out_ch, out_ch, cond_dim, kernel_size))
            self.encoder.append(level)
            if i < len(in_out) - 1:
                self.downsamples.append(Downsample1d(out_ch))

        # bottleneck: 2 个 ResBlock at mid_dim
        self.bottleneck = nn.ModuleList([
            FiLMResBlock1D(mid_dim, mid_dim, cond_dim, kernel_size),
            FiLMResBlock1D(mid_dim, mid_dim, cond_dim, kernel_size),
        ])

        # decoder: 2 levels, 都是 real Upsample (skip[0] 不被使用, DP 一致).
        # 每个 level: cat -> res blocks -> upsample (注意顺序与用户 U_net.py 的 image 域不同)
        decoder_pairs = list(reversed(in_out[1:]))          # [(256,512), (128,256)]
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for skip_ch, deep_ch in decoder_pairs:
            level = nn.ModuleList()
            level.append(FiLMResBlock1D(deep_ch * 2, skip_ch, cond_dim, kernel_size))
            for _ in range(num_res_blocks - 1):
                level.append(FiLMResBlock1D(skip_ch, skip_ch, cond_dim, kernel_size))
            self.decoder.append(level)
            self.upsamples.append(Upsample1d(skip_ch))

        # final: Conv1d -> GroupNorm -> Mish -> Conv1d(start_dim -> action_dim, k=1)
        self.conv_out = nn.Conv1d(start_dim, start_dim, kernel_size, padding=kernel_size // 2)
        self.norm_out = nn.GroupNorm(8, start_dim)
        self.activate = nn.Mish()
        self.exit = nn.Conv1d(start_dim, action_dim, kernel_size=1)

    def forward(self, a_tau: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # a_tau: (B, H, action_dim); t: (B,); obs: (B, obs_dim)

        # 转到 Conv1d 期望的 (B, C, H)
        x = a_tau.permute(0, 2, 1)

        # 时间嵌入: sinusoidal -> MLP, 然后 concat obs 作为 condition
        if t.dim() == 0:
            t = t.unsqueeze(0)
        elif t.dim() > 1:
            t = t.view(-1)
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        t_emb = self.time_mlp(sinusoidal_embedding(self.embedded_dim, t))
        cond = torch.cat([t_emb, obs], dim=-1)

        # encoder
        skips = []
        for i, level in enumerate(self.encoder):
            for block in level:
                x = block(x, cond)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        # bottleneck
        for block in self.bottleneck:
            x = block(x, cond)

        # decoder
        for i, level in enumerate(self.decoder):
            x = torch.cat([x, skips.pop()], dim=1)
            for block in level:
                x = block(x, cond)
            x = self.upsamples[i](x)

        # final
        x = self.conv_out(x)
        x = self.norm_out(x)
        x = self.activate(x)
        x = self.exit(x)

        return x.permute(0, 2, 1)
