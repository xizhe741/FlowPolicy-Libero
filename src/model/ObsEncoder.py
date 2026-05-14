"""
src/models.py

ObsEncoder: c2_plan §3.3 中 condition 路径的观测向量 $o$ 的生成器。
将 data.py 输出的 raw obs concat 张量 $\\tilde{o} \\in \\mathbb{R}^{B \\times 8210}$
映射到 $o = \\text{ObsEncoder}(\\tilde{o}) \\in \\mathbb{R}^{B \\times 2048}$，
作为 ConditionalUnet1D 的 global condition。

输入张量约定 ($\\tilde{o}$, 按 §3.3 raw concat 顺序):
    obs[:, 0:2048]      R3M(I^agent_{t-1})    cam_agent at t-1
    obs[:, 2048:4096]   R3M(I^wrist_{t-1})    cam_wrist at t-1
    obs[:, 4096:4103]   q_{t-1}                Panda joint position (7)
    obs[:, 4103:4105]   g_{t-1}                gripper width (2)
    obs[:, 4105:6153]   R3M(I^agent_t)        cam_agent at t
    obs[:, 6153:8201]   R3M(I^wrist_t)        cam_wrist at t
    obs[:, 8201:8208]   q_t                    (7)
    obs[:, 8208:8210]   g_t                    (2)

token 顺序: [cam1_{t-1}, cam2_{t-1}, q_{t-1}, g_{t-1}, cam1_t, cam2_t, q_t, g_t]
step_idx     = [0,0,0,0,1,1,1,1]   (0 ↔ t-1, 1 ↔ t)
modality_idx = [0,1,2,3,0,1,2,3]   (0=cam1, 1=cam2, 2=proprio, 3=gripper)

设计决策（notes/decisions.md 2026-05-13）:
- dim = 256, num_blocks = 3, num_heads = 4, mlp_ratio = 4
- per-modality 投影: cam1 / cam2 独立, prev / curr 跨步共享权重
- learnable step_emb 与 modality_emb（默认 nn.Embedding 初始化 $\\mathcal{N}(0,1)$）
- PreNormBlock: LayerNorm + MHA(batch_first=True) 残差, LayerNorm + (Linear-SiLU-Linear) 残差
- 末端 final LayerNorm
- 无 dropout, 无 mask
- 所有 Linear 默认 bias=True, 默认 PyTorch 初始化
"""

import torch
import torch.nn as nn


# spec § "PreNormBlock: LayerNorm + MHA(batch_first=True) 残差, LayerNorm + (Linear-SiLU-Linear) 残差"
# → PreNormBlock 类
class PreNormBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        # spec § "LayerNorm + MHA"
        self.norm1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # spec § "LayerNorm + (Linear-SiLU-Linear)", hidden = mlp_ratio * dim
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.SiLU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # spec § "pre-norm 残差": LN → MHA → 残差, LN → FFN → 残差
        h = self.norm1(x)
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class ObsEncoder(nn.Module):
    def __init__(
        self,
        num_blocks: int = 3,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dim: int = 256,
    ):
        super().__init__()
        self.dim = dim

        # spec § "per-modality 投影: cam1 / cam2 独立, prev / curr 跨步共享权重"
        # → 4 个 Linear; prev / curr 在 forward 中复用同一组权重
        self.proj_cam1 = nn.Linear(2048, dim)
        self.proj_cam2 = nn.Linear(2048, dim)
        self.proj_proprio = nn.Linear(7, dim)
        self.proj_gripper = nn.Linear(2, dim)

        # spec § "learnable step_emb 与 modality_emb（默认 nn.Embedding 初始化）"
        self.step_emb = nn.Embedding(2, dim)      # 0 ↔ t-1, 1 ↔ t
        self.modality_emb = nn.Embedding(4, dim)  # 0=cam1, 1=cam2, 2=proprio, 3=gripper

        # spec § "token 顺序" 与 "step_idx / modality_idx"
        self.register_buffer(
            "step_idx",
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        )
        self.register_buffer(
            "modality_idx",
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),
        )

        # spec § "num_blocks = 3" + "末端 final LayerNorm"
        self.blocks = nn.ModuleList(
            [PreNormBlock(dim, num_heads, mlp_ratio) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        # spec § "输入张量约定" → 按 8210 维 raw concat 索引切分
        B = o.shape[0]
        O1_prev = o[:, 0:2048]
        O2_prev = o[:, 2048:4096]
        q_prev = o[:, 4096:4103]
        g_prev = o[:, 4103:4105]
        O1 = o[:, 4105:6153]
        O2 = o[:, 6153:8201]
        q = o[:, 8201:8208]
        g = o[:, 8208:8210]

        # spec § "token 顺序: [cam1_{t-1}, cam2_{t-1}, q_{t-1}, g_{t-1}, cam1_t, cam2_t, q_t, g_t]"
        # spec § "prev / curr 跨步共享权重" → 复用 self.proj_*
        tokens = torch.stack(
            [
                self.proj_cam1(O1_prev),
                self.proj_cam2(O2_prev),
                self.proj_proprio(q_prev),
                self.proj_gripper(g_prev),
                self.proj_cam1(O1),
                self.proj_cam2(O2),
                self.proj_proprio(q),
                self.proj_gripper(g),
            ],
            dim=1,
        )  # (B, 8, dim)

        # spec § "learnable step_emb 与 modality_emb" → 加到 token; (8, dim) broadcast 到 (B, 8, dim)
        x = (
            tokens
            + self.step_emb(self.step_idx).unsqueeze(0)
            + self.modality_emb(self.modality_idx).unsqueeze(0)
        )

        # spec § "无 mask, 无 dropout" → 直接堆叠 PreNormBlock + final LayerNorm
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # spec § "$o \\in \\mathbb{R}^{B \\times 2048}$" → 8 tokens × 256 dim flatten
        return x.reshape(B, -1)
