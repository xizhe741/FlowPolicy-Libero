# Spec: training_loss_pseudocode.md §dp.py
#   - squared_cosine_schedule: Claude 起草、用户审批
#   - dp_loss: 用户原始伪代码
# Pseudocode → code mapping:
#   squared_cosine_schedule(T, s)             → def squared_cosine_schedule(T=100, s=0.008)
#     steps = arange(T+1, fp64)               → torch.arange(T+1, dtype=torch.float64)
#     f = cos(((steps/T+s)/(1+s))*(π/2))**2   → torch.cos(...) ** 2
#     alpha_bar_raw = f / f[0]                → f / f[0]
#     β = (1 - ᾱ_raw[1:]/ᾱ_raw[:-1]).clamp    → (1.0 - alpha_bar_raw[1:] / alpha_bar_raw[:-1]).clamp(max=0.999)
#     α = 1 - β; ᾱ = cumprod(α)               → 1.0 - beta; torch.cumprod(alpha, dim=0)
#     return ᾱ.float()                        → return alpha_bar.float()
#
#   dp_loss signature                         → def dp_loss(model, a_clean, obs, alpha_bar)
#   B, T, device                              → a_clean.shape[0], alpha_bar.shape[0], a_clean.device
#   t_int = randint(0, T, (B,))               → torch.randint(0, T, (B,), device=device)
#   z = randn_like(a_clean)                   → torch.randn_like(a_clean)
#   ab = ᾱ[t_int].view(-1,1,1)                → alpha_bar[t_int].view(-1, 1, 1)
#   a_t = √ᾱ·a_clean + √(1-ᾱ)·z              → ab.sqrt() * a_clean + (1 - ab).sqrt() * z
#   t_norm = (t_int+1)/T                      → (t_int + 1).float() / T
#   eps_pred = model(a_t, t_norm, obs)        → model(a_t, t_norm, obs)
#   return F.mse_loss(z, eps_pred)            → F.mse_loss(z, eps_pred)

import math

import torch
import torch.nn.functional as F


def squared_cosine_schedule(T: int = 100, s: float = 0.008) -> torch.Tensor:
    """IDDPM squared cosine ᾱ schedule.

    Returns ᾱ of shape (T,).
    Index semantics: alpha_bar[i] 对应 physical timestep t = i + 1.
    Internal fp64 to avoid cumprod precision loss; returns fp32.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * (math.pi / 2)) ** 2
    alpha_bar_raw = f / f[0]  # length T+1

    # β from ᾱ ratio, clipped per DP-official (avoid β → 1 at large t)
    beta = (1.0 - alpha_bar_raw[1:] / alpha_bar_raw[:-1]).clamp(max=0.999)

    # Re-derive ᾱ from clipped β for self-consistency
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return alpha_bar.float()


def dp_loss(model, a_clean, obs, alpha_bar):
    """
    a_clean:   (B, H, d_a)  — clean action chunk (DDPM 记号下 x₀)
    obs:       condition的一部分经过obsencoder之后传给model
    alpha_bar: (T,) tensor from squared_cosine_schedule, 需提前 .to(device)
    model:     forward(a_t, t_norm, obs) -> eps_pred, t_norm shape (B,)
    """
    B, T, device = a_clean.shape[0], alpha_bar.shape[0], a_clean.device

    # t ~ U{1, ..., T}, shape (B,)
    t_int = torch.randint(0, T, (B,), device=device)
    # index semantics: alpha_bar[t_int] 对应 physical t = t_int + 1

    # noise
    z = torch.randn_like(a_clean)

    # forward q-process: a_t = √ᾱ_t · a_clean + √(1-ᾱ_t) · z
    ab = alpha_bar[t_int].view(-1, 1, 1)
    a_t = ab.sqrt() * a_clean + (1 - ab).sqrt() * z

    # normalize t to [1/T, 1] for model input
    t_norm = (t_int + 1).float() / T

    # ε-prediction
    eps_pred = model(a_t, t_norm, obs)

    persample_loss = ((eps_pred - z)**2).mean(dim=(1, 2)) #(B,)



    # MSE, mean over (B, H, d_a) → 0-dim scalar
    return F.mse_loss(z, eps_pred), \
        {"per_sample_loss": persample_loss, "tau": t_norm}


def ddim_sample(model, obs, alpha_bar, timesteps, H, d_a):
    """
    DDIM deterministic sampler (eta = 0).

    model:      ConditionalUnet1D, forward(a_t, t_norm, obs) -> eps_pred
                a_t: (B, H, d_a), t_norm: (B,), obs: (B, obs_dim)
    obs:        (B, 2048) ObsEncoder 输出 (与 dp_loss / cfm_loss / euler_sample 同命名)
    alpha_bar:  (T,) tensor, squared_cosine_schedule 输出
                索引 i 对应 physical timestep t = i + 1 (与 dp_loss L33, L67 对齐)
    timesteps:  1d 整数序列 (Python list 或 0d-int tensor 元素均可), 严格降序
                caller (infer.py) 决定生成策略:
                  (a) random subsample: torch.randperm(T)[:16].sort(descending=True)
                  (b) 全量: list(range(T - 1, -1, -1))
                长度 = T_infer
    H, d_a:     chunk horizon, action dim (caller 从 cfg.unet 取, 与 euler_sample 同 idiom)
    返回:        (B, H, d_a) action chunk in normalized [-1, 1] space
    """
    B = obs.shape[0]
    device = obs.device
    T = alpha_bar.shape[0]
    N_steps = len(timesteps)

    # x_T ~ N(0, I)
    x = torch.randn(B, H, d_a, device=device)

    for i in range(N_steps):
        t = int(timesteps[i])
        ab_t = alpha_bar[t]

        # alpha_bar_prev: 最后一步映射回 clean state, 用 1.0 sentinel (HuggingFace
        # DDIMScheduler 同款逻辑); 否则取 alpha_bar[timesteps[i + 1]]
        if i == N_steps - 1:
            ab_prev = torch.tensor(1.0, device=device)
        else:
            ab_prev = alpha_bar[int(timesteps[i + 1])]

        # t_norm 与 dp_loss L71 训练对齐: (t + 1) / T ∈ [1/T, 1]
        t_norm = torch.full((B,), (t + 1) / T, device=device)

        # ε-prediction
        eps = model(x, t_norm, obs)

        # DDIM deterministic step (eta = 0):
        #   x0_hat = (x - sqrt(1 - ᾱ_t) · eps) / sqrt(ᾱ_t)
        #   x_prev = sqrt(ᾱ_prev) · x0_hat + sqrt(1 - ᾱ_prev) · eps
        x0_hat = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
        x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1 - ab_prev) * eps

    # no clip; rely on denormalize() in data.py at eval time
    # (与 euler_sample 一致, 保留 raw range 用于 DP/CFM head replacement 诊断)
    return x
