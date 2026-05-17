# Spec: training_loss_pseudocode.md §cfm.py (user pseudocode, reviewer-approved)
# Pseudocode → code mapping:
#   header                       → import block
#   cfm_loss signature           → def cfm_loss(model, a_clean, obs)
#   B, device line               → B, device = a_clean.shape[0], a_clean.device
#   t = torch.rand(B, ...)       → t = torch.rand(B, device=device)
#   t_bc = t.view(-1, 1, 1)      → t_bc = t.view(-1, 1, 1)
#   z = torch.randn_like(...)    → z = torch.randn_like(a_clean)
#   a_t = z*(1-t_bc) + t_bc*a₁   → a_t = z * (1 - t_bc) + t_bc * a_clean
#   v_target = a₁ - a₀           → v_target = a_clean - z
#   v_pred = model(a_t, t, obs)  → v_pred = model(a_t, t, obs)
#   return F.mse_loss(...)       → return F.mse_loss(v_pred, v_target)

import torch
import torch.nn.functional as F


def cfm_loss(model, a_clean, obs):
    """
    a_clean: (B, H, d_a)  — clean action chunk (CFM 记号下 a₁)
    obs:     经过obsencoder之后传给model
    model:   forward(a_t, t, obs) -> v_pred, t shape (B,)
    """
    B, device = a_clean.shape[0], a_clean.device

    # τ ~ U[0, 1), shape (B,)
    t = torch.rand(B, device=device)

    # broadcast to (B, 1, 1) for element-wise ops with (B, H, d_a)
    t_bc = t.view(-1, 1, 1)

    # a₀ ~ N(0, I), same shape as a_clean
    z = torch.randn_like(a_clean)

    # linear-OT interpolation: a_τ = (1-τ) a₀ + τ a₁
    a_t = z * (1 - t_bc) + t_bc * a_clean

    # target velocity: v* = a₁ - a₀ (constant, does not depend on τ)
    v_target = a_clean - z

    # model prediction
    v_pred = model(a_t, t, obs)

    persample_loss =(( v_pred -v_target)**2 ).mean(dim=(1, 2))

  # MSE, mean over (B, H, d_a) → 0-dim scalar
    return F.mse_loss(v_target, v_pred), \
        {"per_sample_loss": persample_loss, "tau": t}


# === euler_sample (append) ===
# Pseudocode → code mapping:
#   def euler_sample(...) header  → def euler_sample(model, obs, H, d_a, N=4)
#   docstring block               → preserved verbatim
#   B = obs.shape[0]              → B = obs.shape[0]
#   device = obs.device           → device = obs.device
#   x = torch.randn(B,H,d_a,...)  → x = torch.randn(B, H, d_a, device=device)
#   dt = 1.0 / N                  → dt = 1.0 / N
#   for n in range(N):            → for n in range(N):
#   tau = torch.full((B,), ...)   → tau = torch.full((B,), n * dt, device=device)
#   v = model(x, tau, obs)        → v = model(x, tau, obs)
#   x = x + dt * v                → x = x + dt * v
#   no clip / ODE ablation cmts   → preserved verbatim above return
#   return x                      → return x

def euler_sample(model, obs, H, d_a, N=4):
    """
    CFM Euler ODE sampler.

    model: ConditionalUnet1D, forward(a_tau, t, obs) -> v_pred
           a_tau: (B, H, d_a), t: (B,), obs: (B, obs_dim)
    obs:   (B, 2048) ObsEncoder 输出 (与 cfm_loss 同命名)
    H:     chunk horizon (caller 从 cfg.unet 或 cfg.data 传入)
    d_a:   action dim (caller 从 cfg.unet.action_dim 传入)
    N:     Euler 步数, 默认 4
    返回:   (B, H, d_a) action chunk in normalized [-1, 1] space
    """
    B = obs.shape[0]
    device = obs.device

    # a_0 ~ N(0, I)
    x = torch.randn(B, H, d_a, device=device)
    dt = 1.0 / N

    for n in range(N):
        # tau 必须是 (B,) tensor — model 主路径要求 (B,) 向量,
        # unet1d.forward L97-102 的 dim 分支仅做边界 fallback
        tau = torch.full((B,), n * dt, device=device)
        v = model(x, tau, obs)
        x = x + dt * v

    # no clip; rely on denormalize() in data.py at eval time
    # 保留 raw range 作为 ODE ablation (N=1 vs N=4) 数值误差诊断信号
    return x
