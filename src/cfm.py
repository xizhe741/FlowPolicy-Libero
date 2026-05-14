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
