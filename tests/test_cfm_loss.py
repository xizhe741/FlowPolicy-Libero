# Spec: reviewer-approved test skeleton (2026-05-17 对话)
# Pseudo → code 对照（按 src/cfm.py L18-48 表达式逐行配对）:
#   constants B=4 / H=16 / D=7 / OBS=2048       → 模块顶 4 个常量
#   Recorder(nn.Module) 捕获 (a_t, t, obs)      → class Recorder
#   pytest fixture batch → (a_clean, obs)       → @pytest.fixture batch
#
#   test_tau_shape_and_range                    → cfm.py L27 t = torch.rand(B,...)
#   test_model_signature_shapes                 → cfm.py L42 model(a_t, t, obs)
#   test_per_sample_shape_and_scalar_consistency→ cfm.py L44, L47
#   test_a_t_and_loss_at_tau_zero (monkeypatch) → cfm.py L36 端点 τ=0
#   test_a_t_at_tau_one (monkeypatch)           → cfm.py L36 端点 τ=1
#   test_v_target_sign (monkeypatch)            → cfm.py L39 v_target = a_clean - z
#   test_gradient_flows                         → cfm.py L42 model 入图
#   test_aux_dict_keys                          → cfm.py L47-48 return dict
#   test_device_propagation (skipif no cuda)    → cfm.py L27 device=device

import pytest
import torch

import src.cfm as cfm_mod
from src.cfm import cfm_loss


B, H, D, OBS = 4, 16, 7, 2048


class Recorder(torch.nn.Module):
    def __init__(self, fn=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.captured = {}
        self.fn = fn or (lambda at, t, o: torch.zeros_like(at) + self.w)

    def forward(self, a_t, t, obs):
        self.captured = dict(a_t=a_t.clone(), t=t.clone(), obs=obs.clone())
        return self.fn(a_t, t, obs)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(B, H, D), torch.randn(B, OBS)


def test_tau_shape_and_range(batch):
    a, o = batch
    _, aux = cfm_loss(Recorder(), a, o)
    assert aux["tau"].shape == (B,)
    assert (aux["tau"] >= 0).all() and (aux["tau"] < 1).all()


def test_model_signature_shapes(batch):
    a, o = batch
    m = Recorder()
    cfm_loss(m, a, o)
    assert m.captured["a_t"].shape == (B, H, D)
    assert m.captured["t"].shape == (B,)
    assert m.captured["obs"].shape == (B, OBS)


def test_per_sample_shape_and_scalar_consistency(batch):
    a, o = batch
    m = Recorder(fn=lambda at, t, obs: torch.randn_like(at))
    loss, aux = cfm_loss(m, a, o)
    assert aux["per_sample_loss"].shape == (B,)
    assert torch.allclose(loss, aux["per_sample_loss"].mean(), atol=1e-6)


def test_a_t_and_loss_at_tau_zero(monkeypatch, batch):
    a, o = batch
    z_fixed = torch.randn_like(a)
    monkeypatch.setattr(cfm_mod.torch, "randn_like", lambda x: z_fixed)
    monkeypatch.setattr(cfm_mod.torch, "rand", lambda *args, **kw: torch.zeros(B))
    m = Recorder()
    loss, _ = cfm_loss(m, a, o)
    assert torch.allclose(m.captured["a_t"], z_fixed)
    assert torch.allclose(loss, ((a - z_fixed) ** 2).mean(), atol=1e-6)


def test_a_t_at_tau_one(monkeypatch, batch):
    a, o = batch
    z_fixed = torch.randn_like(a)
    monkeypatch.setattr(cfm_mod.torch, "randn_like", lambda x: z_fixed)
    monkeypatch.setattr(
        cfm_mod.torch, "rand",
        lambda *args, **kw: torch.ones(B) - 1e-7,
    )
    m = Recorder()
    cfm_loss(m, a, o)
    assert torch.allclose(m.captured["a_t"], a, atol=1e-5)


def test_v_target_sign(monkeypatch, batch):
    a, o = batch
    z_fixed = torch.randn_like(a)
    monkeypatch.setattr(cfm_mod.torch, "randn_like", lambda x: z_fixed)
    monkeypatch.setattr(
        cfm_mod.torch, "rand",
        lambda *args, **kw: torch.full((B,), 0.3),
    )
    m = Recorder(fn=lambda at, t, obs: a - z_fixed)
    loss, _ = cfm_loss(m, a, o)
    assert loss.item() < 1e-10


def test_gradient_flows(batch):
    a, o = batch
    m = Recorder()
    loss, _ = cfm_loss(m, a, o)
    loss.backward()
    assert m.w.grad is not None and m.w.grad.abs().sum() > 0


def test_aux_dict_keys(batch):
    a, o = batch
    _, aux = cfm_loss(Recorder(), a, o)
    assert set(aux.keys()) == {"per_sample_loss", "tau"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_device_propagation():
    a = torch.randn(B, H, D, device="cuda")
    o = torch.randn(B, OBS, device="cuda")
    _, aux = cfm_loss(Recorder().cuda(), a, o)
    assert aux["tau"].device.type == "cuda"
