"""
src/train.py — CFM / DP 训练入口。

1:1 转译自 pseudocode v3 (2026-05-14). 章节对照通过 `# === pseudocode § XXX ===` 标注.
高价值代码：训练主循环 loss → backward → grad clip → EMA → 早停 state machine.
外包代码：CLI 解析、yaml cfg loader、optimizer / scheduler 实例化、checkpoint I/O、wandb 接线.
"""

import argparse
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import wandb
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.cfm import cfm_loss
from src.data import make_dataloader
from src.dp import dp_loss, squared_cosine_schedule
from src.eval import quick_eval
from src.model.ObsEncoder import ObsEncoder
from src.model.unet1d import ConditionalUnet1D


# === pseudocode § CLI ===
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--task_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def _dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    return d


def load_cfg(args):
    # yaml 加载 + CLI override 合并 (pseudocode §"完全外包，yaml schema 由 executor 设计")
    # cfg 必含字段 (pseudocode §"cfg 必含字段（最小集）"):
    #   method, task_name, seed, device, output_dir,
    #   hdf5_path, cache_dir, batch_size, num_workers,
    #   unet.{action_dim, obs_dim, embedded_dim, down_dims, kernel_size},
    #   obs_encoder.{num_blocks, num_heads, mlp_ratio, dim}
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    for k in ("task_name", "seed", "device", "output_dir"):
        v = getattr(args, k)
        if v is not None:
            raw[k] = v
    cfg = _dict_to_ns(raw)
    # pseudocode §"约束：unet.obs_dim == 8 * obs_encoder.dim"
    assert cfg.unet.obs_dim == 8 * cfg.obs_encoder.dim, (
        f"unet.obs_dim={cfg.unet.obs_dim} 与 8 * obs_encoder.dim={8 * cfg.obs_encoder.dim} 不一致"
    )
    return cfg


# === pseudocode § Optimizer / Scheduler ===
# "warmup=500 step 线性 0→lr，之后 cosine 退到 0"; executor 选 LambdaLR 实现.
def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# === pseudocode § Helpers: ema_update ===
def ema_update(shadow_dict, online_module, decay):
    online_state = online_module.state_dict()
    for k, shadow_val in shadow_dict.items():
        online_val = online_state[k].detach().float()
        shadow_val.mul_(decay).add_(online_val, alpha=1.0 - decay)


# === pseudocode § Helpers: quick_eval_with_ema ===
def quick_eval_with_ema(model, obs_encoder, ema_shadow_model, ema_shadow_obs_encoder,
                        task_name, seed, n_episodes):
    online_state = {k: v.clone() for k, v in model.state_dict().items()
                    if k in ema_shadow_model}
    online_obs_state = {k: v.clone() for k, v in obs_encoder.state_dict().items()
                        if k in ema_shadow_obs_encoder}
    try:
        model.load_state_dict(ema_shadow_model, strict=False)
        obs_encoder.load_state_dict(ema_shadow_obs_encoder, strict=False)
        model.eval()
        obs_encoder.eval()
        result = quick_eval(model, obs_encoder, task_name, seed, n_episodes)
    finally:
        model.load_state_dict(online_state, strict=False)
        obs_encoder.load_state_dict(online_obs_state, strict=False)
        model.train()
        obs_encoder.train()
    return result


# === pseudocode § Helpers: build_last_state ===
def build_last_state(model, obs_encoder, ema_shadow_model, ema_shadow_obs_encoder,
                     normalizer_state, optimizer, scheduler, epoch, global_step,
                     loss_ema, loss_ema_history, phase, best_sr, stale_count,
                     last_eval_epoch, cfg):
    return {
        "model_state_dict": model.state_dict(),
        "obs_encoder_state_dict": obs_encoder.state_dict(),
        "ema_shadow_model": ema_shadow_model,
        "ema_shadow_obs_encoder": ema_shadow_obs_encoder,
        "normalizer_state": normalizer_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "loss_ema": loss_ema,
        "loss_ema_history": loss_ema_history,
        "phase": phase,
        "best_sr": best_sr,
        "stale_count": stale_count,
        "last_eval_epoch": last_eval_epoch,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cfg": cfg.__dict__,
        "method": cfg.method,
        "task_name": cfg.task_name,
        "seed": cfg.seed,
    }


# === pseudocode § Helpers: build_best_state ===
def build_best_state(ema_shadow_model, ema_shadow_obs_encoder,
                     normalizer_state, best_sr, epoch, global_step, cfg):
    return {
        "ema_shadow_model": ema_shadow_model,
        "ema_shadow_obs_encoder": ema_shadow_obs_encoder,
        "normalizer_state": normalizer_state,
        "best_sr": best_sr,
        "epoch_saved": epoch,
        "global_step_saved": global_step,
        "cfg": cfg.__dict__,
        "method": cfg.method,
        "task_name": cfg.task_name,
        "seed": cfg.seed,
    }


def main():
    args = parse_args()
    cfg = load_cfg(args)

    # === pseudocode § Reproducibility ===
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === pseudocode § Instantiation === (master fp32; 不调 .to(bfloat16))
    model = ConditionalUnet1D(
        action_dim=cfg.unet.action_dim,
        obs_dim=cfg.unet.obs_dim,
        embedded_dim=cfg.unet.embedded_dim,
        down_dims=cfg.unet.down_dims,
        kernel_size=cfg.unet.kernel_size,
    ).to(device)
    obs_encoder = ObsEncoder(
        num_blocks=cfg.obs_encoder.num_blocks,
        num_heads=cfg.obs_encoder.num_heads,
        mlp_ratio=cfg.obs_encoder.mlp_ratio,
        dim=cfg.obs_encoder.dim,
    ).to(device)

    if cfg.method == "cfm":
        loss_fn = lambda m, a, o: cfm_loss(m, a, o)
    else:
        alpha_bar = squared_cosine_schedule(T=100).to(device)
        loss_fn = lambda m, a, o: dp_loss(m, a, o, alpha_bar)

    dataloader = make_dataloader(
        cfg.task_name, cfg.hdf5_path, cfg.cache_dir,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )
    # pseudocode §"normalizer 在 dataset 上，state_dict() 提取一次（统计量训练全程不变）"
    normalizer_state = dataloader.dataset.normalizer.state_dict()

    # === pseudocode § Optimizer / Scheduler ===
    params = list(model.parameters()) + list(obs_encoder.parameters())
    optimizer = AdamW(params, lr=1e-4, weight_decay=1e-6)
    total_steps = 200 * len(dataloader)
    scheduler = cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=total_steps,
    )

    # === pseudocode § EMA shadow (fp32; 覆盖 model + obs_encoder) ===
    ema_shadow_model = {
        k: v.detach().clone().float()
        for k, v in model.state_dict().items()
        if v.dtype.is_floating_point
    }
    ema_shadow_obs_encoder = {
        k: v.detach().clone().float()
        for k, v in obs_encoder.state_dict().items()
        if v.dtype.is_floating_point
    }

    EMA_DECAY = 0.9997           # pseudocode row 32 (2026-05-14)
    WARMUP_STEPS = 500
    LOG_INTERVAL = 6             # pseudocode row 36 (用户自设计)
    PLATEAU_THRESHOLD = 0.02     # pseudocode row 25 (2026-05-14)

    # === pseudocode § wandb ===
    wandb.init(
        project="flow-policy",
        name=f"{cfg.method}_{cfg.task_name}_seed{cfg.seed}",
        config=cfg.__dict__,
    )

    # === pseudocode § Early-stop state ===
    loss_ema = None
    loss_ema_history = []
    phase = 1
    best_sr = -1.0
    stale_count = 0
    last_eval_epoch = None
    global_step = 0
    last_idx = 0

    # === pseudocode § Main loop ===
    for epoch in range(200):
        epoch_loss_sum = 0.0
        model.train()
        obs_encoder.train()

        for obs, action, _, _ in dataloader:
            obs = obs.to(device)
            action = action.to(device)

            # autocast 仅包 forward + loss; params fp32 storage 不变
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                obs_emb = obs_encoder(obs)
                loss, aux = loss_fn(model, action, obs_emb)
            loss = loss.float()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()                                  # fp32 梯度
            grad_norm = clip_grad_norm_(params, 1.0)

            if global_step % LOG_INTERVAL == 0:
                params_before = torch.cat([p.detach().flatten() for p in params])

            optimizer.step()                                  # fp32 AdamW
            scheduler.step()                                  # per-step

            if global_step >= WARMUP_STEPS:
                ema_update(ema_shadow_model, model, EMA_DECAY)
                ema_update(ema_shadow_obs_encoder, obs_encoder, EMA_DECAY)

            # === pseudocode § per-LOG_INTERVAL logging ===
            if global_step % LOG_INTERVAL == 0:
                params_after = torch.cat([p.detach().flatten() for p in params])
                step_norm = (params_after - params_before).norm().item()
                shadow_norm = (
                    sum(v.norm().item() ** 2 for v in ema_shadow_model.values())
                    + sum(v.norm().item() ** 2 for v in ema_shadow_obs_encoder.values())
                ) ** 0.5

                log_dict = {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm.item(),
                    "train/optim_step_norm": step_norm,
                    "train/ema_shadow_norm": shadow_norm,
                    "train/action_norm_distribution": wandb.Histogram(
                        action.detach().float().cpu().numpy().flatten()
                    ),
                }

                tau = aux["tau"]
                per_sample_loss = aux["per_sample_loss"]
                buckets = 10
                bucket_idx = (tau * buckets).long().clamp(0, buckets - 1)
                for b in range(buckets):
                    mask = bucket_idx == b
                    if mask.any():
                        log_dict[f"train/loss_tau_bin_{b}"] = per_sample_loss[mask].mean().item()

                wandb.log(log_dict, step=global_step)

            global_step += 1
            epoch_loss_sum += loss.item()

        # === pseudocode § Epoch end ===
        epoch_loss_mean = epoch_loss_sum / len(dataloader)
        loss_ema = epoch_loss_mean if loss_ema is None else 0.95 * loss_ema + 0.05 * epoch_loss_mean
        loss_ema_history.append(loss_ema)

        wandb.log({
            "train/loss_epoch_mean": epoch_loss_mean,
            "train/loss_ema": loss_ema,
            "train/phase": phase,
        }, step=global_step)

        # === pseudocode § Save last (rolling 2，原子 rename) ===
        last_path = output_dir / f"last_{last_idx}.pt"
        last_tmp = output_dir / f"last_{last_idx}.pt.tmp"
        torch.save(build_last_state(
            model, obs_encoder, ema_shadow_model, ema_shadow_obs_encoder,
            normalizer_state, optimizer, scheduler, epoch, global_step,
            loss_ema, loss_ema_history, phase, best_sr, stale_count,
            last_eval_epoch, cfg,
        ), last_tmp)
        os.replace(last_tmp, last_path)
        last_idx = 1 - last_idx

        # === pseudocode § Phase 1: plateau detection ===
        if phase == 1 and epoch >= 20:
            denom = loss_ema_history[epoch - 20]
            relative_drop = (denom - loss_ema) / denom
            if relative_drop < PLATEAU_THRESHOLD:
                phase = 2
                last_eval_epoch = epoch - 20  # 切换当 epoch 立即触发首次 quick eval

        # === pseudocode § Phase 2: quick eval + early stop ===
        if phase == 2 and (epoch - last_eval_epoch) >= 20:
            result = quick_eval_with_ema(
                model, obs_encoder,
                ema_shadow_model, ema_shadow_obs_encoder,
                task_name=cfg.task_name, seed=cfg.seed, n_episodes=20,
            )
            sr = result["sr"]
            last_eval_epoch = epoch

            succ_lengths = [l for l, s in zip(result["episode_lengths"], result["successes"]) if s]
            wandb.log({
                "eval/quick_sr": sr,
                "eval/best_sr": best_sr,
                "eval/stale_count": stale_count,
                "eval/episode_length_mean": float(np.mean(succ_lengths)) if succ_lengths else 0.0,
                "eval/sr_per_episode": wandb.Histogram(
                    np.array([float(s) for s in result["successes"]])
                ),
            }, step=global_step)

            if sr >= best_sr:                # pseudocode §"`>=` 算上升（决策项 2）"
                best_sr = sr
                stale_count = 0
                best_tmp = output_dir / "best.pt.tmp"
                best_path = output_dir / "best.pt"
                torch.save(build_best_state(
                    ema_shadow_model, ema_shadow_obs_encoder,
                    normalizer_state, best_sr, epoch, global_step, cfg,
                ), best_tmp)
                os.replace(best_tmp, best_path)
            else:
                stale_count += 1

            if stale_count >= 2:
                break

    # === pseudocode § Final fallback eval (break / epoch=200 两种退出后都跑) ===
    result = quick_eval_with_ema(
        model, obs_encoder,
        ema_shadow_model, ema_shadow_obs_encoder,
        task_name=cfg.task_name, seed=cfg.seed, n_episodes=20,
    )
    final_sr = result["sr"]
    wandb.log({"eval/final_sr": final_sr}, step=global_step)
    if final_sr >= best_sr:
        best_sr = final_sr
        best_tmp = output_dir / "best.pt.tmp"
        best_path = output_dir / "best.pt"
        torch.save(build_best_state(
            ema_shadow_model, ema_shadow_obs_encoder,
            normalizer_state, best_sr, epoch, global_step, cfg,
        ), best_tmp)
        os.replace(best_tmp, best_path)

    wandb.finish()


if __name__ == "__main__":
    main()
