"""
src/train.py — CFM / DP 训练入口。

1:1 转译自 pseudocode v3 (2026-05-14). 章节对照通过 `# === pseudocode § XXX ===` 标注.
高价值代码：训练主循环 loss → backward → grad clip → EMA → 早停 state machine.
外包代码：CLI 解析、yaml cfg loader、optimizer / scheduler 实例化、checkpoint I/O、wandb 接线.
"""

import argparse
import copy
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

from src.data import make_dataloader
from src.eval import evaluate
from src.model.ObsEncoder import ObsEncoder
from src.model.unet1d import ConditionalUnet1D

from libero.libero import benchmark as libero_benchmark
from libero.libero.envs import OffScreenRenderEnv
from r3m import load_r3m

# torch 2.6+ 把 torch.load weights_only 默认值改成 True, 与 LIBERO 内部
# get_task_init_states 加载 numpy 数组 / 我们自己 ckpt 嵌 numpy 与 python rng state
# 均不兼容. 这里恢复旧默认; 仅加载受信本地文件, 不接收外部 ckpt.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat


# === pseudocode § CLI ===
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--task_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    # resume: 自动选 DIR/last_{0,1}.pt 中 epoch 最大的;
    # cfg 取 ckpt 内 cfg, 仅 scheduler.epoch_max / early_stop.* 可被 --config override
    p.add_argument("--resume_dir", type=str, default=None)
    return p.parse_args()


def _dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    return d


def _ns_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _ns_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ns_to_dict(v) for v in obj]
    return obj


def load_cfg(args):
    # yaml 加载 + CLI override 合并 (pseudocode §"完全外包，yaml schema 由 executor 设计")
    # cfg 必含字段（按 train.py 实际依赖枚举，wandb.* 有 fallback 可省略）:
    #   顶层:
    #     method, task_name, seed, device, output_dir,
    #     hdf5_path (或 hdf5_dir，二选一), cache_dir, batch_size, num_workers
    #   unet.{action_dim, obs_dim, embedded_dim, down_dims, kernel_size}
    #   obs_encoder.{num_blocks, num_heads, mlp_ratio, dim}
    #   data.{chunk_horizon}
    #   infer.{N, T_infer, action_steps}
    #   dp.{T}                     # 仅 method == "dp" 必需
    #   libero.{camera_height, camera_width}
    #   optimizer.{lr, weight_decay}
    #   scheduler.{epoch_max, warmup_steps}
    #   train.{grad_clip_norm, log_interval, tau_buckets}
    #   ema.{decay}
    #   loss_ema.{alpha}
    #   early_stop.{phase1_min_epoch, plateau_window, plateau_threshold,
    #               phase2_eval_interval, eval_episodes, patience}
    #   eval.{max_steps}
    #   wandb.{project, run_name, mode}   # 缺省由本函数 fallback 填充
    # 约束: unet.obs_dim == 8 * obs_encoder.dim
    #
    # resume 模式 (--resume_dir DIR): cfg 取 ckpt 内 cfg, 仅 scheduler.epoch_max 与
    # early_stop.* 子树允许从 --config override; task_name / seed / device CLI 给冲突值则 raise.
    # 返回 (cfg, resume_ckpt or None, last_name or None).
    with open(args.config, "r") as f:
        yaml_raw = yaml.safe_load(f)

    if args.resume_dir is not None:
        resume_dir = Path(args.resume_dir)
        candidates = []
        for name in ("last_0.pt", "last_1.pt"):
            p = resume_dir / name
            if p.exists():
                ck = torch.load(p, map_location="cpu")
                candidates.append((ck["epoch"], name, ck))
        if not candidates:
            raise RuntimeError(
                f"--resume_dir {resume_dir}: 缺 last_0.pt / last_1.pt"
            )
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, last_name, resume_ckpt = candidates[0]

        raw = _ns_to_dict(resume_ckpt["cfg"])
        # 仅允许 epoch_max / early_stop 从 yaml override
        if "scheduler" in yaml_raw and "epoch_max" in yaml_raw["scheduler"]:
            raw["scheduler"]["epoch_max"] = yaml_raw["scheduler"]["epoch_max"]
        if "early_stop" in yaml_raw:
            raw.setdefault("early_stop", {}).update(yaml_raw["early_stop"])
        # output_dir 强制等于 resume_dir
        raw["output_dir"] = str(resume_dir)
        # task_name / seed / device: CLI 给值则必须与 ckpt 一致, 否则 raise
        for k in ("task_name", "seed", "device"):
            v = getattr(args, k)
            if v is not None and v != raw.get(k):
                raise RuntimeError(
                    f"resume 模式禁用 --{k} override: ckpt={raw.get(k)} CLI={v}"
                )

        cfg = _dict_to_ns(raw)
        assert cfg.unet.obs_dim == 8 * cfg.obs_encoder.dim, (
            f"unet.obs_dim={cfg.unet.obs_dim} 与 8 * obs_encoder.dim={8 * cfg.obs_encoder.dim} 不一致"
        )
        return cfg, resume_ckpt, last_name

    raw = yaml_raw
    for k in ("task_name", "seed", "device", "output_dir"):
        v = getattr(args, k)
        if v is not None:
            raw[k] = v

    # hdf5_path 合成: 优先 yaml 显式 hdf5_path; 否则 hdf5_dir + task_name + _demo 后缀
    # (task_name 取 LIBERO 原生名; _demo 是 demonstration 数据集的文件命名约定, 在此注入).
    if raw.get("hdf5_path") is None:
        raw["hdf5_path"] = f"{raw['hdf5_dir']}/{raw['task_name']}_demo.hdf5"

    # spec 改 1: wandb 段 fallback
    raw.setdefault("wandb", {})
    raw["wandb"].setdefault("project", "flow-policy")
    raw["wandb"].setdefault("mode", "online")
    if raw["wandb"].get("run_name") is None:
        raw["wandb"]["run_name"] = f"{raw['method']}_{raw['task_name']}_seed{raw['seed']}"
    cfg = _dict_to_ns(raw)
    # pseudocode §"约束：unet.obs_dim == 8 * obs_encoder.dim"
    assert cfg.unet.obs_dim == 8 * cfg.obs_encoder.dim, (
        f"unet.obs_dim={cfg.unet.obs_dim} 与 8 * obs_encoder.dim={8 * cfg.obs_encoder.dim} 不一致"
    )
    return cfg, None, None


# === pseudocode § Optimizer / Scheduler ===
# "warmup=500 step 线性 0→lr，之后 cosine 退到 0"; executor 选 LambdaLR 实现.
def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# === pseudocode § Helpers: ema_update ===
def ema_update(shadow_dict, online_module, decay):
    online_state = online_module.state_dict()
    for k, shadow_val in shadow_dict.items():
        online_val = online_state[k].detach().float()
        shadow_val.mul_(decay).add_(online_val, alpha=1.0 - decay)


# === pseudocode § Helpers: evaluate_with_ema ===
def evaluate_with_ema(
    model, obs_encoder,
    ema_shadow_model, ema_shadow_obs_encoder,
    env, init_states_all, infer_fn, normalizer, r3m_model, device,
    n_episodes, seed_for_eval, max_steps, action_steps,
):
    """EMA shadow 加载 → 从 init_states_all 中抽样 n_episodes 条 init_state → 调 evaluate → 恢复 online 权重.

    抽样职责划分:
        上游 caller (train.py main() 内 phase 2 quick eval 与 final fallback eval 两处)
        负责传入完整 init_states_all (len=N_all=50) 与 seed_for_eval.
        本函数 evaluate_with_ema 负责用 np.random.RandomState(seed_for_eval).choice
        从 init_states_all 中无放回抽取 n_episodes 条 init_state, 再交给 src.eval.evaluate
        (其形参 init_states_for_episodes 假定 caller 侧 (即本函数) 已完成抽样).
    """
    online_state = {k: v.clone() for k, v in model.state_dict().items()
                    if k in ema_shadow_model}
    online_obs_state = {k: v.clone() for k, v in obs_encoder.state_dict().items()
                        if k in ema_shadow_obs_encoder}
    try:
        model.load_state_dict(ema_shadow_model, strict=False)
        obs_encoder.load_state_dict(ema_shadow_obs_encoder, strict=False)
        model.eval()
        obs_encoder.eval()

        idx = np.random.RandomState(seed_for_eval).choice(
            len(init_states_all), n_episodes, replace=False
        )
        init_states_for_episodes = init_states_all[idx]
        result = evaluate(
            model, obs_encoder, env, init_states_for_episodes,
            infer_fn, normalizer, r3m_model, device,
            collect_failure_videos=False,
            max_steps=max_steps, action_steps=action_steps,
        )
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
                     last_eval_epoch, cfg, last_idx, wandb_run_id):
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
        "last_idx": last_idx,
        "wandb_run_id": wandb_run_id,
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
    cfg, resume_ckpt, resume_last_name = load_cfg(args)

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

    # === Resume: model + obs_encoder weights ===
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        obs_encoder.load_state_dict(resume_ckpt["obs_encoder_state_dict"])

    H, d_a = cfg.data.chunk_horizon, cfg.unet.action_dim
    if cfg.method == "cfm":
        from src.cfm import cfm_loss, euler_sample
        loss_fn = lambda m, a, o: cfm_loss(m, a, o)
        infer_fn = lambda m, o: euler_sample(m, o, H, d_a, N=cfg.infer.N)
    else:
        from src.dp import dp_loss, ddim_sample, squared_cosine_schedule
        alpha_bar = squared_cosine_schedule(T=cfg.dp.T).to(device)
        T, T_infer = alpha_bar.shape[0], cfg.infer.T_infer
        # HuggingFace DDIMScheduler leading spacing, steps_offset=0 (decisions.md 2026-05-17 row 43)
        step_ratio = T // T_infer
        timesteps = (torch.arange(0, T_infer) * step_ratio).flip(0).tolist()
        loss_fn = lambda m, a, o: dp_loss(m, a, o, alpha_bar)
        infer_fn = lambda m, o: ddim_sample(m, o, alpha_bar, timesteps, H, d_a)

    dataloader = make_dataloader(
        cfg.task_name, cfg.hdf5_path, cfg.cache_dir, cfg.data.chunk_horizon,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )
    # pseudocode §"normalizer 在 dataset 上，state_dict() 提取一次（统计量训练全程不变）"
    # Resume 时强制用 ckpt 内 normalizer_state, 防止数据集 reseed 后统计量轻微漂移
    if resume_ckpt is not None:
        dataloader.dataset.normalizer.load_state_dict(resume_ckpt["normalizer_state"])
        normalizer_state = resume_ckpt["normalizer_state"]
    else:
        normalizer_state = dataloader.dataset.normalizer.state_dict()

    # === LIBERO env / init_states_all (一次性, 给 evaluate_with_ema 复用) ===
    # NOTE: LIBERO API 模板与 src/eval.py main 内一致; 若与本机版本不符调整
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_goal"]()
    task_id = None
    for i in range(task_suite.n_tasks):
        if task_suite.get_task(i).name == cfg.task_name:
            task_id = i
            break
    if task_id is None:
        raise RuntimeError(f"task_name '{cfg.task_name}' 不在 libero_goal task suite")
    env = OffScreenRenderEnv(
        bddl_file_name=task_suite.get_task_bddl_file_path(task_id),
        camera_heights=cfg.libero.camera_height,
        camera_widths=cfg.libero.camera_width,
    )
    init_states_all = task_suite.get_task_init_states(task_id)

    # === r3m_model (一次性) ===
    # load_r3m 内部 wrap nn.DataParallel(device_ids=[0]); seed!=42 跑 cuda:1 时
    # .to(cuda:1) 后 DataParallel.device_ids[0]=0 与 module 实际 device 不匹配, forward 报错. 解 wrap.
    # 不调 .half(): R3M.forward (r3m/models/models_r3m.py L99) 内部 `obs = obs.float() / 255.0`
    # 强制 fp32 输入, 与 fp16 weights 不匹配. eval-time 单 image 不需要 fp16 加速.
    r3m_model = load_r3m("resnet50")
    if hasattr(r3m_model, "module"):
        r3m_model = r3m_model.module
    r3m_model = r3m_model.to(device).eval()

    # === normalizer (eval 用 GPU 拷贝; dataset 内实例保持 CPU,
    # 否则 DataLoader worker fork 后 normalize() 会 cuda/cpu 混算炸掉) ===
    normalizer = copy.deepcopy(dataloader.dataset.normalizer).to(device)

    # === pseudocode § Early-stop state (resume 时从 ckpt 恢复) ===
    if resume_ckpt is not None:
        global_step = resume_ckpt["global_step"]
        epoch_start = resume_ckpt["epoch"] + 1
        loss_ema = resume_ckpt["loss_ema"]
        loss_ema_history = list(resume_ckpt["loss_ema_history"])
        phase = resume_ckpt["phase"]
        best_sr = resume_ckpt["best_sr"]
        stale_count = resume_ckpt["stale_count"]
        last_eval_epoch = resume_ckpt["last_eval_epoch"]
        # 续训自 resume_ckpt["last_idx"] (新格式) 或 推断自 resume_last_name (旧 ckpt 缺字段);
        # 写下一个 ckpt 时占用另一 slot, 保留较旧的作 fallback
        if "last_idx" in resume_ckpt:
            last_idx = 1 - resume_ckpt["last_idx"]
        else:
            last_idx = 1 if resume_last_name == "last_0.pt" else 0
        # RNG 恢复 (覆盖 cfg.seed 顶部 set 的初始 state)
        torch.set_rng_state(resume_ckpt["torch_rng_state"])
        torch.cuda.set_rng_state(resume_ckpt["cuda_rng_state"])
        np.random.set_state(resume_ckpt["numpy_rng_state"])
        random.setstate(resume_ckpt["python_rng_state"])
    else:
        global_step = 0
        epoch_start = 0
        loss_ema = None
        loss_ema_history = []
        phase = 1
        best_sr = -1.0
        stale_count = 0
        last_eval_epoch = None
        last_idx = 0

    # === pseudocode § Optimizer / Scheduler ===
    params = list(model.parameters()) + list(obs_encoder.parameters())
    optimizer = AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    total_steps = cfg.scheduler.epoch_max * len(dataloader)
    # Resume: 用 last_epoch=global_step-1 让 LambdaLR 跳到当前进度;
    # 不调 scheduler.load_state_dict, 让新 epoch_max 重塑 cosine 曲线 (退到 0 用新总步数).
    scheduler = cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.scheduler.warmup_steps,
        num_training_steps=total_steps,
        last_epoch=(global_step - 1) if resume_ckpt is not None else -1,
    )

    # === pseudocode § EMA shadow (fp32; 覆盖 model + obs_encoder) ===
    if resume_ckpt is not None:
        # ckpt 用 map_location="cpu" 加载, EMA shadow dict 内 tensor 都在 CPU;
        # model 已 .to(device), ema_update 内 shadow_val.add_(online_val) 跨 device 失败. 显式迁移.
        ema_shadow_model = {k: v.to(device) for k, v in resume_ckpt["ema_shadow_model"].items()}
        ema_shadow_obs_encoder = {k: v.to(device) for k, v in resume_ckpt["ema_shadow_obs_encoder"].items()}
    else:
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

    # === pseudocode § wandb ===
    # spec 改 2: wandb.init 改用 cfg.wandb;
    # resume 时复用 ckpt 内 wandb_run_id 续到同一 run.
    wandb_init_kwargs = {
        "project": cfg.wandb.project,
        "name": cfg.wandb.run_name,
        "mode": cfg.wandb.mode,
        "config": cfg.__dict__,
    }
    if resume_ckpt is not None and resume_ckpt.get("wandb_run_id"):
        wandb_init_kwargs["id"] = resume_ckpt["wandb_run_id"]
        wandb_init_kwargs["resume"] = "allow"
    wandb_run = wandb.init(**wandb_init_kwargs)
    wandb_run_id = wandb_run.id

    # === pseudocode § Main loop ===
    for epoch in range(epoch_start, cfg.scheduler.epoch_max):
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
            grad_norm = clip_grad_norm_(params, cfg.train.grad_clip_norm)

            if global_step % cfg.train.log_interval == 0:
                params_before = torch.cat([p.detach().flatten() for p in params])

            optimizer.step()                                  # fp32 AdamW
            scheduler.step()                                  # per-step

            if global_step >= cfg.scheduler.warmup_steps:
                ema_update(ema_shadow_model, model, cfg.ema.decay)
                ema_update(ema_shadow_obs_encoder, obs_encoder, cfg.ema.decay)

            # === pseudocode § per-log_interval logging ===
            if global_step % cfg.train.log_interval == 0:
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
                buckets = cfg.train.tau_buckets
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
        loss_ema = epoch_loss_mean if loss_ema is None else cfg.loss_ema.alpha * loss_ema + (1 - cfg.loss_ema.alpha) * epoch_loss_mean
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
            last_eval_epoch, cfg, last_idx, wandb_run_id,
        ), last_tmp)
        os.replace(last_tmp, last_path)
        last_idx = 1 - last_idx

        # === pseudocode § Phase 1: plateau detection ===
        if phase == 1 and epoch >= cfg.early_stop.phase1_min_epoch:
            denom = loss_ema_history[epoch - cfg.early_stop.plateau_window]
            relative_drop = (denom - loss_ema) / denom
            if relative_drop < cfg.early_stop.plateau_threshold:
                phase = 2
                last_eval_epoch = epoch - cfg.early_stop.plateau_window  # 切换当 epoch 立即触发首次训练中 evaluate

        # === pseudocode § Phase 2: 训练中 evaluate + early stop ===
        if phase == 2 and (epoch - last_eval_epoch) >= cfg.early_stop.phase2_eval_interval:
            result = evaluate_with_ema(
                model, obs_encoder,
                ema_shadow_model, ema_shadow_obs_encoder,
                env, init_states_all, infer_fn, normalizer, r3m_model, device,
                n_episodes=cfg.early_stop.eval_episodes, seed_for_eval=epoch,
                max_steps=cfg.eval.max_steps, action_steps=cfg.infer.action_steps,
            )
            sr = result["success_rate"]
            last_eval_epoch = epoch

            successes = [m["terminate_reason"] == "success" for m in result["episode_metadata"]]
            episode_lengths = [m["episode_length"] for m in result["episode_metadata"]]
            succ_lengths = [l for l, s in zip(episode_lengths, successes) if s]
            wandb.log({
                "eval/quick_sr": sr,
                "eval/best_sr": best_sr,
                "eval/stale_count": stale_count,
                "eval/episode_length_mean": float(np.mean(succ_lengths)) if succ_lengths else 0.0,
                "eval/sr_per_episode": wandb.Histogram(
                    np.array([float(s) for s in successes])
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

            if stale_count >= cfg.early_stop.patience:
                break

    # === pseudocode § Final fallback eval (break / epoch=200 两种退出后都跑) ===
    # seed_for_eval=epoch+1 与最后一次 phase 2 quick eval (seed_for_eval=epoch) 错开 1 格,
    # 抽样集合不重叠.
    result = evaluate_with_ema(
        model, obs_encoder,
        ema_shadow_model, ema_shadow_obs_encoder,
        env, init_states_all, infer_fn, normalizer, r3m_model, device,
        n_episodes=cfg.early_stop.eval_episodes, seed_for_eval=epoch + 1,
        max_steps=cfg.eval.max_steps, action_steps=cfg.infer.action_steps,
    )
    final_sr = result["success_rate"]
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
