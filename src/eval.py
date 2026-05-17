
import argparse
import csv
import random
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

from src.data import Normalizer, construct_eval_obs
from src.model.ObsEncoder import ObsEncoder
from src.model.unet1d import ConditionalUnet1D

# torch 2.6+ 把 torch.load weights_only 默认值改成 True, 与 LIBERO 内部
# get_task_init_states 加载 numpy 数组 / 我们自己 ckpt 嵌 numpy 与 python rng state
# 均不兼容. 这里恢复旧默认; 仅加载受信本地文件, 不接收外部 ckpt.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat


def rollout(
    model,
    obs_encoder,
    env,
    init_state,                # caller 已 index 选好的单条 LIBERO init state
    max_steps,
    action_steps,              # receding horizon stride: rollout 内每 N 步重规划
    infer_fn,                  # closure (model, obs_feat) → action_chunk shape (1, H, 7)
    normalizer,                # src.data.Normalizer; 提供 .denormalize(action_t)
    r3m_model,
    device,
    collect_failure_videos: bool,
) -> dict:

    env.reset()
    obs_raw = env.set_init_state(init_state)

    image_agent_curr = obs_raw["agentview_image"]
    image_wrist_curr = obs_raw["robot0_eye_in_hand_image"]
    joints_curr      = obs_raw["robot0_joint_pos"]
    grippers_curr    = obs_raw["robot0_gripper_qpos"]
    # 首帧 padding (c2_plan §4.2)
    image_agent_prev, image_wrist_prev = image_agent_curr, image_wrist_curr
    joints_prev, grippers_prev         = joints_curr, grippers_curr

    frames = [] if collect_failure_videos else None
    terminate_reason = None
    episode_length = max_steps
    chunk = None

    for step in range(max_steps):
        if step % action_steps == 0:
            obs_concat = construct_eval_obs(
                image_agent_curr, image_wrist_curr, joints_curr, grippers_curr,
                image_agent_prev, image_wrist_prev, joints_prev, grippers_prev,
                r3m_model, device,
            )
            obs_feat = obs_encoder(obs_concat.unsqueeze(0))
            chunk = infer_fn(model, obs_feat)

        action_t = chunk[:, step % action_steps, :]
        action_denormalized = normalizer.denormalize(action_t)
        action_todo = action_denormalized.squeeze(0).detach().cpu().numpy()

        obs_raw, _, _, info = env.step(action_todo)

        if collect_failure_videos:
            frames.append(obs_raw["agentview_image"].copy())

        image_agent_prev, image_wrist_prev = image_agent_curr, image_wrist_curr
        joints_prev, grippers_prev         = joints_curr, grippers_curr
        image_agent_curr = obs_raw["agentview_image"]
        image_wrist_curr = obs_raw["robot0_eye_in_hand_image"]
        joints_curr      = obs_raw["robot0_joint_pos"]
        grippers_curr    = obs_raw["robot0_gripper_qpos"]

        if info.get("success", False):
            terminate_reason = "success"
            episode_length = step + 1
            break
    else:
        terminate_reason = "timeout"

    success = (terminate_reason == "success")
    frames_out = np.stack(frames, axis=0) if collect_failure_videos else None

    return {
        "success": success,
        "frames": frames_out,
        "terminate_reason": terminate_reason,
        "episode_length": episode_length,
    }


def evaluate(
    model,
    obs_encoder,
    env,
    init_states_for_episodes,   # list[init_state]，长度 = num_episodes，caller 已 index 完

    infer_fn,
    normalizer,
    r3m_model,
    device,
    collect_failure_videos: bool,
    max_steps,
    action_steps,
) -> dict:

    num_episodes = len(init_states_for_episodes)
    successtime = 0
    failure_videos = []
    failure_count = 0
    episode_metadata = []

    for episode_id, init_state in enumerate(init_states_for_episodes):
        result = rollout(
            model, obs_encoder, env, init_state, max_steps, action_steps,
            infer_fn, normalizer, r3m_model, device,
            collect_failure_videos,
        )

        if result["success"]:
            successtime += 1
        else:
            failure_count += 1
            if collect_failure_videos:
                # 蓄水池采样 Algorithm R, k=5
                if len(failure_videos) < 5:
                    failure_videos.append(result["frames"])
                else:
                    idx = random.randint(0, failure_count - 1)
                    if idx < 5:
                        failure_videos[idx] = result["frames"]

        episode_metadata.append({
            "episode_id": episode_id,
            "terminate_reason": result["terminate_reason"],
            "episode_length": result["episode_length"],
        })

    return {
        "success_rate": successtime / num_episodes,
        "episode_metadata": episode_metadata,
        "failure_videos": failure_videos,
    }


def parse_args():
    # --N / --T_infer / --num_episodes / --max_steps default=None: 三级 fallback
    # (CLI 显式 > ckpt cfg > 函数 default); main() 内 resolve.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["cfm", "dp"])
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--T_infer", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=0)
    return parser.parse_args()


def main():
    # ckpt["cfg"] 必含字段（train.py build_*_state 写入 cfg.__dict__，运行时不校验）:
    #   unet.{action_dim, obs_dim, embedded_dim, down_dims, kernel_size}
    #   obs_encoder.{num_blocks, num_heads, mlp_ratio, dim}
    #   data.{chunk_horizon}
    #   infer.{N, T_infer, action_steps}
    #   dp.{T}                                      # 仅 --method dp 必需
    #   libero.{camera_height, camera_width}
    #   eval.{max_steps}
    #   early_stop.{eval_episodes}
    # ckpt 顶层必含: ema_shadow_model, ema_shadow_obs_encoder, normalizer_state,
    #                task_name, cfg
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === step 1: 加载 ckpt ===
    ckpt = torch.load(args.ckpt, map_location=device)
    for k in ("ema_shadow_model", "ema_shadow_obs_encoder",
              "normalizer_state", "task_name"):
        if k not in ckpt:
            raise RuntimeError(f"ckpt 缺字段 '{k}'; train.py 侧未写入该字段")
    task_name = ckpt["task_name"]
    # spec step 2: "ckpt 内 config 或硬编码默认" → 选 ckpt['cfg'] 路径,
    # 保证 model/obs_encoder shape 与训练侧严格一致 (train.py build_best_state 写入 cfg.__dict__)
    cfg_dict = ckpt["cfg"]
    unet_cfg = cfg_dict["unet"]
    obs_encoder_cfg = cfg_dict["obs_encoder"]

    # === step 1.5: CLI args fallback resolution (CLI 显式 > ckpt cfg) ===
    infer_cfg = cfg_dict["infer"]
    eval_cfg = cfg_dict["eval"]
    libero_cfg = cfg_dict["libero"]
    early_stop_cfg = cfg_dict["early_stop"]
    N = args.N if args.N is not None else infer_cfg.N
    T_infer = args.T_infer if args.T_infer is not None else infer_cfg.T_infer
    num_episodes = args.num_episodes if args.num_episodes is not None else early_stop_cfg.eval_episodes
    max_steps = args.max_steps if args.max_steps is not None else eval_cfg.max_steps
    action_steps = infer_cfg.action_steps

    # === step 2: 实例化 model & obs_encoder, 加载 EMA 权重 ===
    model = ConditionalUnet1D(
        action_dim=unet_cfg.action_dim,
        obs_dim=unet_cfg.obs_dim,
        embedded_dim=unet_cfg.embedded_dim,
        down_dims=unet_cfg.down_dims,
        kernel_size=unet_cfg.kernel_size,
    ).to(device)
    obs_encoder = ObsEncoder(
        num_blocks=obs_encoder_cfg.num_blocks,
        num_heads=obs_encoder_cfg.num_heads,
        mlp_ratio=obs_encoder_cfg.mlp_ratio,
        dim=obs_encoder_cfg.dim,
    ).to(device)
    model.load_state_dict(ckpt["ema_shadow_model"], strict=False)
    obs_encoder.load_state_dict(ckpt["ema_shadow_obs_encoder"], strict=False)

    # === step 3: Normalizer (占位构造, load_state_dict 覆盖 min/max buffer) ===
    action_dim = unet_cfg.action_dim
    normalizer = Normalizer(np.zeros(action_dim), np.ones(action_dim)).to(device)
    normalizer.load_state_dict(ckpt["normalizer_state"])

    # === step 4: R3M ===
    # load_r3m 内部 wrap nn.DataParallel(device_ids=[0]); --device cuda:1 时
    # .to(cuda:1) 后 DataParallel.device_ids[0]=0 与 module 实际 device 不匹配, forward 报错. 解 wrap.
    from r3m import load_r3m
    r3m_model = load_r3m("resnet50")
    if hasattr(r3m_model, "module"):
        r3m_model = r3m_model.module
    r3m_model = r3m_model.to(device).eval().half()

    # === step 5: eval mode ===
    model.eval()
    obs_encoder.eval()

    # === step 6: LIBERO env ===
    # NOTE: LIBERO API 模板按 libero==0.1 常见用法写; 若与本机版本不符, 调整
    #       task_suite / task 属性访问 / OffScreenRenderEnv 参数
    #       (executor 无 /root/shared-nvme/LIBERO/ 读权限, 未现场校验)
    from libero.libero import benchmark as libero_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_goal"]()
    task_id = None
    for i in range(task_suite.n_tasks):
        if task_suite.get_task(i).name == task_name:
            task_id = i
            break
    if task_id is None:
        raise RuntimeError(f"task_name '{task_name}' 不在 libero_goal task suite")
    env = OffScreenRenderEnv(
        bddl_file_name=task_suite.get_task_bddl_file_path(task_id),
        camera_heights=libero_cfg.camera_height,
        camera_widths=libero_cfg.camera_width,
    )

    # === step 7: init_states 切片 ===
    init_states_all = task_suite.get_task_init_states(task_id)
    start = args.base_seed
    end = args.base_seed + num_episodes
    if end > len(init_states_all):
        raise RuntimeError(
            f"init_states_all len={len(init_states_all)} < base_seed+num_episodes={end}"
        )
    init_states_for_episodes = init_states_all[start:end]

    # === step 8: infer_fn closure ===
    H, d_a = cfg_dict["data"].chunk_horizon, unet_cfg.action_dim
    if args.method == "cfm":
        from src.cfm import euler_sample
        infer_fn = lambda m, o: euler_sample(m, o, H, d_a, N=N)
    else:
        from src.dp import ddim_sample, squared_cosine_schedule
        alpha_bar = squared_cosine_schedule(T=cfg_dict["dp"].T).to(device)
        T = alpha_bar.shape[0]
        # HuggingFace DDIMScheduler leading spacing, steps_offset=0 (decisions.md 2026-05-17 row 43)
        step_ratio = T // T_infer
        timesteps = (torch.arange(0, T_infer) * step_ratio).flip(0).tolist()
        infer_fn = lambda m, o: ddim_sample(m, o, alpha_bar, timesteps, H, d_a)

    # === step 9: evaluate ===
    result = evaluate(
        model, obs_encoder, env, init_states_for_episodes,
        infer_fn, normalizer, r3m_model, device,
        collect_failure_videos=True, max_steps=max_steps, action_steps=action_steps,
    )

    # === step 10: 写盘 mp4 + CSV ===
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frames in enumerate(result["failure_videos"]):
        imageio.mimsave(str(output_dir / f"failure_{i}.mp4"), frames, fps=20)

    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["success_rate", result["success_rate"]])
        writer.writerow([])
        writer.writerow(["episode_id", "terminate_reason", "episode_length"])
        for ep in result["episode_metadata"]:
            writer.writerow([
                ep["episode_id"],
                ep["terminate_reason"],
                ep["episode_length"],
            ])

    # === step 11: print success_rate + exit 0 ===
    print(result["success_rate"])
    sys.exit(0)


if __name__ == "__main__":
    main()


# ============================================================
# pseudocode → code block 对照
# ============================================================
# spec imports 块                          → import random; import numpy as np;
#                                            from src.data import construct_eval_obs
# spec def rollout(...) 签名               → def rollout(...) -> dict
# spec env.reset / set_init_state /
#      _get_observations                   → env.reset(); env.set_init_state(init_state);
#                                            obs_raw = env._get_observations()
# spec curr 解包 4 modality                → image_agent_curr / image_wrist_curr /
#                                            joints_curr / grippers_curr
# spec "首帧 padding (c2_plan §4.2)"       → prev := curr (4 modality)
# spec frames / terminate_reason /
#      episode_length / chunk 初始化       → 同名 4 行
# spec for step in range(max_steps):       → for step in range(max_steps):
# spec if step % 8 == 0: 重规划块          → construct_eval_obs(...) → obs_encoder(...) →
#                                            infer_fn(model, obs_feat)
# spec action_t 切片 + normalizer.denormalize → chunk[:, step%8, :] →
#      + to numpy                              normalizer.denormalize(action_t) →
#                                              squeeze + cpu().numpy()
# spec env.step (F1: done 弃用, 4-tuple
#      第三位丢弃)                          → obs_raw, _, _, info = env.step(action_todo)
# spec collect_failure_videos →
#      frames.append(agentview.copy())     → if collect_failure_videos: frames.append(...)
# spec prev := curr ; curr := obs_raw      → 同名 4 行 prev shift + 4 行 curr 更新
# spec success → break / else: timeout    → if info.get("success"): break;
#                                            for/else → terminate_reason = "timeout"
# spec success boolean + np.stack frames   → success = (terminate_reason == "success");
#                                            frames_out = np.stack(frames, axis=0) if ...
# spec return 4-key dict                   → return {success, frames, terminate_reason,
#                                            episode_length}
#
# spec def evaluate(...) 签名              → def evaluate(...) -> dict
#                                            [漂移: max_steps 由 spec 位置参数第 5 位
#                                            改为末位 default=300, 待用户裁决]
# spec num_episodes / successtime /
#      failure_videos / failure_count /
#      episode_metadata 初始化              → 同名 5 行
# spec for episode_id, init_state in
#      enumerate(init_states_for_episodes) → 同款 enumerate 循环
# spec rollout(...) 10 参数调用            → result = rollout(model, obs_encoder, env,
#                                            init_state, max_steps, infer_fn, normalizer,
#                                            r3m_model, device, collect_failure_videos)
# spec if result["success"]: successtime
#      += 1 / else: failure_count += 1     → 同款 if/else 分支
# spec 蓄水池 Algorithm R, k=5
#      (collect_failure_videos 门控)       → if collect_failure_videos:
#                                              if len < 5: append
#                                              else: idx = random.randint(0, fc-1);
#                                                    if idx < 5: failure_videos[idx] = ...
# spec episode_metadata.append(3-key dict) → append({episode_id, terminate_reason,
#                                            episode_length})
# spec return 3-key dict                   → return {success_rate, episode_metadata,
#                                            failure_videos}
#
# === 外包段 (c2_coding_plan §eval.py "功能") spec → code 对照 ===
# 功能段: "评估入口. 加载 EMA ckpt → LIBERO 实例化 → episode rollout
#          → metric 聚合 → 失败 video 保存"
#
# spec CLI 参数 (8 项)                     → parse_args(): argparse.ArgumentParser +
#                                            8 add_argument
# spec step 1: torch.load + 期望字段
#      校验 + task_name 缺时 raise         → ckpt = torch.load(...); for k in (...):
#                                            if k not in ckpt: raise RuntimeError(...)
#                                            [二选一: spec "ckpt 内 config 或硬编码默认"
#                                             选 ckpt['cfg'] 路径, 避免 train/eval shape 漂移]
# spec step 2: 实例化 ConditionalUnet1D /
#      ObsEncoder + load_state_dict       → ConditionalUnet1D(action_dim=...,...).to(device);
#                                            ObsEncoder(...).to(device);
#                                            load_state_dict(ckpt["ema_shadow_*"], strict=False)
# spec step 3: Normalizer + load_state_dict → Normalizer(zeros, ones).to(device);
#                                              load_state_dict(ckpt["normalizer_state"])
# spec step 4: r3m_model = load_r3m
#      ("resnet50").to(device).eval().half() → 同款 (lazy import from r3m)
# spec step 5: model.eval/obs_encoder.eval → model.eval(); obs_encoder.eval()
# spec step 6: LIBERO env (task_suite +
#      task_name → OffScreenRenderEnv)    → get_benchmark_dict()["libero_goal"]() →
#                                            遍历 n_tasks 匹配 task.name == task_name →
#                                            OffScreenRenderEnv(bddl_file_name=task.bddl_file,
#                                            camera_heights=128, camera_widths=128)
#                                            [LIBERO API 模板, 待用户验证]
# spec step 7: init_states_all[base_seed:
#      base_seed+num_episodes], 越界 raise → 同款; end > len → RuntimeError
# spec step 8: cfm → euler_sample(N=args.N) → if cfm: from src.cfm import euler_sample
#      / dp → ddim_sample(alpha_bar,         else: from src.dp import ddim_sample,
#      T_infer)                              squared_cosine_schedule;
#                                            alpha_bar = squared_cosine_schedule(T=100)
# spec step 9: evaluate(...,
#      collect_failure_videos=True)        → evaluate(model, obs_encoder, env,
#                                            init_states_for_episodes, infer_fn, normalizer,
#                                            r3m_model, device,
#                                            collect_failure_videos=True,
#                                            max_steps=args.max_steps)
# spec step 10: mp4 (imageio.mimsave
#      fps=20, failure_{i}.mp4) + CSV     → for i, frames in enumerate(failure_videos):
#      (success_rate 顶行 + metadata 表)    imageio.mimsave(...);
#                                            csv.writer 写 success_rate + episode_metadata
# spec step 11: print + sys.exit(0)        → print(result["success_rate"]); sys.exit(0)
