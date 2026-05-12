"""scripts/precompute_r3m.py
Offline R3M feature cache with K=8 augmentations per training image.
Output schema mirrors src/data.py loader.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from r3m import load_r3m

from src.data import preprocess_image  # [L2-3.6] 复用 train/eval 统一增强接口

# ──────── Constants ────────
SEED          = 42
K             = 8
TARGET_SIZE   = 224
CROP_SIZE     = 216
BRIGHTNESS    = 0.3                 # source: see decisions.md
CONTRAST      = 0.3                 # source: see decisions.md
BATCH_SIZE    = 256
CACHE_DTYPE   = np.float32          # [L2-2.6] 与 c2_plan §4.4 磁盘估算式对齐
CACHE_SUFFIX  = "_r3m"              # data.py::load_r3m_cache 期望 {task}_r3m.npy
FEATURE_DIM   = 2048
N_VIEWS       = 2


# ──────── Seeding ────────
def set_global_seeds(seed: int) -> None:
    """覆盖 python / numpy / torch global / cuda global。
    preprocess_image 用显式 torch.Generator,由调用方维护,不走 global。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────── R3M model ────────
def load_r3m_model(device: str = "cuda") -> torch.nn.Module:
    model = load_r3m("resnet50")
    model = model.module if hasattr(model, "module") else model  # unwrap DataParallel
    return model.to(device).eval().half()


# ──────── Forward & cache write ────────
@torch.no_grad()
def r3m_forward(
    r3m: torch.nn.Module,
    batch_imgs: list[torch.Tensor],
) -> np.ndarray:
    # 输入: fp32 cpu, shape (3, 224, 224), 值域 [0, 255]
    # R3M.forward 内部: obs/255 -> ImageNet normalize -> ResNet50
    batch = torch.stack(batch_imgs).cuda(non_blocking=True).half()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        feats = r3m(batch)                              # (B, 2048)
    return feats.float().cpu().numpy().astype(CACHE_DTYPE)


def write_to_cache(
    cache: np.ndarray,
    demo_idx: int,
    batch_index: list[tuple[int, int, int]],
    features: np.ndarray,
) -> None:
    for (t, v, k), feat in zip(batch_index, features):
        cache[demo_idx, t, v, k] = feat


# ──────── Per-task pipeline ────────
def precompute_task(
    task: str,
    data_dir: Path,
    cache_dir: Path,
    r3m: torch.nn.Module,
    rng: torch.Generator,
) -> None:
    hdf5_path = data_dir / f"{task}.hdf5"

    # Pass 1: scan demo lengths via cheap metadata read
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        N_demos = len(demo_keys)
        lengths = np.array(
            [f["data"][k]["actions"].shape[0] for k in demo_keys],
            dtype=np.int32,
        )
    T_max = int(lengths.max())

    # [L2-4.3] N_demos * T_max * 2 * K * 2048 * 4 bytes per task
    cache = np.zeros(
        (N_demos, T_max, N_VIEWS, K, FEATURE_DIM),
        dtype=CACHE_DTYPE,
    )

    # Pass 2: augment via preprocess_image + R3M inference
    for demo_idx, demo_key in enumerate(demo_keys):
        with h5py.File(hdf5_path, "r") as f:
            views = [
                f["data"][demo_key]["obs/agentview_rgb"][:],     # (T, H, W, 3) uint8
                f["data"][demo_key]["obs/eye_in_hand_rgb"][:],
            ]
        T_d = int(lengths[demo_idx])

        batch_imgs:  list[torch.Tensor] = []
        batch_index: list[tuple[int, int, int]] = []

        # [L1-1.3] 循环顺序 (t, view, k) 唯一决定 rng 消耗序列
        for t in range(T_d):
            for view_idx in range(N_VIEWS):
                raw = views[view_idx][t]                          # (H, W, 3) uint8 numpy
                for k in range(K):
                    # [L2-2.4] 显式 rng 取代 torchvision global RNG
                    # preprocess_image 输入约定: numpy uint8 HWC,直接传入
                    aug_tensor = preprocess_image(
                        raw,
                        augment=True,
                        brightness=BRIGHTNESS,
                        contrast=CONTRAST,
                        rng=rng,
                    )
                    batch_imgs.append(aug_tensor)
                    batch_index.append((t, view_idx, k))

                    if len(batch_imgs) == BATCH_SIZE:
                        feats = r3m_forward(r3m, batch_imgs)
                        write_to_cache(cache, demo_idx, batch_index, feats)
                        batch_imgs.clear()
                        batch_index.clear()

        # Tail flush per demo (不可跨 demo 累积:demo_idx 是外层闭包)
        if batch_imgs:
            feats = r3m_forward(r3m, batch_imgs)
            write_to_cache(cache, demo_idx, batch_index, feats)
            batch_imgs.clear()
            batch_index.clear()

    np.save(cache_dir / f"{task}{CACHE_SUFFIX}.npy", cache)
    del cache


# ──────── Entry ────────
def main(
    data_dir: Path,
    cache_dir: Path,
    task_list: Iterable[str],
) -> None:
    set_global_seeds(SEED)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # [L2-2.4] 跨 task / demo / t / view / k 共享同一 rng,
    # cache 由 (SEED, task_list 顺序) 唯一确定
    rng = torch.Generator(device='cpu')
    rng.manual_seed(SEED)

    r3m = load_r3m_model(device="cuda")
    for task in task_list:
        precompute_task(task, data_dir, cache_dir, r3m, rng)


if __name__ == "__main__":
    DATA_DIR  = Path("/root/shared-nvme/data/libero_goal")
    CACHE_DIR = Path("/root/shared-nvme/caches/r3m_k8")
    TASK_LIST = [
        "put_the_bowl_on_the_plate_demo",
        "put_the_wine_bottle_on_the_rack_demo",
        "open_the_middle_drawer_of_the_cabinet_demo",
        "turn_on_the_stove_demo",
        "open_the_top_drawer_and_put_the_bowl_inside_demo",
    ]
    main(DATA_DIR, CACHE_DIR, TASK_LIST)
