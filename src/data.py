"""LIBERO-Goal data pipeline.

伪代码段 → 代码对照见每段顶部 `# pseudo §N` 标注。
"""

import bisect
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


# ──────────────────────────────────────────
# pseudo §1: Normalizer（action 归一化 / 反归一化）
# ──────────────────────────────────────────

class Normalizer(nn.Module):
    """per-dim min-max → [-1, 1]，统计量随 checkpoint 持久化"""

    def __init__(self, action_min, action_max):
        super().__init__()
        self.register_buffer("min", torch.as_tensor(action_min, dtype=torch.float32))
        self.register_buffer("max", torch.as_tensor(action_max, dtype=torch.float32))
        self.eps = 1e-6

    def normalize(self, action):
        return 2 * (action - self.min) / (self.max - self.min + self.eps) - 1

    def denormalize(self, action_normalized):
        clipped = torch.clamp(action_normalized, -1, 1)
        return (clipped + 1) / 2 * (self.max - self.min + self.eps) + self.min


# ──────────────────────────────────────────
# pseudo §2: 图像预处理（train 预计算 / eval 在线共用）
# ──────────────────────────────────────────

def preprocess_image(image, augment=False, rng=None, brightness=None, contrast=None):
    """image: (H_raw, W_raw, 3) uint8 → (3, 224, 224) float32，值域 [0, 255]。

    ImageNet 归一化由 R3M 模型内部完成，preprocess_image 不再外部归一化。
    augment=True 路径下 brightness 与 contrast 由 caller 提供（数值决策属
    scripts/precompute_r3m.py）。rng: torch.Generator。
    """
    image = torch.as_tensor(image).permute(2, 0, 1)
    image = TF.resize(image, [224, 224], antialias=True)
    if augment:
        if brightness is None or contrast is None:
            raise ValueError("augment=True 要求 brightness 与 contrast 参数")
        top = int(torch.randint(0, 224 - 216 + 1, (1,), generator=rng).item())
        left = int(torch.randint(0, 224 - 216 + 1, (1,), generator=rng).item())
        image = TF.crop(image, top, left, 216, 216)
        image = TF.resize(image, [224, 224], antialias=True)
        b_factor = (torch.rand(1, generator=rng).item() * 2 - 1) * brightness + 1.0
        c_factor = (torch.rand(1, generator=rng).item() * 2 - 1) * contrast + 1.0
        image = TF.adjust_brightness(image, b_factor)
        image = TF.adjust_contrast(image, c_factor)
    image = image.float()
    return image


# ──────────────────────────────────────────
# pseudo §3: R3M cache 加载
# ──────────────────────────────────────────

def load_r3m_cache(task_name, cache_dir):
    """返回 mmap array, shape (N_demos, T_max, 2_views, K_aug, 2048).

    task_name 取 LIBERO 原生名(无 _demo); cache 文件名约定带 _demo 后缀,
    与 scripts/precompute_r3m.py TASK_LIST 元素 (带 _demo) 存盘格式一致.
    """
    path = Path(cache_dir) / f"{task_name}_demo_r3m.npy"
    return np.load(path, mmap_mode="r")


# ──────────────────────────────────────────
# pseudo §4: Dataset
# ──────────────────────────────────────────

class LiberoGoalDataset(Dataset):
    def __init__(self, task_name, hdf5_path, cache_dir, chunk_horizon):
        super().__init__()
        # 外包段：HDF5 demo 遍历（LIBERO 布局 data/demo_*）+ 原始数组读取
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            self.actions = [f["data"][k]["actions"][:] for k in demo_keys]
            self.joints = [f["data"][k]["obs"]["joint_states"][:] for k in demo_keys]
            self.grippers = [f["data"][k]["obs"]["gripper_states"][:] for k in demo_keys]
        num_demos = len(self.actions)

        # per-dim min/max 跨所有 demo 所有帧
        all_actions = np.concatenate(self.actions, axis=0)
        action_min = all_actions.min(axis=0)
        action_max = all_actions.max(axis=0)
        self.normalizer = Normalizer(action_min, action_max)

        # R3M mmap cache
        self.r3m_cache = load_r3m_cache(task_name, cache_dir)

        # 前缀和：index → (demo_id, step_t)
        self.cumulative_lengths = [0]
        for i in range(num_demos):
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + len(self.actions[i])
            )
        self.total_length = self.cumulative_lengths[-1]

        self.H = chunk_horizon
        # K 从 cache 末轴派生, 与 precompute_r3m.py 写盘 shape 一致, 不另作 cfg 旋钮.
        self.K = self.r3m_cache.shape[3]
        # obs 窗口固定 2 帧 (prev/curr), 写死在 __getitem__ 与 construct_eval_obs.

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        # 1. index → (demo_id, step_t)
        demo_id = bisect.bisect_right(self.cumulative_lengths, index) - 1
        step_t = index - self.cumulative_lengths[demo_id]
        T_demo = len(self.actions[demo_id])

        # 2. obs 特征：mmap cache + per-sample 抽 aug_idx；obs 窗口顺序 (t-1, t)
        aug_idx = int(torch.randint(0, self.K, (1,)).item())
        obs_features = []
        for tau in [step_t - 1, step_t]:
            if tau < 0:
                tau = 0  # 首帧 padding：复制首帧
            feat_agent = self.r3m_cache[demo_id, tau, 0, aug_idx]
            feat_wrist = self.r3m_cache[demo_id, tau, 1, aug_idx]
            joint = self.joints[demo_id][tau]
            gripper = self.grippers[demo_id][tau]
            obs_features.append(np.concatenate([feat_agent, feat_wrist, joint, gripper]))
        obs = np.concatenate(obs_features)  # (8210,)

        # 3. action chunk + 末端 hold pose padding
        action_chunk = torch.zeros(self.H, 7)
        valid_len = min(self.H, T_demo - step_t)
        action_chunk[:valid_len] = torch.as_tensor(
            self.actions[demo_id][step_t : step_t + valid_len], dtype=torch.float32
        )
        if valid_len < self.H:
            action_chunk[valid_len:] = torch.as_tensor(
                self.actions[demo_id][T_demo - 1], dtype=torch.float32
            )

        # 4. 归一化
        action_chunk_normalized = self.normalizer.normalize(action_chunk)

        # 5. 返回
        return (obs, action_chunk_normalized, demo_id, step_t)


# ──────────────────────────────────────────
# pseudo §5: DataLoader 工厂
# ──────────────────────────────────────────

def make_dataloader(task_name, hdf5_path, cache_dir, chunk_horizon,
                    batch_size=96, num_workers=4):
    dataset = LiberoGoalDataset(task_name, hdf5_path, cache_dir, chunk_horizon)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ──────────────────────────────────────────
# pseudo §6: Eval-time obs 构造（共用 preprocess_image，无增强）
# ──────────────────────────────────────────

def construct_eval_obs(
    image_agent_current, image_wrist_current, joints_current, grippers_current,
    image_agent_prev,    image_wrist_prev,    joints_prev,    grippers_prev,
    r3m_model, device,
):
    """image_*_*: (H, W, 3) uint8 ndarray，agentview / eye_in_hand 各两个时间步.

    返回: obs tensor (8210,) on device. 窗口顺序 (t-1, t)。
    """
    obs_features = []
    for image_agent, image_wrist, joints, grippers in [
        (image_agent_prev, image_wrist_prev, joints_prev, grippers_prev),
        (image_agent_current, image_wrist_current, joints_current, grippers_current),
    ]:
        img_agent = preprocess_image(image_agent, augment=False)
        img_wrist = preprocess_image(image_wrist, augment=False)
        with torch.no_grad():
            feat_agent = r3m_model(img_agent.unsqueeze(0).to(device))  # (1, 2048)
            feat_wrist = r3m_model(img_wrist.unsqueeze(0).to(device))
        obs_features.append(torch.cat([
            feat_agent.squeeze(0),
            feat_wrist.squeeze(0),
            torch.as_tensor(joints, device=device, dtype=torch.float32),
            torch.as_tensor(grippers, device=device, dtype=torch.float32),
        ]))
    return torch.cat(obs_features)
