# Decisions Log

记录设计决策、与 c2_plan 锁定层的差异、微观实验结果。

| 日期 | 文件 / 范围 | 决策内容 | 证据来源（官方实现 / 微观实验 / 讨论） | c2_plan 是否需更新 |
|---|---|---|---|---|
| 2026-05-12 | src/data.py sliding window padding | 首帧（$t < T_o$）复制首帧 obs；末端（$t + H - 1 > T$）hold pose，用末动作 `actions[T_demo - 1]` 复制 | 用户讨论 + 伪代码锁定 | 否 |
| 2026-05-12 | src/data.py action 归一化 | per-dim min-max → $[-1, 1]$；gripper 维度与 6-DoF joint 同等处理（无特殊分支）；`denormalize` 内 `torch.clamp(-1, 1)`；统计量经 `register_buffer` 嵌入 checkpoint | 用户讨论 + 伪代码锁定 | 否 |
| 2026-05-12 | src/data.py + scripts/precompute_r3m.py R3M cache | 单大 array per task；shape `(N_demos, T_max, 2_views, K_aug, 2048)`；`np.load(path, mmap_mode="r")` | 用户讨论 + 伪代码锁定 | 否 |
| 2026-05-12 | src/data.py augment dispatch | per-sample（每次 `__getitem__` 内抽 `aug_idx`） | 用户讨论 + 伪代码锁定 | 否 |
| 2026-05-12 | src/data.py dataset indexing | 长度 $= \sum_d T_d$（前缀和 + `bisect_right` 二分）；`__getitem__` 返回携带 `demo_id` 与 `step_t` 元数据 | 用户讨论 + 伪代码锁定 | 否 |
| 2026-05-12 | src/data.py eval obs 一致性 | eval 与 train 共用 `preprocess_image`（`augment=False`）+ `Normalizer` 反归一化 | 用户讨论 + 伪代码锁定 | 否 |
| 2026-05-12 | src/data.py + plan/c2_plan.md §3.3 obs concat 顺序 | $(t-1, t)$ 顺序，旧帧在前；c2_plan §3.3 公式同步反转以与伪代码对齐 | reviewer flag（§3.3 vs §4.2 内部不一致）+ 用户裁定 | 是（§3.3 已更新）|
| 2026-05-12 | src/data.py 随机数源 | 统一 torch：`aug_idx = torch.randint(0, K, (1,)).item()`；`preprocess_image` augment 路径 rng 类型为 `torch.Generator`，augment 内 `torch.randint(..., generator=rng)` 与 `torch.rand(..., generator=rng)` | 用户讨论 | 否 |
| 2026-05-12 | src/data.py + scripts/precompute_r3m.py preprocess_image augment 参数 | augment 路径 brightness / contrast 经 kwargs 由 caller（`scripts/precompute_r3m.py`）传入；data.py 不持默认值，未传则 `ValueError` | 用户讨论 | 否 |
| 2026-05-12 | src/data.py preprocess_image 输出格式 | 输出 `(3, 224, 224)` float32 值域 $[0, 255]$，**不外部 ImageNet 归一化**（R3M 模型内部处理；外部再做会导致 train cache 与 eval 路径双重归一化） | 用户讨论 + 伪代码修订（原版本 `/255` + `TF.normalize` 已删） | 否 |
| 2026-05-12 | scripts/precompute_r3m.py augment 参数取值 | `BRIGHTNESS = CONTRAST = 0.3`（经 kwargs 传入 `preprocess_image`） | 用户讨论 | 否 |
