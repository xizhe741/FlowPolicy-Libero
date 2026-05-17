# 外包 Checklist

按 [.claude/CLAUDE.md](.claude/CLAUDE.md) §File Ownership 的划分，逐文件枚举当前已落地代码中的外包部分（细到 class / function / 代码段）。

---

## 整个文件外包

### [scripts/precompute_r3m.py](scripts/precompute_r3m.py)
| 模块 | 位置 | 用于 |
|---|---|---|
| `set_global_seeds` | [L33](scripts/precompute_r3m.py#L33) | python / numpy / torch / cuda 四套全局种子统一入口（preprocess_image 仍走显式 Generator） |
| `load_r3m_model` | [L44](scripts/precompute_r3m.py#L44) | 加载 R3M resnet50、unwrap DataParallel、迁移到 cuda + half + eval |
| `r3m_forward` | [L52](scripts/precompute_r3m.py#L52) | batch stack → cuda → fp16 autocast → R3M(.) → cpu numpy fp32 串行 |
| `write_to_cache` | [L64](scripts/precompute_r3m.py#L64) | 把 (B, 2048) feature 写回 (N_demo, T, V, K, 2048) 大数组的索引循环 |
| `precompute_task` | [L75](scripts/precompute_r3m.py#L75) | 单 task 两 pass：pass1 metadata 扫 T_max，pass2 (t, view, k) 三重循环 augment + R3M + 写盘 |
| `main` | [L147](scripts/precompute_r3m.py#L147) | task list driver，绑定路径与跨 task 共享 rng |

### [src/dp.py](src/dp.py) — 对比组，整文件外包
| 模块 | 位置 | 用于 |
|---|---|---|
| `squared_cosine_schedule` | [L29](src/dp.py#L29) | IDDPM 余弦 ᾱ 数组（含 β clip 重导一遍 cumprod），供 `dp_loss` 的 forward q-process 取 √ᾱ_t / √(1-ᾱ_t) |
| `dp_loss` | [L50](src/dp.py#L50) | DP 训练 loss：t_int 采样 → 加噪 a_t → t_norm → eps-prediction → MSE + per-sample 诊断 |

---

## 部分外包

### [src/data.py](src/data.py)
| 模块 | 位置 | 用于 |
|---|---|---|
| `load_r3m_cache` | [L70](src/data.py#L70) | 按 `{task}_r3m.npy` 构路径 + mmap_mode='r' 加载特征 cache |
| HDF5 demo 遍历段 | [L85-L89](src/data.py#L85-L89) | `data/demo_*` 排序 + 读 `actions` / `obs.joint_states` / `obs.gripper_states` 到 numpy list |
| `make_dataloader` | [L157](src/data.py#L157) | `DataLoader` 实例化:batch_size / num_workers / pin_memory / drop_last |

> 同文件内 `Normalizer`、`preprocess_image`、`LiberoGoalDataset.__getitem__`、`construct_eval_obs` 属高价值代码（sliding window padding、augment dispatch、eval/train obs 一致性）— 不在外包清单。

### [src/model/modules.py](src/model/modules.py)
| 模块 | 位置 | 用于 |
|---|---|---|
| `sinusoidal_embedding` | [L18](src/model/modules.py#L18) | 标量 t → (B, embedded_dim) 嵌入；被 `ConditionalUnet1D.time_mlp` 与 `DiT.t_mlp` 调用 |
| `Downsample1d` | [L75](src/model/modules.py#L75) | encoder level 末端 stride=2 Conv1d |
| `Upsample1d` | [L84](src/model/modules.py#L84) | decoder level 末端 ConvTranspose1d (k=4, s=2, p=1) |

> `FiLMResBlock1D` 属高价值代码 — 不在外包清单。

### [src/model/unet1d.py](src/model/unet1d.py)
| 模块 | 位置 | 用于 |
|---|---|---|
| `conv_out` / `norm_out` / `activate` / `exit` 四件套 | [L85-L88](src/model/unet1d.py#L85-L88) | 末端 Conv1dBlock + 1×1 Conv1d，把 start_dim 映回 action_dim |

> `ConditionalUnet1D` 的 condition 拼接、encoder/bottleneck/decoder 主干、3-skip / 2-pop 不对称结构属高价值代码。

### [src/model/ObsEncoder.py](src/model/ObsEncoder.py)
| 模块 | 位置 | 用于 |
|---|---|---|
| `nn.MultiheadAttention(batch_first=True)` 调用细节 | [L44](src/model/ObsEncoder.py#L44), [L56](src/model/ObsEncoder.py#L56) | `PreNormBlock` 内部 q=k=v=h 的形状语义 |
| `nn.Embedding(2/4, dim)` 默认初始化 | [L82-L83](src/model/ObsEncoder.py#L82-L83) | step_emb / modality_emb 的初值（用户已选默认 N(0,1)） |

> `ObsEncoder.forward` 的 8210 raw concat 切分、per-modality 投影 + step/modality emb、`PreNormBlock` 残差结构属高价值。

### [src/train.py](src/train.py)
| 模块 | 位置 | 用于 |
|---|---|---|
| `parse_args` | [L33](src/train.py#L33) | CLI: --config / --task_name / --seed / --device / --output_dir |
| `_dict_to_ns` | [L43](src/train.py#L43) | yaml dict 递归转 `SimpleNamespace`（让 cfg 走点访问） |
| `load_cfg` | [L49](src/train.py#L49) | yaml 加载 + CLI override 合并 + `obs_dim == 8 * obs_encoder.dim` 约束断言 |
| `cosine_schedule_with_warmup` | [L72](src/train.py#L72) | warmup 500 步线性 0→lr + 之后 cosine 退到 0 的 `LambdaLR` 写法 |
| `build_last_state` | [L114](src/train.py#L114) | last ckpt 打包：model / obs_encoder / EMA shadow / optimizer / scheduler / 4 套 RNG / cfg / phase / best_sr / stale_count |
| `build_best_state` | [L146](src/train.py#L146) | best ckpt 打包：EMA shadow + normalizer + best_sr + epoch/global_step + cfg |
| `wandb.init` | [L232-L236](src/train.py#L232-L236) | run name 与 cfg 注入 |
| per-LOG_INTERVAL log_dict 拼装 | [L286-L307](src/train.py#L286-L307) | loss / lr / grad_norm / optim_step_norm / ema_shadow_norm / action 分布直方 / τ-bin per-sample loss |
| epoch-end log + Histogram | [L317-L321](src/train.py#L317-L321), [L354-L362](src/train.py#L354-L362) | loss_epoch_mean / loss_ema / phase / quick_sr / episode_length / sr_per_episode |
| 原子 rename 写盘三段 | [L324-L333](src/train.py#L324-L333), [L367-L373](src/train.py#L367-L373), [L390-L396](src/train.py#L390-L396) | `.pt.tmp` → `os.replace` → `.pt`；rolling-2 last 用 `last_idx = 1 - last_idx` |
| bf16 autocast 包装 | [L259-L262](src/train.py#L259-L262) | `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` 仅包 forward + loss，外面强 `loss.float()` |

> 主循环逻辑流（loss → backward → clip → step → EMA → plateau detection → phase 切换 → 训练中 evaluate → best_sr 比较 → stale_count → early stop）以及 `ema_update`、`evaluate_with_ema` 的 shadow swap 属高价值。

---

## 尚未实现，按 spec 应整文件外包
| 文件 | 用途 |
|---|---|
| [src/infer.py](src/infer.py) | 单次 obs → action chunk 映射；receding horizon 计数器（spec 倾向：放 eval.py，保持 infer 无状态） |
| [scripts/ablation_ode.py](scripts/ablation_ode.py) | `eval.py` 的 thin wrapper，扫 N ∈ {1,2,4,8,16} ODE 步数 |
| [scripts/run_all.sh](scripts/run_all.sh) | 多 seed × method × task 的 shell driver |
| [src/eval.py](src/eval.py) 非 rollout 段 | mp4 保存、CSV 聚合、ckpt I/O、LIBERO `env.reset/step` wrapper |

---

不计入外包清单的存量代码（用户旧仓库迁入，与 C2 主线非耦合）：[src/DiT.py](src/DiT.py)、[src/U_net.py](src/U_net.py)、[src/modules.py](src/modules.py)（顶层）。
