# C2 复现计划：Conditional Flow Matching Policy on LIBERO-Goal

> 本文件是计划主文件 `reading_plan.md` 第 3 节"复现实验"在 C2 选择下的具体规格。所有工程决策已锁定。

---

## 1. 工程目标

本仓库以 LIBERO-Goal 5 个子任务为 benchmark，从零实现一个 CFM 策略，并与 Diffusion Policy 在严格 head-replacement 设定下做对比。读者预期能从 README 与代码中验证以下五点：

1. **CFM 训练目标的精确实现可定位**。README 中数学形式与代码函数双向引用：loss 表达式
   $$\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \mathcal{U}[0,1]} \mathbb{E}_{a_0 \sim \mathcal{N}(0, I),\, a_1 \sim \mathcal{D}}\left\|v_\theta\left(\tau a_1 + (1-\tau) a_0,\, \tau,\, o\right) - (a_1 - a_0)\right\|^2$$
   在代码中对应 `src/cfm.py::cfm_loss`。

2. **Diffusion Policy 对比是 head replacement，不是两个独立 codebase 的横向比较**。U-Net、视觉编码器、数据 pipeline、optimizer、batch、EMA、早停规则、评估协议在 CFM 与 DP 两组实验中严格共享。唯一差异是训练目标（CFM vs $\epsilon$-prediction DDPM）与采样器（Euler vs DDIM）。这使两者 success rate 的差异可以归因到这模模型范式本身。

3. **ODE 步数 ablation 直接探查 CFM 推理效率**。在 $N \in \{1, 2, 4, 8, 16\}$ 上各跑 20 episodes 评估，给出 success rate vs 推理步数曲线。所有 task × seed 共用同一训练 checkpoint，仅推理时改 solver 步数。

4. **Native Continuation 接口已预留**。`infer.py` 中 `chunk_init_strategy` 参数支持 `"gaussian"`（CFM 默认）与 `"native_continuation"`（接口签名已定，实现留作后续工作）。本仓库不评价 native continuation，但工程上为 Gao Yang 组的相关方向预留了直接 plug-in 的位置。

5. **R3M 特征缓存与离线增强是经过显式权衡的工程选择**。R3M frozen 时图像 → 特征是确定性映射，可缓存。但常规图像增强会破坏可缓存性。本仓库采用 $K = 8$ 份离线预增强 + 全部缓存的方案，以 ~5 GB 磁盘换 ~30% 训练加速。该选择在 README 中显式记录。

### 定位说明（与 π₀ 的关系）

本工作不是 π₀ 复刻。π₀ 的核心特征——VLM backbone（PaliGemma + SigLIP）、自然语言任务指令、action expert 中的 token 级 cross-attention conditioning、generalist policy——在本工作内不实现。本工作的 conditioning 机制是 FiLM（channel-wise affine），观测条件不含语言，是任务训练独立 specialist policy。与 π₀ 共享的仅是 CFM 训练目标本身。

本工作的对科对象是 Diffusion Policy，不是 π₀，与 π₀ 的关系仅作为概念背景，不作实证比较。向 π₀ 方向的扩展路径见 §13。

---

## 2. 任务范围与对比规格

| 项 | 规格 |
|---|---|
| 任务 | LIBERO-Goal 5 子任务（具体 5 个在 Day 5–6 锁定，依据 demo 数与 task 复杂度分布）|
| 种子 | 3 / 任务 |
| Eval episodes | 20 / (任务, 种子) |
| ODE ablation 步数 | $N \in \{1, 2, 4, 8, 16\}$ |
| DP 对比 | 同 codebase 内 head replacement，相同 5 任务 × 3 种子 × 20 episodes |
| Native Continuation | 仅预留接口签名，不做 ablation |

---

## 3. 方法

### 3.1 视觉编码器

R3M（Nair et al., CoRL 2022）ResNet-50，frozen。双视觉输入：

- `agentview`（第三人称）→ 2048 维
- `eye_in_hand`（手腕相机）→ 2048 维

不微调，不修改第一层。图像 resize 128 → 224（bilinear）后过 R3M。

### 3.2 速度场网络

1D Temporal U-Net，沿 chunk 时间维做 1D 卷积。结构：

- 4 层下采样 + bottleneck + 4 层上采样 + skip connection。
- ResBlock 内部使用 GroupNorm + FiLM 条件注入。
- 参数量 ~50M。

输入：
- $a_\tau \in \mathbb{R}^{H \times d_a}$，$H = 16$，$d_a = 7$（6-DoF + gripper）。
- 条件 $c = \text{MLP}([\text{embed}_\text{sin}(\tau);\, o])$，其中 $o$ 见 3.3。
- $c$ 在每个 ResBlock 中通过 FiLM 注入：
  $$h_\text{out} = \gamma(c) \odot \text{GroupNorm}(h_\text{in}) + \beta(c)$$

输出：$v_\theta(a_\tau, \tau, o) \in \mathbb{R}^{H \times d_a}$。

### 3.3 观测条件

$o = \text{concat}(\text{R3M}(I^\text{agent}_{t-1}),\, \text{R3M}(I^\text{wrist}_{t-1}),\, q_{t-1},\, g_{t-1},\, \text{R3M}(I^\text{agent}_t),\, \text{R3M}(I^\text{wrist}_t),\, q_t,\, g_t)$

其中 $q_t \in \mathbb{R}^7$ 关节角，$g_t \in \mathbb{R}^2$ gripper 状态，$T_o = 2$ 历史帧。$d_o = 2 \times (4096 + 7 + 2) = 8210$。不含 ee_states，与 DP LIBERO 标尺一致。

### 3.4 训练目标（CFM）

$$\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \mathcal{U}[0,1]} \mathbb{E}_{a_0 \sim \mathcal{N}(0, I)} \mathbb{E}_{(a_1, o) \sim \mathcal{D}}\left\|v_\theta(\tau a_1 + (1-\tau) a_0, \tau, o) - (a_1 - a_0)\right\|^2$$

实现要点：
- $a_1$ 是经 min-max 归一化到 $[-1, 1]^{H \times d_a}$ 的 ground truth action chunk。
- $a_0$ 从标准正态采样，与 $a_1$ 形状相同。
- $\tau$ 每个样本独立从 $\mathcal{U}[0, 1]$ 采。
- Loss 在 $H \times d_a$ 维度上做 mean reduction。

### 3.5 训练目标（DP，对比组）

$\epsilon$-prediction DDPM，squared cosine noise schedule，$T = 100$ 训练步：

$$\mathcal{L}_\text{DP}(\theta) = \mathbb{E}_{t \sim \mathcal{U}\{1, \ldots, T\}} \mathbb{E}_{a_0,\, \epsilon}\left\|\epsilon_\theta(\sqrt{\bar{\alpha}_t} a_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t, o) - \epsilon\right\|^2$$

U-Net、视觉编码器、$o$ 构造、归一化、batch、optimizer、EMA、训练 epoch 与 CFM 严格相同。

### 3.6 推理

CFM：从 $a^{(0)} \sim \mathcal{N}(0, I)$ 出发，Euler 积分

$$a^{(n+1)} = a^{(n)} + \frac{1}{N} v_\theta\left(a^{(n)}, \frac{n}{N}, o\right), \quad n = 0, \ldots, N-1$$

默认 $N = 4$。Action chunking 一次前向输出整 chunk。

DP：DDIM with $T_\text{infer} = 16$。

Action chunking 执行：每个 $k = 8$ env step 重新规划一次（receding horizon），chunk 前向 $k$ 步执行后丢弃 $[k, H-1]$ 部分。

---

## 4. 数据 Pipeline

### 4.1 数据来源

LIBERO-Goal HDF5，每任务约 50 demo。下载后仅保留 Goal suite，其他 suite 删除以控制磁盘占用。

### 4.2 Sliding window

每个 demo 在每个时间步 $t$ 抽取训练样本 $(\{o_{t-1}, o_t\},\, \{a_t, \ldots, a_{t+H-1}\})$。

边界：
- 起始 $t < T_o$：首帧 padding。
- 末端 $t + H - 1 > T$：末动作 padding（机械臂保持静止）。

### 4.3 Action 归一化

Min-max → $[-1, 1]$。统计量从 train set 全部 demo 的 action 中按维度计算并固定。推理时反归一化后送入环境。

### 4.4 R3M 特征缓存与数据增强

离线预处理：每张训练图像生成 $K = 8$ 份增强版本，每份过 R3M 后存盘。

增强：
- Random crop 224 → 216 → resize 224。
- Color jitter（brightness, contrast）。

训练时每个样本随机抽取 1 份增强后特征。Eval 时 R3M 在线推理（无增强、单份）。

磁盘：5 任务 × 50 demo × 150 step × 2 视觉 × 8 份 × 2048 dim × 4 byte ≈ 5 GB。

预处理时长：约 100 min（4090 fp16）。

---

## 5. 训练配置

| 项 | 值 |
|---|---|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-4}$ |
| LR schedule | Cosine + 500 step warmup |
| Weight decay | $1 \times 10^{-6}$ |
| Gradient clip | 1.0 |
| Batch size | 96 / GPU |
| Mixed precision | bf16 autocast |
| EMA decay | 0.9999（inference 用 EMA 权重）|
| Epoch 上限 | 200 |
| 早停规则 | 每 20 epoch quick eval（10 episodes），连续两次 success rate 不上升即停 |
| 并行 | 独立 seed / 卡（4090 × 2 同时跑两个不同 task-seed）|

预期单 run 训练时长：~1.9 hr（150 epoch 收敛假设下）。

---

## 6. 评估协议

- **Quick eval**（训练中早停判定）：10 episodes，固定起始状态种子，每 20 epoch 一次。
- **Full eval**（训练结束）：20 episodes，固定起始状态种子，独立于 quick eval 种子。
- **ODE ablation eval**：每个训练好的 CFM checkpoint 在 $N \in \{1, 2, 4, 8, 16\}$ 上各跑 20 episodes。
- **Inference 配置**：EMA 权重，CFM 用 Euler $N=4$（默认），DP 用 DDIM $T_\text{infer}=16$。
- **失败 video**：保存每任务最多 5 条失败 episode 的 mp4，README 中以 GIF 形式展示典型失败模式（手撇在失败类型分类）。
- **Metrics**：success rate（per task, per seed），success rate 跨 seed 平均与标准差，ODE 步数曲线。

---

## 7. 仓库结构

```
flow-policy-libero/
├── README.md
├── pyproject.toml
├── configs/
│   ├── cfm_default.yaml
│   ├── dp_default.yaml
│   └── ablation_ode_steps.yaml
├── src/
│   ├── data.py          # LIBERO HDF5, sliding window, R3M cache loader, augment dispatch
│   ├── models.py        # 1D Temporal U-Net + FiLM ResBlock
│   ├── cfm.py           # CFM loss, Euler ODE solver
│   ├── dp.py            # DDPM loss, DDIM sampler
│   ├── train.py         # Training entry, EMA, early stopping
│   ├── eval.py          # Full eval, video saving
│   └── infer.py         # Inference with chunk_init_strategy ∈ {"gaussian", "native_continuation"}
├── scripts/
│   ├── precompute_r3m.py    # 离线增强 + R3M 特征缓存
│   ├── run_all.sh           # 5 task × 3 seed CFM + DP 串行调度
│   └── ablation_ode.py      # ODE 步数 ablation
├── tests/
│   ├── test_cfm_loss.py     # 1D 高斯目标分布上验证 closed-form
│   └── test_euler_solver.py # 线性 ODE $\dot{x} = -x$ 解析对种
├── notes/                   # P1 技术 note (Translating Flow to Policy, Native Continuation)
└── results/
    ├── success_rates.csv
    ├── ablation_ode.csv
    └── failure_videos/
```

### README 引用规范

数学公式 ↔ 函数的引用形式（混合 α + β）：

- README 中每个 loss 与 sampler 公式后给出函数级引用，如 `src/cfm.py::cfm_loss`。
- 关键 commit message 中包含数学说明（commit-level 引用作为可产物，git blame 时可见）。

### Commit 策略

Feature-based。每个 commit 对应一个功能闭环：

1. Repo skeleton + dependencies
2. LIBERO env wrapper + HDF5 dataloader
3. R3M cache + offline augment script
4. 1D U-Net + FiLM
5. CFM loss + Euler solver + unit tests
6. Training loop + EMA + early stopping
7. Eval runner + video saving
8. ODE ablation runner
9. DP head + DDIM sampler + unit tests
10. Native continuation interface
11. README + results aggregation

预计 10–12 个 commit。

---

## 8. 时间线

| Day | 目标 |
|---|---|
| 5 | 计划锁定，启动 env setup（LIBERO 安装、headless 渲染依赖、R3M 加载验证）|
| 5–7 | Repo skeleton，HDF5 dataloader，R3M 缓存脚本跑通，单任务无增强的预处理完成 |
| 7–9 | 1D U-Net + FiLM，CFM loss + Euler，单元测试通过，单任务 1 seed 端到端跑通 |
| 9–11 | 训练 loop + EMA + 早停，eval runner，单任务出第一个 success rate |
| 11–14 | 至少 1 任务 success rate 入邮件，目标 75–85%。Repo 含完整 README、CFM 实现、单元测试、eval video |
| 14 | 邮件发出 |
| 14–18 | 5 任务 × 3 种子 CFM 完成 |
| 18–21 | DP head replacement 实现，DP 5 任务 × 3 种子启动 |
| 21–25 | DP 完成，ODE ablation eval，技术 note 写作 |
| 25–28 | README 定稿，失败 video 整理，给博士生的技术问题邮件 |

---

## 9. 资源预算

### GPU

| 阶段 | GPU-hr |
|---|---|
| CFM 5 task × 3 seed | ~28 |
| DP 5 task × 3 seed | ~28 |
| ODE ablation eval | ~4 |
| Pipeline 调试与 retry | ~10 |
| **合计** | **~70** |

云端 4090 × 2，wall-clock 约 18–20 天（与时间线匹配）。

### 磁盘（50 GB 上限）

| 项 | 占用 |
|---|---|
| LIBERO Goal HDF5 | ~4 GB |
| R3M checkpoint | ~340 MB |
| R3M 特征缓存（$K=8$）| ~5 GB |
| CFM checkpoint（best + last × 15 run）| ~1.5 GB |
| DP checkpoint（best + last × 15 run）| ~1.5 GB |
| 失败 video | ~2 GB |
| Conda env | ~8 GB |
| 临时文件（tensorboard 等）| ~2 GB |
| **小计** | **~24 GB** |
| **缓冲** | **~26 GB** |

---

## 10. Native Continuation 接口预留

`src/infer.py` 暴露：

```python
def rollout(
    policy,
    env,
    chunk_init_strategy: Literal["gaussian", "native_continuation"] = "gaussian",
    ode_steps: int = 4,
):
    ...
```

`"gaussian"` 为本工作默认。`"native_continuation"` 在内部 raise `NotImplementedError("Reserved for future work; see Native Continuation paper §X")`，README 中说明该接口的存在与未来扩展计划。

---

## 11. 风险与不收敛回退

LIBERO 跑不通的具体风险：

- Headless 云端 EGL/OSMesa 依赖：Day 5–6 验证。
- R3M 与当前 PyTorch 版本兼容性：Day 5 验证。
- LIBERO 仿真速度（每 step ~50–100 ms）：影响 eval wall-clock，不影响训练。

应对：技术问题逐个解决，不切换到 C1。

不收敛诊断顺序（如 100 epoch loss 未明显下降）：

1. 单元测试是否通过（CFM loss、Euler solver）。
2. R3M 特征 norm 是否合理（不应过大或过小）。
3. Action 归一化后的统计量（mean、std、range）。
4. Batch 内 $\tau$ 分布是否覆盖 $[0, 1]$。
5. Velocity field 输出范围是否与 $a_1 - a_0$ 量级匹配。
6. EMA 权重与 online 权重的 success rate 是否都低（若 online 高 EMA 低，是 EMA decay 与 epoch 数不匹配）。
7. Learning rate scale。

每步与 Claude 交互形式：现象描述 + 拐杖排查。

---

## 12. 与本计划相关的协作约定

- 架构变动：先做交互讨论，确认后再改。
- 超参调整：Claude 给推荐区间，决策由开发者做。
- 代码：Claude 可写局部功能性代码（FiLM block、Euler solver、单元测试断言），不写完整训练脚本与 eval runner。
- Debug：现象描述 + Claude 协助拐杖排查。

---

## 13. 范围与未来扩展

### 13.1 范围内

- 训练目标：CFM
- 视觉编码器：R3M frozen（vision-only）
- 条件注入：FiLM
- 任务设定：每任务一个 specialist policy
- 对科：DP head replacement
- ODE 步数 ablation
- Native Continuation 接口签名（不评价）

### 13.2 范围外

- 自然语言任务指令
- VLM backbone（PaliGemma / SigLIP）
- Token 级 cross-attention conditioning
- Generalist policy（一个模型处理多任务）
- 跨 LIBERO suite 泛化（仅 LIBERO-Goal，不含 Object / Spatial / Long）
- Native Continuation 完整实现与评价

### 13.3 向 π₀ 方向扩展的最小路径

下面三步按工程复杂度递增。每一步独立有意，无需全做。

1. **加语言条件**（约 0.5 天代码 + 1 次重训 ~10 GPU-hr）
   CLIP text encoder frozen，编码任务描述得 $e_\text{lang} \in \mathbb{R}^{512}$，拼入观测条件 $o$。FiLM 注入机制不变。15 个 specialist 模型缩为 3 seed × 1 generalist 模型。这是从 task-specialized 走向 multi-task 的最低成本路径。

2. **换视觉编码器为 V-L 模型**（约 1 天代码 + 1 次重训 ~10 GPU-hr）
   R3M → SigLIP（PaliGemma 视觉部分）或 CLIP image encoder，仍 frozen。视觉表征质量提升。FiLM 注入机制不变。

3. **改 action 网络为 transformer + cross-attention**（约 3–5 天代码 + 1 次重训 ~15 GPU-hr）
   1D U-Net → action transformer。Action chunk 作 query，VLM 输出 token 序列作 key/value。这是 π₀ minimal 复刻，但会破坏与 DP 的 head replacement 对种性，应作为独立 follow-up 项目而非本仓库的内部扩展。

### 13.4 向 Native Continuation 方向扩展

读完 Native Continuation 论文后，在 `infer.py` 中实现 `chunk_init_strategy = "native_continuation"` 分支。代码量约 50 行。Ablation 设计：5 任务 × 3 种子 × 2 strategy × 20 episodes = 600 episodes，eval 约 2 hr。该扩展不需重训，直接在已有 checkpoint 上做。

---

## 附：关键数学形式速览（供 reviewer 快速定位）

CFM 训练 loss（§3.4）→ `src/cfm.py::cfm_loss`

DDPM 训练 loss（§3.5）→ `src/dp.py::dp_loss`

CFM Euler 推理（§3.6）→ `src/cfm.py::euler_sample`

DDIM 推理（§3.6）→ `src/dp.py::ddim_sample`

FiLM ResBlock（§3.2）→ `src/models.py::FiLMResBlock1D`

R3M 特征缓存（§4.4）→ `scripts/precompute_r3m.py`

ODE 步数 ablation（§6）→ `scripts/ablation_ode.py`
