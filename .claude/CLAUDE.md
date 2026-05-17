# FlowPolicy-Libero（C2 Reproduction Repo）

## Identity
LIBERO-Goal 5-task subset，C2 reproduction，与 `plan/c2_plan.md`（设计层）+ `plan/c2_coding_plan.md`（代码实施层）配对。

## Environment（硬路径）
- conda env: `/root/shared-nvme/envs/flow-policy`（Python 3.9.25, torch 2.8.0+cu128）
- LIBERO repo: `/root/shared-nvme/LIBERO/`
- Dataset: `/root/shared-nvme/data/libero_goal/`
- R3M checkpoint: `~/.r3m/r3m_50/`

## Output Format（所有 subagent 受约束）
- 公式用 LaTeX，不用 plain text。
- 禁类比、禁跨域术语借用、禁口语包装（如 "清版本""含噪样本"）。用 clean sample / noisy sample / clean trajectory 或论文原术语。
- 禁 meta 旁白（如 "the key point is""note that""这一条最关键"）。完成任务立即停止。
- 禁 "而不是 X" 式纠错尾巴。直接陈述应做之事。
- 用户复述内容核心正确但不精确时，只输出精确版本，不解释修正。
- 事实不存在时直接说 "没有" 或 "错了"，禁找替代品。

## Immutable Files
不可改：`plan/c2_plan.md`、`plan/c2_coding_plan.md`、`plan/reading_plan.md`、`.claude/CLAUDE.md`、`.claude/agents/*.md`。任何 subagent 提议改 → 必须 raise，等用户确认。

## Pseudocode-as-Spec
伪代码是 spec。代码与伪代码不一致时以伪代码为准——要么改伪代码、要么改代码回对齐，禁 "代码已修但伪代码未更新" 的并存状态。任何 subagent 检测到漂移必须 raise，禁默认按代码补伪代码或按伪代码静默改代码。

伪代码合格标准（来自 plan/c2_coding_plan.md）：交给 executor 转化后输出在用户预期之内，没有任何"原来是这样"的意外。reviewer 审查的根本目标即此。

## Two Decision Paths（来自 plan/c2_coding_plan.md）

### 工程实现卡住（"这个 API 怎么用"）
来源：官方文档 + LLM 直接答。
具体指向：PyTorch API、LIBERO 环境 API、R3M 加载与推理、h5py 读取。
路径：reviewer / executor 可基于自身知识或官方文档直接答，不走决策流程。

### 设计决策卡住（"该选 A 还是 B"）
来源顺序：官方实现 → 微观实验 → 讨论。
路径：reviewer 严格按此顺序，禁跳步、禁直接给推荐。详见 reviewer.md。

### 数学推导
LLM 不可用。所有数学推导（如 `tests/test_cfm_loss.py` 的 closed-form 参考值）由用户手算。reviewer 可指出推导思路或验证用户推导结果，但禁替用户产出推导链。

## Outsourced vs High-Value Files
"完全外包"文件（见 File Ownership 表）：无伪代码。executor 的 spec = 用户当次贴入的 plan/c2_coding_plan.md §"功能" 段落原文。
"高价值代码"文件：executor 的 spec = 用户审查通过的伪代码。

## Decisions Log
所有讨论中达成的决策、与 c2_plan 锁定层的差异、微观实验结果，由用户记录在 `notes/decisions.md`。reviewer 与 debugger 在调用时若需查阅历史决策，读此文件。

## File Ownership

| 文件 | 类别 | 决策经验来源 |
|---|---|---|
| src/data.py | 高价值代码：sliding window padding、action normalize / denormalize、augment dispatch、dataset indexing 逻辑、eval-time obs 构造函数。外包代码：HDF5 文件遍历与原始数组读取、R3M cache mmap 加载器、DataLoader 实例化（batch_size / num_workers / pin_memory 从 config 透传）、collate 函数（默认 PyTorch stack 即可，除非 variable-length demo 需要特殊处理）。 | Padding 规则、归一化策略 → Diffusion Policy 官方 repo `diffusion_policy/dataset/` 与 robomimic `normalize_util.py`，查标准做法后决定是否跟进。Gripper 维度分布 → 加 3 行代码对 LIBERO-Goal 一个 task 的 actions 第 7 维做 histogram，直接看数据。Cache schema → 微观实验（per-demo 文件方案下 random batch I/O 延迟 vs 单大 array mmap 延迟，各跑 10 batch 计时）。 |
| src/model/modules.py | 高价值代码：FiLMResBlock1D（含 FiLM affine、GroupNorm、残差连接）。外包代码：sinusoidal_embedding 函数（标准公式）、Downsample1d / Upsample1d 的 conv 配置。生疏内容：GroupNorm 的 `num_groups` 参数语义、`nn.Conv1d` 的 padding mode。 | γ 初始化 → π₀ 与 DP 官方实现中的具体选择。 |
| src/model/unet1d.py | 高价值代码：ConditionalUnet1D 主类 forward、condition MLP（time_mlp(sinusoidal(t)) ⊕ obs → c）、encoder / bottleneck / decoder 拼接顺序与 skip 消费规则。外包代码：final Conv1dBlock + 1×1 Conv1d 输出头。生疏内容：DP 官方 `ConditionalUnet1D` 的 3-skip / 2-pop 不对称结构。 | Time embedding max period → Diffusion Policy 官方 repo 中的值；CFM / DP 共享同一 embedding 时需确认 DP 把 t/T 归一化到 [0,1] 后与 CFM 的 τ ∈ [0,1] 走同一个 embedding 是否合理。 |
| src/model/ObsEncoder.py | 高价值代码：ObsEncoder forward（8210 raw concat → 8 token 切分 → per-modality 投影 + step_emb + modality_emb → PreNormBlock × num_blocks → final LayerNorm → flatten 为 (B, 8·dim)）、PreNormBlock（LN + MHA 残差，LN + Linear-SiLU-Linear 残差）。外包代码：`nn.MultiheadAttention` 调用细节、`nn.Embedding` 默认初始化。生疏内容：`batch_first=True` 下 MHA 的 q/k/v 形状语义。 | 结构、超参（dim=256 / num_blocks=3 / num_heads=4 / mlp_ratio=4）、per-modality 权重共享策略（cam1 / cam2 独立，prev / curr 跨步共享）、step_emb / modality_emb 初始化 → 已在 notes/decisions.md 2026-05-13 第三行锁定。 |
| src/cfm.py | 高价值代码：整个文件。`cfm_loss`（τ 采样、a₀ 采样、插值 a_τ、target 构造、MSE reduction）与 `euler_sample`（初始化、N 步迭代、τ_n 取值、输出处理）。外包代码：无。 | τ 端点 → Lipman et al. (2023) 原论文与官方实现；π₀ 的 logit-normal 采样策略（作为参考，C2 不采用）。Clip 策略 → 不加 clip 跑一个 task 的 `euler_sample` 输出 range，若频繁超出 [-1,1] 再加。微观实验优于文献。 |
| src/dp.py | 高价值代码：无。DP 是对比组，设计标准是"与 DP 官方实现严格一致"，独创性是负面贡献。外包代码：整个文件。生疏内容：squared cosine schedule 的公式推导（IDDPM 论文 §3.2）、DDIM 的 reverse process 推导（DDIM 论文 §4.1）。理解这两项使你在 head replacement 对比中能解释 DP 端的行为。 | 全部从 Diffusion Policy 官方 repo 复制选择。η、schedule 类型、T 值不另改动。 |
| src/train.py | 高价值代码：训练主循环中 loss 调用 → backward → grad clip → EMA update → 早停 state machine 的逻辑流。外包代码：config 加载、CLI 参数解析、optimizer / scheduler 实例化、checkpoint I/O、logging 接线、bf16 `GradScaler` 管理。易解决项：cosine annealing with warmup 的 `LambdaLR` 写法。 | EMA fp32 累计 → PyTorch EMA 标准实现（`torch.optim.swa_utils.AveragedModel` 或手写 shadow copy），查 DP 官方 repo 的写法。早停规则 → 已锁定（见 notes/decisions.md row 23–25 / 27 / 29 / 34）；首轮训练后若需调整，改 cfg `early_stop.*` 字段。 |
| src/eval.py | 高价值代码：episode rollout 循环（obs 构造 → 推理调用 → action 反归一化 → receding horizon 执行 → done/success 判定）。这段是 train 与 eval 的 data flow 对齐点。外包代码：mp4 保存逻辑、CSV 聚合、checkpoint 加载 I/O、LIBERO 环境 reset/step wrapper。生疏内容：LIBERO 环境 API（`env.reset()`、`env.step(action)` 返回值结构、`info` dict 内容）。 | Episode 最大步数 → LIBERO 官方 benchmark 的设定，查 LIBERO repo 的 eval 脚本。Success 判定 → LIBERO 官方 eval 协议。通常直接信任 `info['success']`，但先跑 5 个 episode 人工确认 flag 与视觉结果一致。 |
| src/infer.py | 高价值代码：无。外包代码：整个文件。易解决项：receding horizon 计数器逻辑（chunk 内已执行 k 步后触发重规划）。 | 职责边界 → 无文献参考，纯工程判断。倾向：`infer.py` 只做单次 obs → action chunk 映射，receding horizon 计数器放 `eval.py`，使 `infer.py` 无状态。可单独测试。 |
| scripts/precompute_r3m.py | 高价值代码：无。外包代码：整个文件。R3M 加载、增强 pipeline、batch 推理、存盘 I/O 都为标准操作。易解决项：R3M 的输入预处理要求（ImageNet normalize、resize 顺序）、torchvision transforms 的随机种子控制方式。 | 存盘格式 → 与 `data.py` 统一决策，见 `data.py` 来源。种子策略 → 固定种子是 reproducibility 的默认选择，除非有明确理由不固定。 |
| scripts/run_all.sh | 完全外包。 | 无。 |
| scripts/ablation_ode.py | 完全外包。是 `eval.py` 的 thin wrapper。 | 无。 |
| tests/test_cfm_loss.py | 高价值代码（你设计测试用例）：测试用例选择、解析式推导、tolerance 设定、边界情况枚举。外包代码：pytest 框架语法。生疏内容：pytest 的 fixture / parametrize / approx 用法。 | 解析式推导 → 自查文献。$a_1 \sim \mathcal{N}(\mu, \sigma^2)$，$a_0 \sim \mathcal{N}(0, 1)$ 时 $\mathbb{E}\|v^* - (a_1 - a_0)\|^2$ 在最优 $v^*$ 下的值。 |

## Per-File Decision Checklist
每个文件 plan/c2_coding_plan.md §"待决策项" 的原文枚举。reviewer 审查伪代码时对照此清单，确认每一项在伪代码中已显式给出答案。executor 看到伪代码未覆盖此清单中任一项时 raise。

| 文件 | 待决策项原文 |
|---|---|
| src/data.py | 1. Sliding window padding 规则：首帧（t < T_o）复制首帧 obs 还是 zero observation；末端（t + H − 1 > T）hold pose 还是 zero velocity。<br>2. Action 归一化：per-dimension 还是 global min-max；gripper 维度是否与 6-DoF joint 同等处理；反归一化后是否 clip 到 [-1, 1]；统计量存在哪（checkpoint 内嵌 vs 单独文件）。<br>3. R3M cache 存盘格式与文件布局（与 `scripts/precompute_r3m.py` 联合决策）：单大 array per task 还是 per-demo 文件。<br>4. Augment dispatch 粒度：per-sample（每次 `__getitem__` 内抽 aug_idx）还是 per-epoch。<br>5. Dataset indexing 语义：长度 = $\sum_d T_d$ 还是 $\sum_d (T_d - H + 1)$；`__getitem__` 是否携带 demo_id 与 step 元数据。<br>6. Eval-time obs 构造路径与 train-time 的一致性保证方案。 |
| src/model/modules.py | 1. FiLM γ 初始化选择。 |
| src/model/unet1d.py | 1. Time embedding 是否 CFM / DP 共享；涉及 max period 选择。 |
| src/model/ObsEncoder.py | 无未决项；结构与超参已在 notes/decisions.md 2026-05-13 第三行锁定。 |
| src/cfm.py | 1. τ 采样是否排除 τ = 0 端点（$\mathcal{U}[0,1)$ vs $\mathcal{U}[\epsilon, 1)$）。<br>2. `euler_sample` 推理输出是否 clip 到 $[-1,1]^{H \times d_a}$。<br>3. `cfm_loss` 返回 scalar 还是 per-sample $(B,)$。<br>4. `euler_sample` 是否支持 batched sampling。 |
| src/dp.py | 1. $\bar\alpha_t$ 序列预计算 `register_buffer` 还是动态索引。<br>2. DDIM η 取值（η = 0 确定性 vs η > 0 随机）。<br>3. `dp_loss` 返回 scalar 还是 per-sample $(B,)$（与 `cfm_loss` 保持一致）。<br>4. DDIM `ddim_sample` `timesteps` 子采样策略（caller 责任，长度 $= T_{\text{infer}}$，严格降序）：HuggingFace `DDIMScheduler` leading / trailing / linspace / random subsample / 全量降序。 |
| src/train.py | 1. EMA 在 bf16 训练下是否用 fp32 累计。<br>2. 早停中"不下降"的判定：严格大于还是大于等于。<br>3. Warmup 期间（前 500 步）是否启用 EMA update。<br>4. Best checkpoint 选择依据：训练中 evaluate success rate 还是 training loss。 |
| src/eval.py | 1. Eval-time obs 构造与 train-time 共用一个函数（`data.py` 给统一接口）还是 eval 独立实现。<br>2. Episode 最大步数上限（如 300 或 400）。<br>3. Success 判定：直接信任 LIBERO `info['success']` 还是额外 sanity check。<br>4. 训练中 evaluate 与训练后 evaluate 的起始状态种子是否重叠。 |
| src/infer.py | 1. `rollout` 与 `eval.py` 的职责边界：`infer.py` 管半 episode step-level 控制流，还是只做单次 obs → action chunk 映射。<br>2. Native continuation 预留接口是否定义输入签名（上一个 chunk 末端 ODE 状态），还是仅 raise。 |
| scripts/precompute_r3m.py | 1. 存盘格式与文件布局（与 `data.py` 联合决策，见 `data.py` 第 3 项）。<br>2. K=8 增强的随机种子策略：固定种子（可复现）还是纯随机。 |
| tests/test_cfm_loss.py | 1. 解析参考值推导（1D Gaussian 下 CFM loss 的 closed-form，数学推导）。<br>2. 测试用例设计：选什么分布、tolerance 取多少、哪些边界情况要覆盖。 |

## Subagent Routing
- `@reviewer`：默认 agent，处理讨论与审查。
- `@executor`：显式调用。pseudocode → PyTorch（或外包文件的功能段 → PyTorch）。
- `@debugger`：显式调用。训练/eval 异常诊断。
不要靠 description 自动路由 executor 或 debugger，避免讨论中途误触发。
