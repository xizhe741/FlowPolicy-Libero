# C2 Coding Plan

> 本文件是 c2_plan.md 的代码实施层。c2_plan 锁定了"做什么"与"用什么参数"，本文件锁定"每个文件谁写、写之前要做什么决策、卡住了去哪里找答案"。

---

## 工作流

每个文件按以下顺序推进：

1. 确认已锁定层（c2_plan 已规定 + 抽象目标合法）：不重新决定。
2. 对待决策层选择：与 Claude 讨论。
3. 你写伪代码（自包含所有决策）。
4. Claude 审查伪代码，提出隐含未决项。
5. 修订完成后交 Claude Code 转 PyTorch。
6. 你 review Claude Code 输出，确认与伪代码一致后 commit。

伪代码合格标准：交给 Claude Code 转化后，输出在你预期之内，没有任何"原来是这样"的意外。如果有意外，说明伪代码漏掉了决策点。

---

## 文件清单

### `src/data.py`

**功能**：LIBERO-Goal 训练数据 pipeline。HDF5 demo → sliding window → action 归一化 → R3M cache 加载 → augment dispatch → DataLoader batch。附 eval-time obs 构造路径（R3M 在线推理，不走 cache）。

**待决策项**：

1. Sliding window padding 规则：首帧（$t < T_o$）复制首帧 obs 还是 zero observation；末端（$t + H - 1 > T$）hold pose 还是 zero velocity。
2. Action 归一化：per-dimension 还是 global min-max；gripper 维度是否与 6-DoF joint 同等处理；反归一化后是否 clip 到 $[-1, 1]$；统计量存哪（checkpoint 内嵌 vs 单独文件）。
3. R3M cache 存盘格式与文件布局（与 `scripts/precompute_r3m.py` 联合决策）：单大 array per task 还是 per-demo 文件。
4. Augment dispatch 粒度：per-sample（每次 `__getitem__` 内抽 aug_idx）还是 per-epoch。
5. Dataset indexing 语义：长度 = $\sum_d T_d$ 还是 $\sum_d (T_d - H + 1)$；`__getitem__` 是否携带 demo_id 与 step 元数据。
6. Eval-time obs 构造路径与 train-time 的一致性保证方案。

**高价值代码（你写伪代码）**：sliding window padding、action normalize / denormalize、augment dispatch、dataset indexing 逻辑、eval-time obs 构造函数。

**外包代码**：

- 完全外包：HDF5 文件遍历与原始数组读取。R3M cache mmap 加载器。DataLoader 实例化（batch_size / num_workers / pin_memory 从 config 透传）。
- 易解决可：collate 函数（默认 PyTorch stack 即可，除非 variable-length demo 需要特殊处理）。

**决策经验来源**：

- Padding 规则、归一化策略 → Diffusion Policy 官方 repo `diffusion_policy/dataset/` 与 robomimic `normalize_util.py`，查标准做法后决定是否跟进。
- Gripper 维度分布 → 加 3 行代码对 LIBERO-Goal 一个 task 的 actions 第 7 维做 histogram，直接看数据。
- Cache schema → 微观实验（per-demo 文件方案下 random batch I/O 延迟 vs 单大 array mmap 延迟，各跑 10 batch 计时）。

---

### `src/model/modules.py`

**功能**：1D U-Net 子模块。`sinusoidal_embedding`（time → 位置编码）、`FiLMResBlock1D`（Conv→GN→Mish → FiLM scale/bias → Conv→GN→Mish + residual）、`Downsample1d`、`Upsample1d`。

**待决策项**：

1. FiLM $\gamma$ 初始化选择。

**高价值代码（你写伪代码）**：`FiLMResBlock1D`（含 FiLM affine、GroupNorm、残差连接）。

**外包代码**：

- 完全外包：sinusoidal embedding 函数（标准公式）。上采样 / 下采样模块的 conv 配置。
- 生疏内容：GroupNorm 的 `num_groups` 参数语义。`nn.Conv1d` 的 padding mode。

**决策经验来源**：

- $\gamma$ 初始化 → π₀ 与 DP 官方实现中的具体选择。

---

### `src/model/unet1d.py`

**功能**：`ConditionalUnet1D`，CFM 与 DP 共享的 1D Temporal U-Net backbone。time_mlp(sinusoidal(t)) ⊕ obs → condition；3-level encoder + bottleneck + 2-level decoder（skip[0] 不消费，DP 一致）→ $v_\theta \in \mathbb{R}^{B \times H \times d_a}$。

**待决策项**：

1. Time embedding 是否 CFM / DP 共享；涉及 max period 选择。

**高价值代码（你写伪代码）**：U-Net 主类 forward。condition MLP（time embedding + obs → $c$）。

**外包代码**：

- 完全外包：encoder/decoder/bottleneck 的 FiLMResBlock1D 堆叠 pattern；skip connection concat；final Conv1dBlock + 1×1 Conv1d。
- 生疏内容：DP 官方 `ConditionalUnet1D` 的 skip 数与 Upsample/Identity 配比。

**决策经验来源**：

- Time embedding max period → Diffusion Policy 官方 repo 中的值；如果 CFM / DP 共享，需确认 DP 把 $t/T$ 归一化到 $[0,1]$ 后与 CFM 的 $\tau \in [0,1]$ 走同一个 embedding 是否合理。

---

### `src/model/ObsEncoder.py`

<!-- TODO: 用户自加。参考 notes/decisions.md 2026-05-13 第三行记录。 -->

---

### `src/cfm.py`

**功能**：CFM 训练 loss 与 Euler ODE 推理。两个函数：`cfm_loss` 与 `euler_sample`。

**待决策项**：

1. $\tau$ 采样是否排除 $\tau = 0$ 端点（$\mathcal{U}[0,1)$ vs $\mathcal{U}[\epsilon, 1)$）。
2. `euler_sample` 推理输出是否 clip 到 $[-1,1]^{H \times d_a}$。
3. `cfm_loss` 返回 scalar 还是 per-sample $(B,)$。
4. `euler_sample` 是否支持 batched sampling。

**高价值代码（你写伪代码）**：整个文件。`cfm_loss`（$\tau$ 采样、$a_0$ 采样、插值 $a_\tau$、target 构造、MSE reduction）与 `euler_sample`（初始化、$N$ 步迭代、$\tau_n$ 取值、输出处理）。

**外包代码**：无。

**决策经验来源**：

- $\tau$ 端点 → Lipman et al. (2023) 原论文与官方实现；π₀ 的 logit-normal 采样策略（作为参考，C2 不采用）。
- Clip 策略 → 不加 clip 跑一个 task 的 `euler_sample` 输出 range，若频繁超出 $[-1,1]$ 再加。微观实验优于文献。

---

### `src/dp.py`

**功能**：DP 对比组的 $\epsilon$-prediction DDPM loss、squared cosine noise schedule、DDIM sampler。与 `src/cfm.py` 对称。

**待决策项**：

1. $\bar\alpha_t$ 存储方案：(a) `register_buffer` 嵌入 `dp.py` 模型类、(b) 动态索引在 forward 内计算、(c) `dp.py` 出 `squared_cosine_schedule(T)` 工具函数，`train.py` 持有 tensor 并作为 `dp_loss` / sampler 形参传入。
2. DDIM $\eta$ 取值（$\eta = 0$ 确定性 vs $\eta > 0$ 随机）。
3. `dp_loss` 返回 scalar 还是 per-sample $(B,)$（与 `cfm_loss` 保持一致）。
4. DDIM `ddim_sample` `timesteps` 子采样策略：`ddim_sample` 由 caller 传入严格降序整数序列，长度 $= T_{\text{infer}}$。caller 须在 (a) HuggingFace `DDIMScheduler` leading spacing / (b) trailing spacing / (c) linspace / (d) random subsample / (e) 全量降序 中选定。

**高价值代码（你写伪代码）**：`dp_loss` 函数体。理由：与 `cfm_loss` 对称的 return-format、model-signature、$t$ 归一化约定需在伪代码层显式锁定，避免外包实现偏离对称性。`squared_cosine_schedule` 与 DDIM sampler 仍外包。

**外包代码**：

- 完全外包：`squared_cosine_schedule` 函数（IDDPM §3.2 标准公式）、DDIM sampler。
- 生疏内容：squared cosine schedule 的公式推导（IDDPM 论文 §3.2），DDIM 的 reverse process 推导（DDIM 论文 §4.1）。理解这两项使你在 head replacement 对比中能解释 DP 端的行为。

**决策经验来源**：

- `squared_cosine_schedule` 与 DDIM sampler：从 Diffusion Policy 官方 repo 复制选择。$\eta$、schedule 类型、$T$ 值不另改动。
- `dp_loss` 对称约定（return scalar、$t/T$ 归一化、`alpha_bar` 形参传入）：与 `cfm_loss` 联合决策，记入 `notes/decisions.md`。

---

### `src/train.py`

**功能**：训练入口。config 加载 → model / optimizer / scheduler 实例化 → 训练主循环（forward → loss → backward → grad clip → EMA update）→ 训练中 evaluate 触发 → 早停 → checkpoint 保存。

**待决策项**：

1. EMA 在 bf16 训练下是否用 fp32 累计。
2. 早停中"不下降"的判定：严格大于还是大于等于。
3. Warmup 期间（前 500 步）是否启用 EMA update。
4. Best checkpoint 选择依据：训练中 evaluate success rate 还是 training loss。

**高价值代码（你写伪代码）**：训练主循环中 loss 调用 → backward → grad clip → EMA update → 早停 state machine 的逻辑流。

**外包代码**：

- 完全外包：config 加载、CLI 参数解析、optimizer / scheduler 实例化、checkpoint I/O、logging 接线、bf16 `GradScaler` 管理。
- 易解决可：cosine annealing with warmup 的 `LambdaLR` 写法。

**决策经验来源**：

- EMA fp32 累计 → PyTorch EMA 标准实现（`torch.optim.swa_utils.AveragedModel` 或手写 shadow copy），查 DP 官方 repo 的写法。
- 早停规则 → 已锁定（见 `notes/decisions.md` row 23–25 / 27 / 29 / 34）；首轮训练后若需调整，改 cfg `early_stop.*` 字段。

---

### `src/eval.py`

**功能**：评估入口。加载 EMA checkpoint → LIBERO 环境实例化 → episode rollout（obs 构造 → 推理 → 反归一化 → receding horizon 执行 → success 判定）→ metric 聚合 → 失败 video 保存。

**待决策项**：

1. Eval-time obs 构造与 train-time 共用一个函数（`data.py` 给统一接口）还是 eval 独立实现。
2. Episode 最大步数上限（如 300 或 400）。
3. Success 判定：调用 `env.check_success()` 还是额外 sanity check。（本机 LIBERO `env.step` 返回 info 为空 dict，原 `info['success']` 路径已于 2026-05-18 GT replay 证伪，改用 `env.check_success()`）
4. 训练中 evaluate 与训练后 evaluate 的起始状态种子是否重叠。

**高价值代码（你写伪代码）**：episode rollout 循环（obs 构造 → 推理调用 → action 反归一化 → receding horizon 执行 → done/success 判定）。这段是 train 与 eval 的 data flow 对齐点。

**外包代码**：

- 完全外包：mp4 保存逻辑、CSV 聚合、checkpoint 加载 I/O、LIBERO 环境 reset/step wrapper。
- 生疏内容：LIBERO 环境 API（`env.reset()`、`env.step(action)` 返回值结构、`info` dict 内容）。

**决策经验来源**：

- Episode 最大步数 → LIBERO 官方 benchmark 的设定，查 LIBERO repo 的 eval 脚本。
- Success 判定 → LIBERO 官方 eval 协议。2026-05-18 GT replay 实测：本机 LIBERO `env.step` 返回 info 为空 dict，正确出口是 `env.check_success()`；replay demo_0 全 170 步末尾 `env.check_success() = True`，确认 BDDL goal predicate 链路健康。

---

### `src/infer.py`

**功能**：推理接口。封装 obs → action chunk 的完整前向路径，供 `eval.py` 调用。Receding horizon 状态管理。`chunk_init_strategy` 参数预留 native continuation。

**待决策项**：

1. `rollout` 与 `eval.py` 的职责边界：`infer.py` 管半 episode step-level 控制流，还是只做单次 obs → action chunk 映射。
2. Native continuation 预留接口是否定义输入签名（上一个 chunk 末端 ODE 状态），还是仅 raise。

**高价值代码**：无。

**外包代码**：

- 完全外包：整个文件。待决策项确定后实现是机械粘贴。
- 易解决可：receding horizon 计数器逻辑（chunk 内已执行 $k$ 步后触发重规划）。

**决策经验来源**：

- 职责边界 → 无文献参考。纯工程判断。倾向：`infer.py` 只做单次 obs → action chunk 映射，receding horizon 计数器放 `eval.py`，使 `infer.py` 无状态。可单独测试。

---

### `scripts/precompute_r3m.py`

**功能**：离线预处理。每张训练图像 × $K=8$ 份增强 → R3M frozen fp16 → 2048 维特征存盘。

**待决策项**：

1. 存盘格式与文件布局（与 `data.py` 联合决策，见 `data.py` 第 3 项）。
2. $K=8$ 增强的随机种子策略：固定种子（可复现）还是纯随机。

**高价值代码**：无。

**外包代码**：

- 完全外包：整个文件。R3M 加载、增强 pipeline、batch 推理、存盘 I/O 都为标准操作。
- 易解决可：R3M 的输入预处理要求（ImageNet normalize、resize 顺序）、torchvision transforms 的随机种子控制方式。

**决策经验来源**：

- 存盘格式 → 与 `data.py` 统一决策，见 `data.py` 来源。
- 种子策略 → 固定种子是 reproducibility 的默认选择，除非有明确理由不固定。

---

### `scripts/run_all.sh`

**功能**：调度脚本。5 任务 × 3 种子 × 2 方法的训练与 eval 串行/并行调度，GPU 分配。

**待决策项**：无。

**外包代码**：完全外包。

---

### `scripts/ablation_ode.py`

**功能**：ODE 步数扫描。对 CFM checkpoint 在 $N \in \{1, 4\}$ 上各跑 full eval，输出 success rate 表格。

**待决策项**：无。

**外包代码**：完全外包。是 `eval.py` 的 thin wrapper。

---

### `tests/test_cfm_loss.py`

**功能**：CFM loss 单元测试。在 1D 高斯目标分布上验证 `cfm_loss` 实现与解析 closed-form 一致。

**待决策项**：

1. 解析参考值推导（1D Gaussian 下 CFM loss 的 closed-form，数学推导）。
2. 测试用例设计：选什么分布、tolerance 取多少、哪些边界情况要覆盖。

**高价值代码（你设计测试用例）**：测试用例选择、解析式推导、tolerance 设定、边界情况枚举。

**外包代码**：

- 完全外包：pytest 框架语法。
- 生疏内容：pytest 的 fixture / parametrize / approx 用法。

**决策经验来源**：

- 解析式推导 → 自查文献。$a_1 \sim \mathcal{N}(\mu, \sigma^2)$，$a_0 \sim \mathcal{N}(0, 1)$ 时 $\mathbb{E}\|v^* - (a_1 - a_0)\|^2$ 在最优 $v^*$ 下的值。

---

### `tests/test_euler_solver.py`

**功能**：Euler solver 单元测试。用线性 ODE $\dot{x} = -x$ 的解析解验证 `euler_sample` 的离散化正确性与收敛阶。

**待决策项**：

1. 测试用例设计：ODE 选择、误差 tolerance、收敛阶验证方法。

**高价值代码（你设计测试用例）**：同上。

**外包代码**：

- 完全外包：pytest 框架语法。

**决策经验来源**：

- 收敛阶验证 → 数值分析教科书（Euler 一阶，误差 $\sim O(h)$，$h = 1/N$）。用 $N$ 与 $2N$ 的误差比是否接近 2 来验证。

---

### `configs/*.yaml`

**功能**：`cfm_default.yaml`、`dp_default.yaml`、`ablation_ode_steps.yaml` 三个配置文件。

**待决策项**：无。c2_plan §5 已规定全部超参数值。

**外包代码**：完全外包。

---

## 卡住时的查询策略

按问题类型选择查询来源，不混用。

### 数学推导卡住

来源：教科书与论文原文。

具体指向：

- CFM loss 的解析性质 → Lipman et al. (2023) §3，特别是 Theorem 1 的证明。
- Euler 离散化误差分析 → 任意数值分析教科书（Atkinson, Burden & Faires 等）的 ODE 初值问题章节。
- DDPM / DDIM 推导 → Ho et al. (2020) §3, Song et al. (2021) §4。
- FiLM conditioning 的表达力分析 → Perez et al. (2018) FiLM 原论文。

不用 LLM 查：LLM 在推导中容易给出"看起来合理但符号不一致"的中间步骤，对数学推导引入隐蔽错误。

### 工程实现卡住（"这个 API 怎么用"）

来源：官方文档 + LLM。

具体指向：

- PyTorch API（`nn.Conv1d` padding、`GradScaler` 用法、`register_buffer` 语义）→ PyTorch 官方文档 + Claude 查询。
- LIBERO 环境 API → LIBERO GitHub repo 的 README 与源码。
- R3M 加载与推理 → R3M GitHub repo。
- HDF5 读写 → h5py 官方文档。

LLM 在这类问题上可靠：API 用法是 pattern matching，不涉及推导。

### 设计决策卡住（"该选 A 还是 B"）

来源：官方实现 → 微观实验 → Claude 讨论。按此顺序，不跳步。

流程：

1. 查 Diffusion Policy / π₀ / robomimic 官方 repo 的对应位置，确定标准做法。
2. 如果标准做法不适用于 C2 设定（如 DP 用了 C2 没有的模型），设计 5–30 分钟的微观实验，用数据本身回答。
3. 微观实验不可行或结果 ambiguous 时，带现象描述与倾向来找 Claude 讨论。

不直接问 LLM "该选哪个"：LLM 会给一个"合理"的答案，但你不知道这个答案的依据是文献、是训练数据中的代码片段、还是 pattern completion。决策的可追溯性丢失。

### 训练不收敛 / eval 全 0

来源：c2_plan §11 的诊断顺序 → Claude 交互。

流程：按 c2_plan §11 列出的 7 步诊断清单逐步排查，每步把现象描述发给 Claude，Claude 给下一步建议。不跳步、不猜。

---

## 与 c2_plan 的差异记录

1. ODE ablation 步数从 $N \in \{1, 2, 4, 8, 16\}$ 改为 $N \in \{1, 4\}$。GPU-hr 从 ~4 降到 ~1.5。待确认后更新 c2_plan §2。
