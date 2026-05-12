# 流式策略线阅读与产出计划

> 本文档作为 Claude Project 的共享上下文使用。每次开启新对话时，先告知 Claude 当前位于哪一天、哪一篇论文、哪个门槛，Claude 据此选择交互模式。

---

## 0. 计划性质与总目标

最终目标：向高阳（IIIS, Tsinghua）发出兼具 CV、复现 repo、研究 proposal 的定义化/科研助理申请邮件，并向至少一位组内博士生（Yufeng Liu 或 Yitian Zheng）发出兼具具体技术问题的邮件。

时间锚点：

- **第 14 天前**：发出邮件给高阳，含 proposal 草稿与最小复现 repo 链接。
- **第 21 天前**：完成 Gao Yang 组两篇 S 级论文的简化技术 note，至少一个复现实验子任务出可展示结果。
- **第 28 天前**：完成 ablation 实验，发出给组内博士生的具体技术问题邮件。

如果第 14 天 proposal 与 repo 不达 P0 要求，邮件最迟推到第 21 天。超过 21 天意味着方案需重新评估，考虑切换到许华哲组备择方向。

---

## 1. 论文依赖关系

阅读顺序按相互依赖确定，而非按重要性。下面给出依赖图：

```
─────────────────── Layer 0 (前置) ───────────────────
│  FM (Lipman)    DDPM    DAgger    HER    BC 经典分析  │
──────┬───────────┬───────────┬──────────────────────
       │           │           │
       │           ▼           │
       │   ──── Layer 1 ────   │
       │   │ Diffusion     │   │
       │   │ Policy (DP)   │   │
       │   ──────┬─────────   │
       │        │              │
       └────────┼──────────────│
                ▼
        ──── Layer 2 ────
        │     π_0        │
        ────┬─────────┬───
            │        │
            ▼        ▼
    ─── Layer 3 ──────────
    │ Translating   Native  │
    │ Flow to       Cont.   │
    │ Policy                │
    ───────────────────────
```

依赖关系的解读：

- **Diffusion Policy** 引入了在策略学习语境下的 visuomotor 编码模型 + action chunking + receding horizon。后续所有流策略论文都默认读者熟悉这套设定。
- **$\pi_0$** 把 DP 中的 diffusion 换成 flow matching，加 VLM backbone 与 action expert。它是 Gao Yang 组两篇论文的事实 base policy。
- **Translating Flow to Policy** 假设读者懂 $\pi_0$ 风格的 flow model，并在其上做 online IL（DAgger 框架）+ hindsight relabeling（HER 概念）。因此需要前置 DAgger 与 HER。
- **Native Continuation** 仅依赖 $\pi_0$ 加 ODE 数值积分基础。

### 前置概念快速补足（Day 0–1）

下面每条不超过两段，目标是快速建立词汇表。每条都附"何时回查"——如果在后续阅读中遇到该概念卡壳，再回到对应原始文献。

**1. Flow Matching（Lipman et al., ICLR 2023）—— 你已知**

仅复查 CFM 等价性：边缘速度场损失 $\mathbb{E}_{\tau, x_\tau}\|v_\theta(x_\tau, \tau) - u_\tau(x_\tau)\|^2$ 与条件速度场损失 $\mathbb{E}_{\tau, x_1, x_\tau \mid x_1}\|v_\theta(x_\tau, \tau) - u_\tau(x_\tau \mid x_1)\|^2$ 在对 $\theta$ 的梯度上相同（差一个不依赖 $\theta$ 的常数）。

回查触发：在 $\pi_0$ 论文中如果对 loss 形式存疑，回到 Lipman et al. §3。

**2. DDPM 与 $\epsilon$-prediction —— 你已知**

仅在读 Diffusion Policy 时确认其 noise schedule 的具体选择（DDPM iDDPM 还是其他）与 prediction target 是 $\epsilon$ 还是 $x_0$。

**3. DAgger（Ross et al., AISTATS 2011）—— 需补**

核心思想：Behavior Cloning 在测试时遇到训练分布外的 state，预测误差导致更深层的偏移（covariate shift）。DAgger 通过迭代收集——用当前策略在环境中 rollout，把得到的 state 给 expert 标注 action，aggregate 到训练集再训练——使训练分布逐渐覆盖测试分布。有数据保证：在 expert 是 "bounded loss" 假设下，DAgger 的次优 gap 是 $O(T \epsilon)$ 而 BC 是 $O(T^2 \epsilon)$（$T$ 为 horizon，$\epsilon$ 为单步 loss）。

补足资源：原论文 §3 即可，不需要看证明。重点是"为何 BC 在 horizon 上的 error 复合是平方而 DAgger 是线性"这一直觉。

回查触发：读 *Translating Flow to Policy* 时若对"online IL"机制不清。

**4. Hindsight Experience Replay（Andrychowicz et al., NeurIPS 2017）—— 需补**

核心思想：在 goal-conditioned RL 中，即使 agent 没达到原 goal，把它实际到达的 state 作为新的 "假装本来就是这个 goal" 的目标，重新计算 reward。这样失败轨迹被复用。

为什么"hindsight"这个词在 *Translating Flow to Policy* 中出现：作者把 offline flow model 在 online rollout 中产生的"非 expert"轨迹用类似的事后 relabel 方式重新利用。具体 relabel 规则需要在读论文时确定。

补足资源：原论文 §3 + §4.1 算法表。

**5. Behavior Cloning 的 covariate shift 经典分析 —— 应已知**

基础事实：BC 把 imitation 视为监督学习，但测试时 state 分布由策略本身决定（$d^\pi$）而非数据分布（$d^{\pi^*}$），二者不一致即 covariate shift。Ross & Bagnell 2010 给出 $O(T^2)$ 的悲观 bound。

如果不熟，DAgger 论文 §2 已包含必要内容。

**6. Receding horizon control —— 应已知**

控制论 标准概念：在 $t$ 时刻规划 horizon $H$ 步动作 $\{a_t, a_{t+1}, \ldots, a_{t+H-1}\}$，但只执行 $a_t$（或前 $k$ 步），到 $t+1$（或 $t+k$）时重新规划。Action chunking 是其在 IL 中的应用。

回查触发：读 *Native Continuation* 时如对 "chunk boundary" 的几何含义不清。

---

## 2. 阅读路径与门槛

按依赖层级推进。每篇有"通过门槛"——能用自己的话向 Claude 复述对应内容才算通过。

### Day 0–1: Layer 0 前置补足

简单过完上述六条概念，不要求深度掌握。Claude 不主动启动交互。如阅读后续论文遇到具体问题再问。

无产出物要求。无门槛验证。

### Day 1–3: Layer 1 — Diffusion Policy

入口问题：

- Diffusion Policy 的训练目标在策略学习语境下的精确形式：输入是什么、输出是什么？
- Action chunking：chunk 长度 $H$ 与执行步数 $k$ 的区别，为何要 receding horizon？
- 在 PushT 上为何 Diffusion Policy 显著优于 BC？关键瓶颈是什么（multi-modality？compounding error？）？

通过门槛：能描述 Diffusion Policy 的 forward / backward / sampling 过程，并指出三个关键设计：visuomotor encoder 的选择、prediction target、action chunking 与 receding horizon 的取舍。

交互模式：苏格拉底式开放提问 + 推导验证。

### Day 3–5: Layer 2 — $\pi_0$

带读协议：三 pass 节点遍历。每轮对话只展示一个点；展示完即停用户 ack。继续。/ 提问 / 推导反馈，改进上一点。用户可随时要求跳过、回退、或层级切换。

知识分层依据 Project Knowledge 中的 L1/L2 节点表（4 个 axis × 14 个 L2 节点 × 共 55 个 L3 子项）。每个点的描述详略由其自身层级判定（→ 简要功能性描述，↓↓ 完整描述，含原文符号定义、公式、一段解读）。

**Pass 1 — L1（4 轮）**：按 axis A → B → C → D 顺序，每轮展示 1 个 axis。
- axis A：定位与上下文（→），axis D：实验（→）：简要
- axis B：方法（↓↓↓），axis C：训练 setup（↓↓）：完整

**Pass 2 — L2（14 轮）**：按节点编号 1 → 14，每轮展示 1 个节点。深度按节点自身。
- 节点 1, 2, 8, 9, 10, 11, 12, 13, 14（→）：简要
- 节点 3, 4, 5, 6, 7（↓↓ 含以下）：完整

**Pass 3 — L3（55 轮）**：按子项编号 1.1 → 14.2，每轮展示 1 个子项。深度按子项自身。

**门槛验证**：三 pass 完成后进入。能向 Claude 推导 $\pi_0$ 的 conditional flow matching loss，并描述 VLM backbone 的输入输出接口（token 化方式与 action expert 的连接点）。

**复现池更新检查点**：门槛通过后，Claude 主动询问是否更新复现候选池（见第 3 节）。

### Day 5–7: Layer 3a — Translating Flow to Policy

入口问题：

- 从 offline flow model 到 online policy 的 gap 是什么？为何训练好的 flow model 不能直接当 policy 用？
- Hindsight 在这里指什么？relabeling 的具体操作？与 HER 的对应关系？
- Online imitation 阶段的 distribution shift 来源：是 covariate shift 还是 reward-induced shift？
- DAgger 框架如何与 flow model 结合？policy improvement 步在哪里？

通过门槛：能完整描述从 offline flow training 到 online policy 的算法流程，并准确指出 hindsight 操作插入在哪一步。解决哪个具体的失败模式。

产出：本篇的简化技术 note（800–1500 字）。

**复现池更新检查点**：读完后 Claude 主动询问。

### Day 7–9: Layer 3b — Native Continuation

入口问题：

- Action chunking 的边界不连续在数学上如何刻画？是动作值不连续，还是速度场不连续，还是 ODE 初始条件不一致？
- Native continuation 与传统 receding horizon 在 ODE 积分路径上的差别？
- 这个修正是否改变了训练目标，还是仅在推理时生效？
- 与 Translating Flow to Policy 的关系：两者解决的问题是相同 distribution shift 的不同侧面，还是无关？

通过门槛：能描述 chunk 边界处速度场的行为，对比 native continuation 与 receding horizon 的 ODE 积分轨迹。能清晰定位它与 Translating Flow to Policy 的差异。

产出：本篇的简化技术 note（800–1500 字）。

**复现池更新检查点**：读完后 Claude 主动询问。

### Day 9–11: A 级支撑（按需）

仅在前面阅读中遇到瓶颈时回来。

- **Lipman et al. Flow Matching**：如果 Day 3–5 推导 $\pi_0$ 的 loss 时对 CFM 等价性细节存疑，再回这篇。重点是等价性定理的卡性质 + 平方展开两步。
- **Diffusion Policy 二次精读**：如果在读 $\pi_0$ 或 Native Continuation 时对 action chunking 的具体实现细节不清，回到 DP §4。

### B 级（Day 11+，按需）

- **Rectified Flow**：在复现实验做 ODE 步数 ablation 时如果想理解为何低步数仍可用，扫读这篇。
- **OT-CFM**：proposal 题目若涉及 endpoint 配对优化，扫读。
- **Diffuser**：与流策略关系较远，仅作为背景。
- **EfficientZero V2**：仅作为"理解 Gao Yang 起源"的背景，最后扫读。

每篇产出物仅为一段话总结。

---

## 3. 复现实验

### 设计原则：动态候选池

复现实验的具体选择不在 Day 0 锁定。维护一个候选池，每读完一篇 S 级论文后由 Claude 主动询问是否更新池或切换选择。最终在 **Day 5 前** 锁定（在开始投入大量编码前）。

#### 初始候选池（Day 0）

| 编号 | 描述 | 风险 | 展示价值 |
|---|---|---|---|
| C1 | PushT + 自实现 CFM head（在 Diffusion Policy 官方 code 上改造）| 低 | 中 |
| C2 | LIBERO-Goal subset + 自实现 CFM policy | 中 | 高 |
| C3 | 直接跑 Translating Flow to Policy 官方 code（若已开源）| 极低 | 低 |

#### 候选池更新规则

每读完以下论文，Claude 主动问"是否要把候选 X 加入池 / 替换当前选择"：

- 读完 **$\pi_0$**：可能浮现的新候选——
  - C4: 实现一个最小 $\pi_0$-style policy + 单任务训练。VLM backbone 使用 100M–1B 级别小模型（SmolLM、TinyLlama 等）。
  - 评估：双卡 4090 可承载。VLM 不微调时可常驻一卡，action expert 训练在另一卡。实现成本中等，展示价值高于 C2。
- 读完 **Translating Flow to Policy**：可能浮现的新候选——
  - C5: 在 C1/C2 基础上加 hindsight relabel 模块，做"是否加入 hindsight 影响 success rate"的小 ablation。
  - 评估：这是把"复现"升级为"小贡献"的机会，但实现成本更高。
- 读完 **Native Continuation**：可能浮现的新候选——
  - C6: 在 C1/C2 基础上实现 native continuation 推理，测试 chunk 边界附近的动作平滑度。
  - 评估：纯推理时修改，实现成本低，能直接引出 Gao Yang 组论文。

#### 决策机制

Claude 在每次询问时给出：

1. 新候选与当前选择的对比（实现成本、计算成本、展示价值）。
2. 是否建议切换。
3. 切换的时间代价（已投入的工作多少要重做）。

最终决策由你做。Claude 的角色是提供权衡材料。

#### 当前选择（Day 0）

默认 **C2: LIBERO-Goal subset + 自实现 CFM policy**，备选 **C1**。

选择理由：

- 自己从头实现的 PyTorch CFM policy（约 300 行）能在 README 中直接展示训练目标的精确实现位置。
- LIBERO-Goal 子任务在双卡 4090 上每任务每种子约 2–4 小时训完。
- 与 Diffusion Policy 在同一 setting 对比，展示对两个范式差异的实质理解。

如果 Day 10 仍未在云端跑通 LIBERO 仿真，则降到 C1。

### 设计（针对当前选择 C2）

最小可行系统：

- 视觉编码器：ResNet-18，使用 ImageNet 预训练权重，不微调。
- 速度场网络：MLP $v_\theta(a_\tau, \tau, o)$，输入维度 = action_dim + 1 + obs_emb_dim。
- CFM 训练 loss：

$$\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim U[0,1],\, a_0 \sim \mathcal{N}(0, I),\, a_1 \sim \mathcal{D}}\left\|v_\theta\left(\tau a_1 + (1-\tau) a_0,\, \tau,\, o\right) - (a_1 - a_0)\right\|^2$$

- 推理：Euler ODE solver，默认 4 步。

实验目标：

1. LIBERO-Goal 至少 3 个子任务上达到 success rate $\geq 50\%$。
2. 与 Diffusion Policy（robomimic 默认配置）在同任务上对比 success rate。
3. ODE 步数 ablation：$N \in \{1, 2, 4, 8, 16\}$，给出 success rate vs $N$ 曲线。

时间预算：

- Day 5–10：跑通环境，单任务训练 pipeline 验证。
- Day 11–20：完成 3 个子任务的对比实验（用于第 14 天邮件时已有 1 个任务的初步结果）。
- Day 21–28：ablation + 写完整 README。

### 资源配置

- 双卡 4090 云端实例。
- 总训练时长预算 $\leq 80$ GPU-小时。
- 提前在 Day 5–6 排雷 LIBERO 在 headless 云端的渲染依赖（EGL / OSMesa）。

---

## 4. 产出物清单

按交付优先级排序。

### P0（必须）

1. **Research proposal**：2–3 页 LaTeX。题目从以下两个候选中选定（Day 8 前决定）：
   - 流式策略在动作流形上的稳定性分析。
   - Native continuation 与 hindsight relabeling 的统一视角（offline-to-online 的视角下两者解决相关但不同的 distribution shift）。

2. **复现 repo**：GitHub 公开仓库，含完整代码、训练脚本、README。README 中明确标注训练目标的数学形式与代码实现的对应行号。

3. **给高阳的邮件**：附 CV、成绩单 PDF、proposal PDF、repo 链接。邮件正文 $\leq 400$ 字，直接说明：(a) 已精读组内两篇 flow policy 工作；(b) proposal 题目与核心思路；(c) 已开始复现，当前进度。

### P1（强烈建议）

4. **Gao Yang 组两篇论文的简化技术 note**（每篇 800–1500 字），放在 repo 的 `notes/` 目录。

5. **给组内博士生的邮件**：附一个具体技术问题（不是兴趣表达）。问题来源应是阅读 S 级论文时产生且未被论文回答的疑问。

### P2（如时间允许）

6. **Lipman CFM 等价性证明的 LaTeX 推导文档**（1–2 页）。

### 明确不产出

- A 级与 B 级论文的技术 note。
- Layer 0 前置概念的扩写（仅一页词汇表）。
- 学习日志、进度条、可视化图表。
- 任何"读后感"或"心得体会"形式的内容。

---

## 5. 与 Claude 的交互协议

每次开启新对话时：

- 你声明：当前 Day 数 、当前论文、当前门槛进度。
- Claude 据此选择交互模式：

| 论文层级 | Claude 的角色 |
|---|---|
| Layer 0 前置 | 不主动交互。如果具体问题再问。|
| Layer 1 (DP) | 苏格拉底式开放提问 + 推导验证。|
| Layer 2 ($\pi_0$) | 三 pass 节点带读（L1/L2/L3 各一轮，每轮一点；单点等 ack）+ 三 pass 完成后门槛推导验证 + 复测。|
| Layer 3 (Gao Yang 组) | 苏格拉底式开放提问 + 推导验证 + 复测 + 复现池更新询问。|
| A 级 | 答疑为主，关键定理逐行验证。|
| B 级 | 纯答疑，不主动推导。|

通用规则：

- 你做推导，Claude 验证。Claude 不替你做推导（除非你明确要求"给出参考答案"）。
- 复现实验阶段，Claude 协助 debug、审查超参选择，但不写完整代码。架构与超参由你决定。
- proposal 写作阶段，Claude 给反馈与修改建议，不替你起草初稿。

### 复现池询问的触发条件

读完 $\pi_0$、Translating Flow to Policy、Native Continuation 三篇中任一篇并通过门槛后，Claude 必须主动启动一次复现池更新询问，分别：

1. 本篇论文是否提示新的复现切入点？
2. 是否有新候选要加入池？
3. 当前选择是否仍是最佳？切换的时间代价？

询问后由你决定是否更新。决策结果记录在附录 C。

---

## 6. 检查点

每周日（Day 7、14、21、28），向 Claude 报告：

- 当前 Day 数。
- 通过门槛进度（已通过 / 进行中 / 未开始）。
- 复现实验当前 success rate 数值。
- proposal 当前题目与摘要（一段话）。

Claude 据此判断是否需要调整计划。例如 Layer 1–2 拖到 Day 10 而复现尚未启动，则需缩减后续阅读、保护核心交付。

---

## 7. 失败应对

- 第 14 天 proposal 不完整：邮件推到 Day 21。
- 第 21 天复现仍无结果：邮件改为不带 repo 的版本，仅靠 proposal + 已读论文撑场。
- 第 28 天整体进度严重落后：重新评估方向。考虑切换到许华哲组（撑场前面）或备选方向的领头复制其他组（Sergey Levine, Pieter Abbeel, Chelsea Finn 等）。这不是失败，是基于新信息的更新。

---

## 附录 A：当前状态记录

> 每次对话开始前在此处更新。

- 当前 Day：[填入]
- 当前论文 / 层级：[填入]
- 当前门槛进度：[填入]
- 复现实验状态：[填入]
- proposal 题目：[填入]
- 待解决问题：[填入]

## 附录 B：已通过门槛清单

> 每次通过一个门槛时在此处追加。

- [ ] Layer 0 前置概念词汇表
- [ ] Diffusion Policy
- [ ] $\pi_0$
- [ ] Translating Flow to Policy
- [ ] Native Continuation
- [ ] Lipman CFM 等价性（按需）
- [ ] B 级论文扫读
- [ ] 复现实验单任务跑通
- [ ] 复现实验 3 任务对比
- [ ] ablation 完成
- [ ] proposal 定稿
- [ ] 邮件已发出

## 附录 C：复现池更新历史

> 每次复现池询问与决策记录在此。

| Day | 触发论文 | 询问内容 | 决策 |
|---|---|---|---|
| 0 | (初始) | 候选池 C1/C2/C3 设立，默认 C2 | 接受默认 |
| | | | |
