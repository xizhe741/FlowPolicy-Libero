# 实验报告：Conditional Flow Matching Policy on LIBERO-Goal

---

## 0. TL;DR

本实验比较 CFM（conditional flow matching）与 DP（diffusion policy）两种基于视觉的动作生成方法。两个模型共享 R3M 视觉编码与 FiLM 条件注入的 1D U-Net 骨干，仅训练目标和推理方式不同。评估在 LIBERO-Goal 的 5 个 task 上进行（每方法 5 task × 3 seed，共 30 run）。结论：从成功率、平均成功步数、推理时间三方面比较，CFM 在这三方面都优于 DP，推理速度快约 4×。两个模型均出现跨任务成功率分化，主要是因为任务对末端定位的容错不同。

## 1. 项目定位

本实验在 LIBERO-Goal 上比较 CFM 和 DP 两种动作生成方法。两者共享视觉编码、骨干网络、训练流程和评估方式，只在训练目标和推理方式上不同，表现差异归结于生成方法本身。本实验主要比较两点：1.策略在多个任务上的成功情况；2.策略的推理时间。

## 2. 方法概要

### 2.1 CFM、DP 训练目标与 Euler、DDIM 推理
**训练目标**

数据 $x_1 \sim q$，先验 $x_0 \sim p_0 = \mathcal{N}(0, I)$，时间 $t \sim \mathcal{U}[0,1]$。线性（OT）路径：

$$x_t = (1-t)\,x_0 + t\,x_1, \qquad u_t(x_t \mid x_0, x_1) = x_1 - x_0.$$

损失：

$$\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\,x_0,\,x_1} \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2.$$

**推理**

求解 ODE

$$\frac{dx_t}{dt} = v_\theta(x_t, t), \qquad x_0 \sim \mathcal{N}(0, I),$$

从 $t=0$ 积到 $t=1$。Euler 步：

$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t).$$

### 2.2 Head Replacement 对照

相同点：共享 [R3M](https://github.com/facebookresearch/r3m) 图像处理网络；[ObsEncoder](../src/model/ObsEncoder.py#L63)；骨干网络；训练全程（除训练目标）。

差异：训练目标不同，推理方式不同。



### 2.3 模型架构
模型接受Libero2帧2摄像头一共4个图像、自身proprio(同样两帧)，输出一个action chunk。 因此模型采用了朴素的生成方法 ：用带条件注入的unet生成action chunk。DP和flow matching共享骨干网络，因此U-net可以做$epsilon$预测，也可以做速度场预测。

模型相对常见生成模型的唯一不同点在于条件注入方式：作为条件的输入图像会先进入R3M模型转化为特征图，此后特征图和proprio在一个 [encoder](../src/model/ObsEncoder.py#L63) 中，经过注意力机制转化为一个维数较小的特征向量o（[降维原因见 §3.1](#31-obsencoder-引入)）。特征向量o最终通过[FiLM](../src/model/modules.py#L29)注入到Unet的每一层中。在网络中，另一个需要注入的条件是t，（diffusion步数或fm的时间），模型直接将t,o拼接一并注入到unet，不引入区分他们的inductive bias.


### 2.4 视觉编码与 R3M 缓存
为了加快训练进度，实验中固定了用于图像处理R3M网络（做法和diffusion policy一致），并且进行图像预处理：在训练前，将demos里面的图像通过 [R3M](https://github.com/facebookresearch/r3m),在之后这样可以减少显存占用和训练时间。

## 3. 关键工程决策

### 3.1 ObsEncoder 引入
相比于DP的resnet18,本实验使用了输出维度更大的resnet50(r3m)。因此条件向量更大，代价是条件向量FiLM注入到unet的全连接层也增大。r3m返回的特征向量约8000维；因此考虑用注意力机制将它缩减到更低的维度,这个注意力层被称为[obsencoder](../src/model/ObsEncoder.py#L63)

### 3.2 两阶段早停 + SR uplift margin
为了减少训练时长，实验引入了[early-stopping](../src/train.py#L543)，简要做法如下：先监测loss平台，如果loss进入平台，每20epochs对策略进行一次quick_eval,如果eval持续不提升，就提前终止训练。由于缺少early-stopping的参数调整经验，因此本次实验中没有取得很好的时间节约效果。

### 3.3 单一 evaluate 接口
本实验的evaluate接口是 [evaluate()](../src/eval.py#L103), 具体评估流程：选定推理参数（Euler步数，DDIM去噪步数）以及Eval规模（episode数目，maxstep:通常200-400，以及是否录制成功失败视频），evaluate函数call [rollout()](../src/eval.py#L26)，rollout执行若干次并且返回是否成功（bool)，结束原因,录制视频等。evaluate统计成功率，结束原因，并且蓄水池采样若干个视频。

### 3.4 DDIM timesteps 子采样
config中默认采样步数是100，也可以选择16步，本实验最终评估使用16步推理，timesteps 为 `[90, 84, 78, 72, 66, 60, 54, 48, 42, 36, 30, 24, 18, 12, 6, 0]`。


## 4. 实验设置

CFM 与 DP 的 head replacement 对比，在 LIBERO-Goal 的 5 个 task 上进行：

- put the cream cheese in the bowl 
- turn on the stove 
- put the wine bottle on top of the cabinet 
- push the plate to the front of the stove
- open the top drawer and put the bowl inside 

每个 task 用 3 个 seed（42 / 43 / 44）。CFM、DP 各 5 × 3 = 15 run，共 30 run。

## 5. 当前结果

### 5.1 CFM / DP run 结果
| Task | cfm (s42/43/44) | cfm 均值±std | dp (s42/43/44) | dp 均值±std | Δ (cfm−dp) |
|---|---|---|---|---|---|
| cream_cheese_in_bowl | 0.28/0.32/0.10 | **0.233**±0.117 | 0.20/0.14/0.12 | 0.153±0.042 | +0.080 |
| turn_on_the_stove | 0.90/0.84/0.98 | **0.907**±0.070 | 1.00/0.48/0.98 | 0.820±0.295 | +0.087 |
| wine_bottle_on_cabinet | 0.82/0.88/0.84 | **0.847**±0.031 | 0.74/0.62/0.34 | 0.567±0.205 | +0.280 |
| push_the_plate | 0.84/0.84/0.66 | **0.780**±0.104 | 0.58/0.62/0.58 | 0.593±0.023 | +0.187 |
| open_top_drawer | 0.54/0.60/0.48 | **0.540**±0.060 | 0.66/0.42/0.52 | 0.533±0.121 | +0.007 |
| **Overall (15 runs)** |  | **0.661** |  | **0.533** | **+0.128** |

平均成功步数（step）：

| Task | cfm (s42/43/44) | cfm 均值±std | dp (s42/43/44) | dp 均值±std | Δ (cfm−dp) |
|---|---|---|---|---|---|
| cream_cheese_in_bowl | 95.2/93.6/91.6 | **93.5**±1.8 | 112.5/77.6/86.7 | 92.3±18.1 | +1.2 |
| turn_on_the_stove | 85.3/109.6/82.8 | **92.6**±14.8 | 79.7/133.0/80.1 | 97.6±30.7 | −5.0 |
| wine_bottle_on_cabinet | 87.2/92.0/95.5 | **91.6**±4.2 | 94.5/92.7/117.2 | 101.5±13.7 | −9.9 |
| push_the_plate | 131.5/124.5/125.7 | **127.2**±3.7 | 131.0/128.8/132.7 | 130.8±2.0 | −3.6 |
| open_top_drawer | 193.5/200.0/192.3 | **195.3**±4.1 | 207.2/212.7/220.5 | 213.5±6.7 | −18.2 |
| **Overall (15 runs)** |  | **120.0** |  | **127.1** | **−7.1** |

评估规模：每个 checkpoint 评估 50 个 episode，maxstep 400。训练上限 800 epoch、batch size 256，并且带有 early stopping，最终训练在6k - 8k steps区间 结束。训练硬件为 RTX 4090。

5 task 选取理由：偏向primitive 覆盖最大化、horizon 跨度明确，并且希望为Legato的引入提供接口。

### 5.2 head-replacement 差异初步观察
综合 §5.1、§5.3：

- 成功率：CFM 总均值 0.661，DP 0.533，ΔSR +0.128；5 个 task 上 CFM 均值都 ≥ DP（open_top_drawer 几乎持平，+0.007；其余 +0.08~+0.28）。
- 平均成功步数：CFM 在 4/5 个 task 上更短，成功时完成得更快。
- 推理速度：CFM（N=4）chunk 级比 DP（T=16）快 4.02×。

这批 run 里 CFM 在成功率、完成效率、推理速度上相比 DP都有提升。但每 task 只有 3 个 seed，且 DP 有 turn_on_the_stove 单 run 成功率极低现象，目前只是初步观察，不是统计结论。

### 5.3 推理时间比较

| 方法 | 步数 | ms / 步 | ms / chunk |
|---|---|---|---|
| CFM | N=4 | 5.286 | 21.145 |
| DP | T=16 | 5.312 | 84.996 |

每步耗时几乎相同（共享 backbone，每步一次 U-Net 前向）；chunk 级 CFM 比 DP 快 4.02×，速度差完全来自步数（4 vs 16）。

## 6. 结论：

本实验在 5 个 LIBERO-Goal task 上对比 CFM 与 DP。head replacement 上，CFM 在成功率（总均值 0.661 vs DP 0.533）、平均成功步数、推理速度三方面都不劣于 DP；推理速度 CFM 比 DP 快 4.02×，且因两者共享 backbone、每步耗时相同，这一差距是结构性的，只来自采样步数。

绝对表现上，CFM 在 turn on the stove、wine bottle、push plate 三个 task 上良好（0.78–0.91），在最复杂的 open the top drawer and put the bowl inside 上约 0.54（π₀ 为 0.76），在名义最简单的 put the cream cheese in the bowl 上最差（0.23）。

按任务类型看，成功率主要由任务对末端定位精度的容错决定。接触类任务（turn on the stove 转旋钮、push the plate 推动）和大物体抓取-放置（wine bottle）对定位误差容错高，表现好；需要精确抓取小物体的 cream cheese 容错低，一旦出现§7 记录的坐标偏移立刻失败，成功率最低；长程多阶段的 open the top drawer and put the bowl inside 居中，两个阶段累积误差。

## 7. 失败模式记录以及一些非平凡观察

- 模型容量对任务的影响：初版 U-Net 约 35M，训练效果差；加宽 U-Net 后增至约 95M，训练效果明显提升。推测：模型宽度不足是条件化调制的通道较少，因此容易容易发生模式探索。

- 失败模式：坐标偏移失败 -- 失败视频显示策略会发生系统性偏移，该偏移会直接导致抓握失败，推测可能来源于OOD。这个问题主要发生在任务`put the cream cheese in the bowl` 之中，并且显著拉低了成功率。

- 失败模式：gripper松开 -- gripper 在抓握过程中突然松开。推测 gripper 对生成噪声敏感，其值从 1 偏移到约 0.9 即导致松手；可能的解决方案：推理时将 gripper 维阈值化，snap 到最近模式；或用自引导去噪锐化生成（注入基于先验观测的 negative score，[Self-Guidance + Adaptive Chunking](https://arxiv.org/pdf/2510.12392)）；或在训练侧拉直概率路径以提升少步采样精度（Rectified Flow / MeanFlow）。

- 失败后 OOD 的任务间分化：`put the cream cheese in the bowl` 失败后策略进入 OOD 状态，`open the top drawer and put the bowl inside` 失败后则不会。推测与失败后的状态是否落在训练数据分布内有关。

- 采用ResNet-50的收益未知：没有选取标准的DP官方的resnet18是因为训练前判断：图像条件的信息量更丰富对策略有帮助，未验证实际收益。

- DP 训练种子不稳定：turn_on_the_stove 上 DP 三个 seed 的评估结果为 1.00 / 0.48 / 0.98，一个评估结果仅0.48,其余两个接近满分。这说明seed43下面模型收敛到较差的策略。同 task 的 CFM 为 0.90/0.84/0.98，该现象不发生。

## 8. 后续工作
- 尝试修复gripper松开的失败模式；

- 推理侧引入RTC改善准确性和一致性；

- 尝试将Legato的训练目标修正引入CFM。

---
