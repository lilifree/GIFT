# GIFT Microbatch Token-only：公式、代码与训练链路完整说明

> 本文只解释当前已经实现并用于实验的 **GIFT Microbatch token-only**，对应配置
> `qwen3_8b_gift_microbatch_token_anneal.yaml`。
> 文中描述以代码的真实行为为准，不把设计草稿中未实现的内容算入方法。

---

## 1. 一句话先说清楚

这个版本在每个真实 actor training microbatch 内，把所有 valid response token 放在一起排序：

1. 用当前 token 的 clipped GRPO loss 梯度作为参考方向；
2. 把 OPSD forward-KL loss 梯度分解为沿 GRPO 的平行分量和垂直于 GRPO 的残差；
3. 冲突 token 用负的平行冲突强度打分，非冲突 token 用正交残差大小打分；
4. 每个 microbatch 只给排名前 $K$ 个 token 加完整 OPSD KL；
5. $K$ 从“几乎全选”逐步降到 0，但 OPSD 系数始终固定为 1；
6. step 78 起跳过 teacher，训练目标变成纯 GRPO。

最核心的分数是：

\[
s_i=
\begin{cases}
a_i=-|a_i|, & a_i<0,\\[3pt]
b_i, & a_i\ge 0,
\end{cases}
\qquad
a_i=\left\langle d_i,\frac{r_i}{\|r_i\|}\right\rangle,
\qquad
b_i=\left\|d_i-a_i\frac{r_i}{\|r_i\|}\right\|.
\]

这里 $r_i$ 和 $d_i$ 都是 **loss gradient**。优化器做梯度下降，因此：

- $a_i>0$：两个 loss gradient 同向，它们对应的下降方向也同向，记为非冲突；
- $a_i<0$：两个 loss gradient 反向，记为冲突；
- $b_i$：teacher 梯度中无法由该 token 的局部 GRPO 方向表示的部分。

特别重要：**投影只用于 detached 排名，不用于反向传播。** 选中 token 后回传的是完整、原始的 forward KL，不是 $d_i^\perp$，也没有做 PCGrad 式梯度手术。

---

## 2. 这套实验到底固定了什么

8B 主实验的配置文件是：

```text
scripts/gift/main/
└── qwen3_8b_gift_microbatch_token_anneal.yaml
```

关键配置如下：

```yaml
gift_piecewise_enabled: true
gift_enabled: true
gift_selection_scope: micro_batch
gift_score_mode: piecewise_conflict_residual
gift_schedule: linear_token_budget_only
gift_total_training_steps: 96
gift_initial_retention: 1.0
gift_hold_steps: 10
gift_exit_step: 78
gift_norm_eps: 1.0e-8
gift_piecewise_zero_grpo_policy: neutral_score_zero
gift_piecewise_zero_opsd_policy: exclude

method: opsd_grpo_curriculum
group_size: 8
ppo_mini_batch_size: 64
actor_max_token_len_per_gpu: 24576
rollout_is: token
rollout_is_threshold: 2.0

beta: 0.0
jsd_token_clip: 0.0
full_logit_distill: true
distill_add_tail: true
top_k_distill: 100
opsd_grpo_lambda: 1.0

val_before_train: false
save_steps: 10
eval_steps: 10
```

因此当前实验的明确口径是：

| 维度 | 当前实现 |
|---|---|
| 模型 | Qwen3-8B |
| 训练集 | DeepMath level 7–10 |
| 验证集 | AMC23 + AIME24 + AIME25 |
| 每题 rollout 数 | 8 |
| 排序范围 | 每个实际 actor microbatch |
| 排序依据 | GIFT 分段梯度分数 |
| 是否使用 entropy 排序 | 否 |
| 是否用 cosine 直接排序 | 否 |
| 是否用原始梯度内积直接排序 | 否 |
| 蒸馏方向 | 固定 forward KL |
| sparse support | teacher top-100 + 一个 tail bucket |
| token 预算 | step 1–10 为 100%，step 11–77 线性减至接近 0，step 78 起为 0 |
| OPSD 权重 | 全程固定 $\lambda=1$ |
| step 0 评测 | 不做 |
| 保存/评测 | 每 10 个 actor step 一次 |
| 训练总步数 | 96 |

配置中的 `gift_cosine_min: 0.0` 是复用基础 runtime 配置结构留下的字段。GIFT overlay 实际会绕过基础版本的正 cosine hard gate；冲突 token 不是立即过滤，而是得到负分并在预算收缩时优先退出。

---

## 3. 文件分工：哪一层做什么

### 3.1 本实验自己的文件

| 文件 | 职责 |
|---|---|
| `gift_runtime/patch.py` | 定义 GIFT 分段评分、microbatch/group scope、额外诊断，并覆盖基础 runtime 的三个 hook |
| `qwen3_8b_gift_microbatch_token_anneal.yaml` | 8B microbatch token-only 的唯一实验参数来源 |
| `gift_bootstrap/sitecustomize.py` | Python 进程启动时自动安装 overlay |
| `run_gift.sh` | 多节点 Ray 启动（拓扑取自配置 `nnodes` / `n_gpus_per_node`）、环境隔离、preflight 和训练命令 |
| `verify_gift.py` | 配置、输入路径、输出目录和退火曲线的 fail-closed 校验 |
| `verify_gift_ray_workers.py` | 在每个 Ray 节点验证导入路径和 patch 是否真正生效 |
| `test_gift.py` | CPU 单元测试：几何、scope、边界和 schedule |

### 3.2 复用的共享 runtime

GIFT 没有复制整套 actor/loss hook，而是复用同目录下的：

```text
scripts/gift/main/
└── core_runtime/patch.py
```

共享 runtime 提供：

- sparse top-k + tail 的解析梯度几何；
- no-grad scoring pass；
- mask FIFO 队列；
- 原始 GRPO + masked OPSD loss；
- actor update hook；
- schedule 与日志。

GIFT 在调用 `core.apply_runtime_patches()` 前替换三个公开 hook：

```python
core.CoreSettings.validate = _validate_base_settings
core.sparse_tail_gradient_geometry = sparse_tail_gradient_geometry
core.build_selection_masks = build_gift_masks
core._consume_selection_mask = _consume_gift_mask
core.apply_runtime_patches()
```

对应代码在 `gift_runtime/patch.py:579-614`。这个顺序很重要：如果基础 overlay 已经先 patch，GIFT 会直接报错，而不是静默运行成另一种方法。

### 3.3 没有被修改的原始训练代码

runtime 最终在内存中 patch：

- `verl.trainer.ppo.core_algos`；
- `verl.workers.actor.dp_actor`；
- `src.self_distill_main`。

磁盘上的 `src/` 和 `third_party/verl/` 没有为本实验改写。GIFT 仍调用原生 `compute_policy_loss_vanilla` 计算 GRPO，并沿用原生动态 batching 和 optimizer step。

---

## 4. 符号和张量形状

对某个真实 training microbatch：

| 符号 | 形状 | 含义 |
|---|---:|---|
| $B$ | 标量 | 这个动态 microbatch 中的 rollout 行数，不是固定值 |
| $L$ | 标量 | padding 后 response 长度 |
| $V$ | 标量 | 完整词表大小 |
| $K$ | 标量 | teacher sparse top-k，当前为 100 |
| $p_i\in\mathbb R^V$ | `[V]` | student 在 token 位置 $i$ 的概率分布 |
| $q_i\in\mathbb R^V$ | `[V]` | privileged teacher 在相同输出位置的概率分布 |
| $u_i$ | 标量 token id | rollout 实际采样到的 action/token |
| $A_i$ | 标量 | GRPO advantage；同一 response 内通常重复到每个 valid token |
| $\rho_i$ | 标量 | PPO current/old action probability ratio |
| $w_i$ | 标量 | token-level rollout importance-sampling weight |
| $h_i$ | 0 或 1 | clipped/dual-clip surrogate 当前分支是否有梯度 |
| $r_i\in\mathbb R^V$ | `[V]` | token-local GRPO loss 的 output-logit 梯度 |
| $d_i\in\mathbb R^V$ | `[V]` | token-local OPSD loss 的 output-logit 梯度 |
| `response_mask` | `[B,L]` | valid response token 为 1，padding 为 0 |
| teacher/student top-k log-probs | `[B,L,K]` | teacher 选出的同一组词表坐标上的概率 |
| `sum_pi_squared` | `[B,L]` | $\sum_{j=1}^{V}p_{ij}^2$ |
| GIFT score/mask | `[B,L]` | detached 分数和 0/1 蒸馏选择掩码 |

本文的“梯度”均指对当前位置 output logits 的 loss gradient。为了突出 token 几何，公式省略了所有 token 共同拥有的正数聚合因子，例如全 valid-token 分母；省略这类公共正比例因子不会改变方向或排序。

当前训练温度为 1，所以不需要额外写出温度对 logit 梯度的比例因子。

---

## 5. 第一步：GRPO advantage 从哪里来

GRPO 先对同一个 prompt 的 8 条 rollout 计算 outcome reward。设第 $g$ 个 prompt 的第 $r$ 条 response reward 为 $R_{g,r}$，默认标准化 advantage 为：

\[
A_{g,r}
=
\frac{R_{g,r}-\operatorname{mean}_{r'}R_{g,r'}}
{\operatorname{std}_{r'}R_{g,r'}+\epsilon}.
\]

随后这个 response-level 标量被复制到该 response 的每个 valid token：

\[
A_i=A_{g,r},\qquad i\in y_{g,r}.
\]

对应原生实现位于 `third_party/verl/verl/trainer/ppo/core_algos.py:269-331`：

```python
scores = token_level_rewards.sum(dim=-1)
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
scores = scores.unsqueeze(-1) * response_mask
```

这解释了两点：

1. advantage 的来源是同题 group 内的相对 reward；
2. 虽然 $A$ 在一条 response 内是常数，token 的 $p_i$、$u_i$、$\rho_i$、clip 状态和 teacher 分布仍然逐位置变化。

---

## 6. 第二步：得到真实 clipped GRPO logit 梯度

### 6.1 原始 PPO/GRPO token loss

对 token $i$：

\[
\rho_i
=\exp\left(\log\pi_\theta(u_i)-\log\pi_{\mathrm{old}}(u_i)\right).
\]

未 clip 的 surrogate loss 为：

\[
\ell_{\mathrm{pg},i}^{(1)}=-A_i\rho_i.
\]

原生代码同时构造普通 clip 分支和 negative-advantage dual-clip 分支，位于
`third_party/verl/verl/trainer/ppo/core_algos.py:1319-1348`。

### 6.2 active 指示 $h_i$

GIFT 不是只看未 clip 的理论梯度，而是显式复现当前 surrogate 是否有梯度：

\[
h_i=0
\]

当且仅当命中下列任一严格条件：

\[
\begin{aligned}
&A_i>0\ \land\ \rho_i>1+\epsilon_{\rm high},\\
&A_i<0\ \land\ \rho_i<1-\epsilon_{\rm low},\\
&A_i<0\ \land\ \rho_i>c_{\rm dual}.
\end{aligned}
\]

否则 $h_i=1$。代码在基础 geometry 的 `patch.py:312-320`：

```python
active = torch.ones_like(response_mask, dtype=torch.bool)
active &= ~((advantage > 0.0) & (ratio > 1.0 + clip_ratio_high))
active &= ~((advantage < 0.0) & (ratio < 1.0 - clip_ratio_low))
active &= ~((advantage < 0.0) & (ratio > dual_clip_ratio))
```

边界使用严格 `>`/`<`，因此恰好落在边界时仍被标为 active。这是代码口径，不应另写成包含等号的条件。

### 6.3 rollout-IS 权重 $w_i$

配置使用：

```yaml
rollout_is: token
rollout_is_threshold: 2.0
```

因此 $w_i$ 是 token-level 的 train/rollout importance ratio，经上界 2.0 截断。它与 PPO ratio $\rho_i$ 是两个不同的量：

- $\rho_i$ 进入 clipped policy surrogate；
- $w_i$ 是 rollout correction，对 GRPO 和 OPSD token loss 都乘一次。

原生 rollout correction 位于
`third_party/verl/verl/trainer/ppo/rollout_corr_helper.py:524-655`。

### 6.4 最终解析式

交叉熵 action log-prob 对 logits 的导数给出：

\[
\nabla_z[-\log p(u_i)] = p_i-e_{u_i}.
\]

所以当前 clipped GRPO token loss 的 logit 梯度为：

\[
\boxed{
r_i=A_i\rho_i h_iw_i(p_i-e_{u_i})
}
\]

并且：

\[
\|p_i-e_{u_i}\|^2
=\sum_{j=1}^{V}p_{ij}^2-2p_i(u_i)+1.
\]

实现不需要保留整个 $V$ 维向量，只需要 sampled-action probability 和
`sum_pi_squared`。

---

## 7. 第三步：OPSD 的 top-100 + tail KL 到底是什么

### 7.1 它不是随便截掉 tail

teacher 先在完整词表上选 top-100 集合 $T_i$，student 再在完全相同的词表坐标上取概率。其余 $V-K$ 个词合并成一个 tail bucket：

\[
P_T=\sum_{j\in T_i}p_{ij},\qquad
Q_T=\sum_{j\in T_i}q_{ij},
\]

\[
P_{\rm tail}=1-P_T,\qquad
Q_{\rm tail}=1-Q_T.
\]

实际训练的压缩分布为：

\[
\tilde p_i=(\{p_{ij}\}_{j\in T_i},P_{\rm tail}),\qquad
\tilde q_i=(\{q_{ij}\}_{j\in T_i},Q_{\rm tail}).
\]

OPSD token loss 是普通 forward KL：

\[
\ell_{\rm OPSD,i}
=D_{\rm KL}(\tilde q_i\|\tilde p_i)
=\sum_{j\in T_i}q_{ij}\log\frac{q_{ij}}{p_{ij}}
+Q_{\rm tail}\log\frac{Q_{\rm tail}}{P_{\rm tail}}.
\]

`_sd_prepare_sparse_topk_log_probs` 在
`third_party/verl/verl/trainer/ppo/core_algos.py:2600-2624` 添加 tail；
`_sd_generalized_jsd(..., mix=0.0)` 在 `2627-2644` 实际计算的就是
$D_{\rm KL}(q\|p)$。

这里要准确区分：

- 它是 **top-100 + tail 压缩目标**，不是完整词表 KL 本身；
- 下文解析式是这个压缩目标对 **完整 student logits** 的精确梯度，不是再做一层近似。

“精确”还应理解为正常数值区间内对当前压缩目标精确。训练 helper 为 tail log-prob 设置约 $10^{-7}$ 的概率保护，geometry 对 $P_{\rm tail}$ 使用 $10^{-8}$ 下界；只有 top-100 概率质量极端接近 1 时，两处保护阈值才可能造成很小的边界差异。

### 7.2 完整 logit 梯度

定义：

\[
\alpha_i=\frac{Q_T-P_T}{P_{\rm tail}}.
\]

不含 rollout-IS 权重的 OPSD 梯度 $d_i^{(0)}$ 为：

\[
d^{(0)}_{ij}=
\begin{cases}
p_{ij}-q_{ij}, & j\in T_i,\\[3pt]
\alpha_i p_{ij}, & j\notin T_i.
\end{cases}
\]

实际 loss 还乘了 $w_i$，所以：

\[
\boxed{d_i=w_i d_i^{(0)}}.
\]

这套公式在基础 runtime `patch.py:283-310` 中实现：

```python
student_topk_mass = p_student_topk.sum(dim=-1)
teacher_topk_mass = p_teacher_topk.sum(dim=-1)
student_tail_mass = (1.0 - student_topk_mass).clamp_min(norm_eps)
tail_alpha = (teacher_topk_mass - student_topk_mass) / student_tail_mass

tail_sum_p_squared = (
    full_sum_p_squared - p_student_topk.square().sum(dim=-1)
).clamp_min(0.0)
topk_d = p_student_topk - p_teacher_topk
d0_norm_sq = (
    topk_d.square().sum(dim=-1)
    + tail_alpha.square() * tail_sum_p_squared
)
```

### 7.3 为什么只存少量统计量就能算精确几何

设 $v_i=p_i-e_{u_i}$。为了得到 $\|r_i\|$、$\|d_i\|$ 和
$\langle r_i,d_i\rangle$，只需：

\[
\|v_i\|^2=\sum_jp_{ij}^2-2p_i(u_i)+1,
\]

\[
\|d_i^{(0)}\|^2
=\sum_{j\in T_i}(p_{ij}-q_{ij})^2
+\alpha_i^2\sum_{j\notin T_i}p_{ij}^2,
\]

\[
p_i^\top d_i^{(0)}
=\sum_{j\in T_i}p_{ij}(p_{ij}-q_{ij})
+\alpha_i\sum_{j\notin T_i}p_{ij}^2,
\]

\[
v_i^\top d_i^{(0)}=p_i^\top d_i^{(0)}-d^{(0)}_{i,u_i}.
\]

其中：

\[
d^{(0)}_{i,u_i}=
\begin{cases}
p_i(u_i)-q_i(u_i), & u_i\in T_i,\\
\alpha_i p_i(u_i), & u_i\notin T_i.
\end{cases}
\]

因此不需要在 scoring 结果中保存 `[B,L,V]` 的完整概率，只需要：

- teacher/student top-k 概率；
- sampled action probability；
- sampled action 是否在 teacher top-k；
- $\sum_jp_{ij}^2$。

基础代码据此计算：

```python
policy_scale = advantage * ratio * active * importance_weight
r_norm_sq = policy_scale.square() * policy_base_norm_sq
d_norm_sq = importance_weight.square() * d0_norm_sq
dot = policy_scale * importance_weight * policy_base_dot_d
```

即：

\[
\|r_i\|^2=(A_i\rho_ih_iw_i)^2\|v_i\|^2,
\]

\[
\|d_i\|^2=w_i^2\|d_i^{(0)}\|^2,
\]

\[
\langle r_i,d_i\rangle
=A_i\rho_ih_iw_i^2\,v_i^\top d_i^{(0)}.
\]

### 7.4 这里的 teacher 是谁

当前配置是：

```yaml
privileged_text_mode: solution_answer
teacher_mode: fixed
use_lora: false
teacher_enable_thinking: true
```

`src/self_distill_main.py:353-428` 会用题目、标准 solution 和 answer 构造 privileged teacher context，再把 rollout response 对齐到这个 context 上计算 teacher token 分布。也就是说，teacher 的额外信息主要来自 privileged context，而不是让 teacher 重新采一条 response。

`teacher_mode=fixed` 在 `dp_actor.py:660-713` 的真实语义是：

- teacher forward 总在 `torch.no_grad()` 下执行；
- 若 actor 有 adapter，进入 teacher forward 时禁用 adapter；
- 否则直接用 actor module 做 privileged-context forward；
- 不建立 EMA/snapshot shadow，也不在 teacher forward 中更新参数。

本实验又设置了 `use_lora=false`，所以不要把这里的 teacher 描述成“另行加载的外部大模型”。它是这套 self-distillation actor 中的 privileged forward 路径；student 与 teacher 的关键输入差异是 teacher 能看到标准解答上下文。

---

## 8. 第四步：GIFT 分段分数

### 8.1 分解

当 $r_i$ 方向有效时：

\[
\hat r_i=\frac{r_i}{\|r_i\|},
\]

\[
a_i=\langle d_i,\hat r_i\rangle
=\frac{\langle r_i,d_i\rangle}{\|r_i\|}
=\cos(r_i,d_i)\|d_i\|,
\]

\[
d_i^\parallel=a_i\hat r_i,
\qquad
d_i^\perp=d_i-a_i\hat r_i,
\]

\[
b_i=\|d_i^\perp\|
=\sqrt{\|d_i\|^2-a_i^2}
=\sqrt{\|d_i\|^2-
\frac{\langle r_i,d_i\rangle^2}{\|r_i\|^2}}.
\]

代码不构造 $V$ 维的 $d_i^\perp$，而是直接用范数恒等式算 $b_i$。

### 8.2 分段打分

\[
\boxed{
s_i=
\begin{cases}
a_i=-|a_i|, & a_i<0,\\[3pt]
b_i, & a_i\ge 0.
\end{cases}}
\]

真实代码位于 `gift_runtime/patch.py:259-309`：

```python
parallel = torch.zeros_like(d_norm)
parallel[direction_defined] = (
    cosine[direction_defined] * d_norm[direction_defined]
)

residual = torch.zeros_like(residual_raw)
residual[direction_defined] = residual_raw[direction_defined]
conflict = direction_defined & (parallel < 0.0)

score = torch.zeros_like(d_norm)
score[direction_defined] = torch.where(
    conflict[direction_defined],
    parallel[direction_defined],
    residual[direction_defined],
)
```

### 8.3 排序含义

按分数降序 Top-K：

| 几何情形 | 分数范围 | 预算减少时的行为 |
|---|---:|---|
| 强冲突 $a\ll0$ | 很负 | 最早退出 |
| 弱冲突 $a<0$ | 略负 | 比强冲突晚，但早于所有零分/正分 token |
| 中性回退或同向且完全平行 | 0 | 冲突之后退出 |
| 非冲突、残差小 | 小正数 | 较早退出 |
| 非冲突、残差大 | 大正数 | 最晚保留 |

所以其总体优先级是：

\[
\text{非冲突且残差大}
\succ
\text{非冲突且残差小}
\succ
\text{零分}
\succ
\text{弱冲突}
\succ
\text{强冲突}.
\]

注意这是“谁更晚退出”的排序，不是说高分 token 一定对最终 reward 更有帮助。

### 8.4 它不是原始梯度内积

代码确实先算了：

\[
\operatorname{dot}_i=\langle r_i,d_i\rangle,
\]

但最终冲突分支使用的是：

\[
a_i=\frac{\operatorname{dot}_i}{\|r_i\|},
\]

非冲突分支使用的是 $b_i$。因此最终 score **不是** raw dot product。

这意味着：

- $\lvert A_i\rvert$ 不直接决定排序；
- 正的 $\rho_i$ 幅度不直接决定排序；
- $A_i$ 的符号仍会翻转 $\hat r_i$，从而改变冲突/非冲突分支；
- $\rho_i$ 仍会通过 clip active 状态 $h_i$ 影响方向是否存在；
- $w_i$ 同时乘在真实 OPSD 梯度上，因此 teacher gradient 的尺度中仍保留 $w_i$；
- $p_i$、$q_i$、sampled action $u_i$ 的 token 几何始终逐 token 变化。

对 active 且非零的 token：

\[
\hat r_i
=\operatorname{sign}(A_i)\frac{p_i-e_{u_i}}{\|p_i-e_{u_i}\|},
\]

因为 $\lvert A_i\rvert$、$\rho_i>0$ 和 $w_i>0$ 都在归一化中约掉。这里应诚实表述为“保留 advantage 的方向作用”，不能写成“保留完整 advantage 强度”。

---

## 9. 第五步：边界与 fail-closed 策略

GIFT 的候选资格不是简单的 `response_mask`。代码依次检查：

```python
finite = (
    isfinite(r_norm)
    & isfinite(d_norm)
    & isfinite(cosine)
    & isfinite(residual)
    & isfinite(ratio)
    & isfinite(advantage)
)
teacher_nonzero = d_norm > norm_eps
eligible = valid_token & finite & teacher_nonzero
direction_defined = eligible & active & (r_norm > norm_eps)
```

当前 `norm_eps=1e-8`。

### 9.1 $d_i\approx0$：排除

teacher 与 student 在该压缩分布上没有有效梯度时：

```text
eligible = false
teacher_zero = true
```

它不占 Top-K 名额。全选阶段排除这种 token 不改变训练更新，因为它的 KL gradient 本来就是零。

### 9.2 $r_i\approx0$：中性零分，不排除

下列情况可能使 GRPO 方向不可定义：

- $A_i=0$；
- PPO/dual-clip 分支 inactive，即 $h_i=0$；
- 数值饱和导致 $\|r_i\|\le\epsilon$。

当前实现策略是：

```text
eligible = true，只要 d 有效
direction_defined = false
parallel = 0
residual = 0
score = 0
neutral_grpo = true
```

这不是把“零 GRPO”数学证明成了“正交”；它只是一个明确的工程回退：没有参考方向时不猜正负，给中性排名。

### 9.3 $a_i=0$ 但 $r_i$ 有效：真正的正交边界

这与上面的零向量回退不同。若 $r_i$ 有效且 $a_i=0$，代码走非冲突分支：

\[
s_i=b_i=\|d_i\|.
\]

因此真正正交且 teacher 梯度大的 token 会得到高分。

### 9.4 非有限值：排除

任何相关量为 NaN/Inf 时 token 不进入候选，避免 `topk` 静默选择异常值。

### 9.5 “step 1 全选”的精确定义

step 1–10 的 $\kappa=1$，但代码不会选择：

- padding；
- 非有限 token；
- $\|d_i\|\le\epsilon$ 的零 OPSD token。

所以“全选”准确含义是：**选中全部 valid、finite 且 OPSD 梯度非零的候选**。这与字面选择所有 tensor 位置不同。

---

## 10. 第六步：microbatch 内 Top-K

### 10.1 microbatch 指什么

这里的 microbatch 是 VERL 在 actor 的一个 PPO mini-batch 内，根据 student/teacher 实际 workload 动态切出来、随后逐个执行 forward/backward 的 training record。

它不是：

- 单独一道题的 8 条 rollout；
- 整个 PPO mini-batch；
- 一张卡本轮拿到的所有样本；
- 多 actor rank 汇总后的 global batch。

动态切分位于 `third_party/verl/verl/workers/actor/dp_actor.py:811-857`。它综合 student 和 teacher 的有效序列 workload，并受：

```yaml
actor_max_token_len_per_gpu: 24576
ppo_mini_batch_size: 64
```

等设置影响。因此不同训练时刻的实际 $B$ 和 token 数可以不同。

### 10.2 竞争集合

对第 $m$ 个实际 microbatch：

\[
I_m=\{i:\ i\text{ 是该 microbatch 的 valid response token}\}.
\]

令：

\[
N_m=|I_m|,
\qquad
E_m=\{i\in I_m:\ i\text{ 满足 eligible}\}.
\]

预算和实际选择数为：

\[
K_m(t)=\left\lfloor\kappa_tN_m\right\rfloor,
\]

\[
K_m^{\rm actual}(t)=\min(K_m(t),|E_m|).
\]

最终 mask：

\[
m_i(t)=
\mathbb 1\left[
i\in\operatorname{TopK}_{j\in E_m}
(s_j,K_m^{\rm actual}(t))
\right].
\]

代码位于 `gift_runtime/patch.py:388-430`：

```python
scores = torch.cat(score_chunks)
eligible = torch.cat(eligible_chunks)
eligible_indices = torch.nonzero(eligible).squeeze(-1)

budget = math.floor(retention_ratio * valid_count)
selected_count = min(budget, eligible_indices.numel())

selected_local = torch.topk(
    scores[eligible_indices],
    k=selected_count,
    largest=True,
    sorted=False,
).indices
selected[eligible_indices[selected_local]] = True
```

三个细节不能省略：

1. `budget` 的分母是 valid token 数 $N_m$，不是 eligible 数；
2. 候选不足时不拿无效 token 凑满预算，所以 `selected_count` 可能小于 `budget`；
3. 使用 `floor`，因此较小 microbatch 在 $\kappa>0$ 时也可能提前出现 $K=0$。

### 10.3 跨题预算转移

一个 microbatch 如果同时含 prompt A 和 prompt B 的 rollout，所有 token 共用一个预算。例如：

```text
microbatch 中共有 1000 个 valid token，kappa=0.3
=> K=floor(0.3*1000)=300
```

如果最高的 250 个分数来自 prompt B，B 可以拿走 250 个名额，A 只拿 50 个。没有 per-question 或 per-rollout quota。

因此：

- 同一 rollout 可以选很多 token，也可以一个都不选；
- 同一 microbatch 内不同题可以转移预算；
- 不同 microbatch 不能转移预算；
- 不同 actor rank 不能转移预算；
- `global_selected` 的 all-reduce 只用于日志，不用于 global Top-K。

### 10.4 仍然检查 rollout group 完整性

即便选择 scope 是 microbatch，`_selection_scopes()` 仍会先按 `uid` 汇总当前 actor 侧这批 records，并要求每个 uid 恰好有 8 条 rollout：

```python
incomplete = {
    uid: len(refs)
    for uid, refs in groups.items()
    if len(refs) != group_size
}
if incomplete:
    raise RuntimeError(...)
```

这项检查的作用是确保数据组织仍符合 GRPO group 假设；它不把选择范围改回 group。真正的 scope 仍是每个 record：

```python
for record_idx, record in enumerate(records):
    refs = [(record_idx, row_idx) for row_idx in range(record["valid"].shape[0])]
    scopes.append((f"micro_batch:{record_idx}", refs))
```

### 10.5 同分 token

实现调用 `torch.topk(..., sorted=False)`。如果多个 token 分数完全相同，具体哪一个进入边界名额不应被解释成额外算法语义，也不保证跨硬件的 tie order。

---

## 11. 第七步：只退火 token 数，不退火权重

### 11.1 精确调度

step 是从 1 开始计数的 actor global step：

\[
\boxed{
\kappa_t=
\begin{cases}
1, & t\le10,\\[3pt]
1-\dfrac{t-10}{78-10}
=\dfrac{78-t}{68}, & 10<t<78,\\[8pt]
0, & t\ge78.
\end{cases}}
\]

OPSD 权重则是：

\[
\boxed{\lambda_t=1\quad\forall t\in[1,96].}
\]

实现位于共享 runtime `coverage_ratio()` 和
`opsd_lambda()`，即 `core_runtime/patch.py:190-228`。

### 11.2 关键 step 的数值

| step | $\kappa_t$ | 含义 |
|---:|---:|---|
| 1–10 | 1.000000 | 全预算起步 |
| 11 | $67/68=0.985294$ | 第一次减少 token |
| 20 | $58/68=0.852941$ | 最多保留约 85.29% valid token |
| 30 | $48/68=0.705882$ | 最多保留约 70.59% |
| 40 | $38/68=0.558824$ | 最多保留约 55.88% |
| 44 | $34/68=0.500000$ | 预算正好一半 |
| 50 | $28/68=0.411765$ | 最多保留约 41.18% |
| 60 | $18/68=0.264706$ | 最多保留约 26.47% |
| 70 | $8/68=0.117647$ | 最多保留约 11.76% |
| 77 | $1/68=0.014706$ | 最后一个仍有正预算比例的 step |
| 78–96 | 0.000000 | 不蒸馏，纯 GRPO |

对具体 microbatch，实际整数预算仍需代入：

\[
K_m(t)=\lfloor\kappa_tN_m\rfloor.
\]

例如 $N_m=1000$：step 44 的 $K=500$；但若只有 470 个 eligible token，实际只选 470 个。

### 11.3 为什么这是 token-only

退火期间每个被选 token 的系数始终是：

\[
\frac{\lambda}{N_{\rm valid}}=\frac{1}{N_{\rm valid}}.
\]

变化的只有 $m_i\in\{0,1\}$ 的数量，没有：

- $\lambda_t$ 衰减；
- selected token 内连续 soft weight；
- token 先减到某个固定比例后再做 weight annealing。

这与 `GIFT Microbatch staged` 不同：staged 版本复用原 microbatch-rank 的 token→weight 两阶段调度；本文版本从 $\kappa=1$ 一直把 token 数减到 0。

### 11.4 step 78 为什么能真正省掉 teacher

```python
teacher_active = (lambda > 0) and (retention > 0)
```

step 78 起 retention 为 0，因此：

- 上游不再构造 privileged teacher 工作；
- actor 不执行 teacher forward；
- 不执行额外 scoring pass；
- policy loss wrapper 直接返回原生 GRPO loss。

这叫 objective-level pure GRPO。optimizer 的历史 moments 不会被重置，所以不是把训练状态恢复成“从零开始的 GRPO”。

---

## 12. 第八步：mask 只选位置，训练仍用完整 KL

### 12.1 完整目标

记全局 valid response token 总数为 $N_{\rm valid}$。实际目标可写为：

\[
\boxed{
L_t=L_{\rm GRPO}
+\frac{\lambda}{N_{\rm valid}}
\sum_i m_i(t)w_i
D_{\rm KL}(\tilde q_i\|\tilde p_i),
\qquad \lambda=1.
}
\]

其中 GRPO 始终在全部 valid token 上计算；$m_i$ 只控制是否额外添加 OPSD。

### 12.2 代码实现

loss wrapper 位于基础 runtime `core_runtime/patch.py:772-893`：

```python
grpo_loss, metrics = module.compute_policy_loss_vanilla(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    response_mask=response_mask,
    rollout_is_weights=rollout_is_weights,
    ...
)

student_distill, teacher_distill = module._sd_prepare_sparse_topk_log_probs(
    student_topk_log_probs=student_topk,
    teacher_topk_log_probs=teacher_topk.detach(),
    add_tail=True,
)

opsd_per_token = module._sd_generalized_jsd(
    student_distill,
    teacher_distill,
    mix=0.0,
    vocab_clip=None,
)
if rollout_is_weights is not None:
    opsd_per_token = opsd_per_token * rollout_is_weights

opsd_loss = module.agg_loss(
    loss_mat=opsd_per_token * selected_mask,
    loss_mask=response_mask,
    loss_agg_mode="token-mean",
    **config.global_batch_info,
)

total_loss = grpo_loss + opsd_lambda * opsd_loss
```

### 12.3 为什么 token 越少不会把剩余 token 放大

这里传给 `agg_loss` 的 `loss_mask` 仍是完整 `response_mask`，而不是
`selected_mask`。所以分母是全部 valid token 数：

\[
L_{\rm OPSD}
=\frac{1}{N_{\rm valid}}
\sum_i m_iw_i\operatorname{KL}_i.
\]

如果改成 selected-token mean：

\[
\frac{1}{\sum_i m_i}\sum_i m_i\operatorname{KL}_i,
\]

那么 token 越少，每个剩余 token 的系数越大，不能称为单纯按 token 数退火。当前实现特意没有这样做。

### 12.4 没有投影 loss

scoring pass 中计算的 $a_i,b_i,s_i,m_i$ 全部 detached。反向传播的是
$D_{\rm KL}(\tilde q_i\|\tilde p_i)$ 的完整梯度 $d_i$，不是：

- $d_i^\perp$；
- $a_i\hat r_i$；
- $s_i d_i$；
- 负分对应的负 KL；
- 修改后的 GRPO gradient。

即使一个冲突 token 在早期全预算阶段被选中，它也会接受普通的正权重 forward KL。负分只表示它在预算下降时更早被移除。

---

## 13. 第九步：一次 optimizer update 的完整时序

下面按代码实际发生顺序展开。

### 阶段 A：进入 actor update

`DataParallelPPOActor.update_policy()` 被 runtime 包装：

1. 从 `data.meta_info["global_steps"]` 读取当前 step；
2. 计算 $\kappa_t$ 和 teacher 是否 active；
3. 检查 `use_prefix_grouper=true`；
4. 检查 batch 内存在 `uid`；
5. 创建线程局部 `deque()` 作为 mask FIFO；
6. 调用原始 `update_policy()`；
7. 返回前检查 queue 必须被完全消费；
8. `finally` 中清空状态，避免污染下一步。

代码：`core_runtime/patch.py:912-937`。

### 阶段 B：形成 PPO mini-batch

原生 actor 先按 `ppo_mini_batch_size=64` 切 PPO mini-batch，见
`dp_actor.py:1152-1162`。

### 阶段 C：动态形成真实 training microbatch

`_prepare_update_micro_batches()` 根据 student 和 teacher 的有效长度/workload 做动态平衡，返回 `micro_batches`。GIFT 的 microbatch scope 就在这里确定；不是 YAML 里的固定 `ppo_micro_batch_size_per_gpu=4` 直接决定最终 record 边界。

### 阶段 D：额外 no-grad scoring pass

overlay 在原训练循环开始前，对这个 PPO mini-batch 的全部动态 microbatch 先执行 `_score_micro_batches()`：

1. clone 当前 record 到设备；
2. 检查 `uid` 和 teacher privileged tensors；
3. teacher 在 privileged context 上 no-grad forward；
4. teacher 返回 top-100 log-probs 和 indices；
5. student 在相同 indices 上 no-grad forward；
6. 临时启用 `sum_pi_squared` 统计；
7. 用 $A,\rho,h,w,p,q,u$ 计算解析几何；
8. 把 score 和诊断张量 detach 后搬到 CPU；
9. 对每个实际 microbatch 独立 Top-K；
10. 按训练 microbatch 顺序把 mask 放入 FIFO queue。

代码：`core_runtime/patch.py:542-692` 和
`gift_runtime/patch.py:312-527`。

### 阶段 E：正常可微 training pass

然后才进入原生的 `for micro_batch in micro_batches`：

1. 再做一次 teacher forward，获得用于真实 OPSD loss 的 top-k；
2. 再做一次 student forward，这次建立 autograd graph；
3. 调用被注册的 `opsd_grpo_curriculum` policy loss；
4. loss wrapper 从 queue 头部 pop 对应 mask；
5. 算原生 clipped GRPO；
6. 算 selected-token forward KL；
7. 两项相加；
8. 对这个 microbatch 执行 backward；
9. 所有动态 microbatch 梯度累计结束后做一次 optimizer step。

原生训练循环对应 `dp_actor.py:1203-1210`、`1368-1395`、`1518-1615`。

### 阶段 F：次数应该怎样描述

active step 不是“只多一次普通 forward”这么简单。更精确地说，每个 training microbatch 有两个阶段：

```text
scoring phase:  teacher no-grad forward + student no-grad forward
training phase: teacher no-grad forward + student grad-enabled forward + backward
```

即 active 时 teacher 和 student 各 forward 两次，但只有正常 training student forward 建图，且只有一次 backward 链路。没有为了 GIFT 几何额外做 autograd backward。

step 78 起 scoring phase 和 teacher forward 都不再执行，只剩正常 GRPO student forward/backward。

---

## 14. 第十步：mask FIFO 如何防止 scoring/training 错位

scoring pass 与 training pass 分离后，最大的工程风险是“给 microbatch A 算的 mask 被 microbatch B 消费”。实现采用线程局部 FIFO，并在消费时做以下检查：

1. queue 不能为空；
2. mask shape 必须等于当前 `response_mask.shape`；
3. 每一行 valid token 数的 tuple 必须相同；
4. mask 不能落到 padding；
5. selected token 总数必须与记录值一致；
6. 一个 PPO mini-batch 开始前，上一批 queue 必须已空；
7. 整个 actor update 返回前，queue 必须已全部消费。

核心代码在 `core_runtime/patch.py:695-765`：

```python
item = queue.popleft()
mask = item["mask"].to(response_mask.device)

if mask.shape != response_mask.shape:
    raise RuntimeError(...)

signature = tuple(int(x) for x in response_mask.bool().sum(dim=1).cpu().tolist())
if signature != item["valid_signature"]:
    raise RuntimeError(...)

if (mask.bool() & ~response_mask.bool()).any():
    raise RuntimeError(...)
```

这里的 `valid_signature` 只是“每行 valid token 数”的 tuple，不是 token ID hash。因此它能发现 shape、行长度顺序和 padding 错位，但不能保证检测两个等长 response 行之间的任意互换。文档不能把它夸大成完整内容校验。

---

## 15. entropy、cosine、内积在这里分别有没有用

### 15.1 entropy

GIFT selector 完全不接收 entropy 参数。单元测试还显式检查：

```python
self.assertNotIn(
    "entropy",
    inspect.signature(sparse_tail_gradient_geometry).parameters,
)
```

正常 training path 可能为了已有 teacher metric 计算/记录 teacher entropy，但这不代表 entropy 参与 GIFT 排序。日志中的：

```text
gift/ranking_uses_entropy = 0
gift/uses_entropy = 0
```

用于把这个事实固定下来。

### 15.2 cosine

cosine 是中间量：

\[
c_i=\frac{\langle r_i,d_i\rangle}{\|r_i\|\|d_i\|}.
\]

它用于恢复：

\[
a_i=c_i\|d_i\|,
\qquad
b_i=\|d_i\|\sqrt{1-c_i^2}.
\]

因此该方法不是“只用 cosine 正负”，也不是“按 cosine 大小排名”。OPSD 范数 $\|d_i\|$ 被重新乘回来了。

### 15.3 raw inner product

raw dot $\langle r_i,d_i\rangle$ 也只是中间统计。若直接用 raw dot 排名，$\lvert A_i\rvert$、$\rho_i$、$w_i^2$ 和两边 norm 都会混在一起；当前方案有意把 GRPO 只当参考方向，再按 teacher 的平行/垂直实际长度排序。

因此准确概括是：

```text
不是 entropy selector
不是 cosine-only selector
不是 raw-inner-product selector
而是 normalized-GRPO-reference + unnormalized-teacher-components selector
```

---

## 16. 日志：每一个关键指标怎样读

### 16.1 schedule 与数量

| 指标 | 含义 |
|---|---|
| `gift/target_budget_ratio` | 当前公式给出的 $\kappa_t$ |
| `gift/actual_retention_ratio` | 当前 record 实际 selected / valid；候选不足或 floor 会使其低于 target |
| `gift/selected_token_count_local` | 当前 training microbatch 被选 token 数 |
| `gift/selected_token_count_global` | distributed all-reduce 后的 selected 数，仅日志用途 |
| `gift/valid_token_count_local` | 当前 training microbatch 的 valid response token 数 |
| `gift/eligible_ratio` | valid 中 finite 且 OPSD 非零的候选比例 |
| `gift/budget_fill_ratio` | actual selected / budget；低于 1 表示候选不足 |
| `gift/negative_adv_valid_ratio` | valid token 中负 advantage 的比例 |
| `gift/negative_adv_selected_ratio` | selected token 中负 advantage 的比例，可检查分段评分是否产生符号偏置 |
| `gift/action_in_teacher_topk_ratio` | rollout action 落在 teacher top-100 中的比例 |
| `gift/teacher_forward_enabled` | 该 step 是否执行 teacher 路径 |
| `gift/lambda` | 应始终为 1 |

### 16.2 几何与分支

| 指标 | 含义 |
|---|---|
| `gift/conflict_valid_ratio` | valid token 中 $a<0$ 的比例 |
| `gift/conflict_selected_ratio` | selected token 中冲突 token 的比例；预算下降后理论上应降低 |
| `gift/nonconflict_selected_ratio` | selected 中非冲突与 neutral 的比例 |
| `gift/neutral_grpo_valid_ratio` | valid 中 GRPO 方向不可定义但 OPSD 非零的比例 |
| `gift/neutral_grpo_selected_ratio` | selected 中 neutral token 的比例 |
| `gift/direction_defined_ratio` | valid 中能够真正做分解的比例 |
| `gift/teacher_zero_excluded_ratio` | valid 中因 OPSD norm 过小而排除的比例 |
| `gift/parallel_mean` | valid 上 $a_i$ 的均值；neutral/排除位置在实现中为 0 |
| `gift/parallel_selected_mean` | selected 上 $a_i$ 的均值 |
| `gift/cosine_selected_mean` | selected token 的 cosine 均值，不是最终 score |
| `gift/residual_selected_mean` | selected token 的 $b_i$ 均值 |
| `gift/grpo_norm_selected_mean` | selected token 的 $\|r_i\|$ 均值，仅诊断，不参与最终分段排名 |
| `gift/opsd_norm_selected_mean` | selected token 的 $\|d_i\|$ 均值 |

`score_mean` 是在所有 valid 位置上求均值；不 eligible 和 neutral 位置的 score 在 tensor 中是 0，因此它不是“仅候选 token 的 score 均值”。

### 16.3 分配集中度

| 指标 | 含义 |
|---|---|
| `gift/zero_token_rollout_fraction` | 当前局部范围内完全没有拿到 OPSD token 的 rollout 比例 |
| `gift/rollout_allocation_cv` | 每条 rollout 获得 token 数的变异系数 |
| `gift/zero_eligible_scope_fraction` | 完全没有 eligible token 的 microbatch scope 比例 |
| `gift/scope_count_local` | 本次 mask 构造包含的实际 microbatch scope 数 |
| `gift/scope_is_group` | microbatch 版本应为 0 |

若 `zero_token_rollout_fraction` 很高且 `allocation_cv` 很大，说明 microbatch 全局竞争把监督集中到了少数 rollout；这可能是预算效率，也可能造成覆盖不足，需要结合验证分数判断。

### 16.4 固定语义哨兵

以下指标应固定：

```text
gift/ranking_uses_entropy = 0
gift/loss_uses_projected_gradient = 0
gift/lambda_is_constant = 1
gift/all_valid_token_denominator = 1
```

如果它们不符合预期，应首先怀疑加载了错误 runtime。

---

## 17. 为什么这样设计

### 17.1 为什么不是 hard gate

hard gate 会在训练第一步就彻底丢掉所有 $a<0$ token。GIFT 分段版选择：

- 早期 $\kappa=1$：允许完整 teacher 监督，包括冲突 token；
- 中期预算收缩：强冲突先退出，弱冲突后退出；
- 后期：只保留非冲突且 residual 大的 token；
- 最终：所有 OPSD 全部退出。

它检验的是“冲突监督是否应该渐退”，而不是预设“任何局部冲突都必须立即禁止”。

### 17.2 为什么冲突分支不用 residual 补偿

如果把冲突程度和 residual 简单相加，大 residual 可能掩盖很强的反向平行分量。分段区间把二者隔开：

- 冲突 token 永远是负分；
- 非冲突 token 永远是非负分。

因此逆向 token 无法靠大 $b_i$ 越过非冲突 token，也不需要再调一个水平/垂直混合系数。

### 17.3 为什么非冲突分支看 residual

在不冲突的 token 中，较大的 $b_i$ 表示 teacher 提供了更多不共线的局部 logit-space 信息。它是“补充程度”的代理，而不是 teacher 正确性证明。

### 17.4 为什么 microbatch scope

microbatch scope 允许预算流向当前实际计算单元里分数最高的 token，不强制每道题拿到同样比例。这可能提高预算利用率，且最接近原始 microbatch-rank 的竞争方式。

代价是：

- 排名可能受题目难度、response 长度、OPSD norm 尺度影响；
- 某些 rollout 可能没有监督；
- 动态 batching 边界变化会改变竞争对手集合。

这正是需要与 `prompt_group` scope 做对照的原因。

### 17.5 为什么分母用全 valid token

这样 $\kappa_t$ 下降时，辅助项总体贡献自然减弱；不会因为被选 token 变少而把每个 token 放大。这使“只退火 token 数”在 objective 上有清楚含义。

---

## 18. 与几个容易混淆的版本对比

| 版本 | scope | selector | 退火 | 是否回传投影梯度 |
|---|---|---|---|---|
| GIFT Microbatch token-only（本文） | actual actor microbatch | 冲突 `a` / 非冲突 residual 的分段分数 | $\kappa:1\to0$，$\lambda=1$ | 否 |
| GIFT Group token-only | 同题 8 rollouts | 完全相同的分段分数 | 完全相同 | 否 |
| GIFT Microbatch staged | actual actor microbatch | 完全相同的分段分数 | 先减 token，后减权重 | 否 |
| Conflict-Free Direction | prompt group | 正 cosine hard gate 后按 residual | token-only，初始覆盖率不同 | 否 |
| Conflict-Free Strength | prompt group | 正 cosine hard gate 后按 $\|r\|\,b$ | token-only | 否 |
| 原始 TIP microbatch-rank | microbatch | Q3/entropy-divergence 系列分数 | 原生 token→weight schedule | 否 |

本文版本与 Group token-only 的代码级差异主要只有：

```yaml
gift_selection_scope: micro_batch
```

以及独立的输出目录/实验名。二者的评分、loss、schedule 和其他训练参数保持一致。

---

## 19. 分布式与 Ray 注入链路

### 19.1 为什么要做 runtime overlay

用户要求不覆盖项目代码，所以实验逻辑放在新目录，通过 Python import hook 注入。`sitecustomize.py` 在进程启动时检查：

```python
if GIFT_ENABLE:
    from gift_runtime.patch import apply_runtime_patches
    apply_runtime_patches()
```

### 19.2 driver 和 worker 都必须注入

启动脚本同时使用：

- `PYTHONPATH`；
- 临时 `.pth`；
- `sitecustomize`；
- Ray `worker_process_setup_hook`；
- Ray runtime env vars。

原因是只在启动端 import 成功不代表远端 Ray worker 也加载了同一模块。`verify_gift_ray_workers.py` 会在每个 live node 上检查：

- `verl` 是否来自本次源码包；
- GIFT/base runtime 路径是否正确；
- geometry、selector、actor、loss hook 是否生效；
- 其他相似 overlay 是否全部关闭；
- step 78 的 retention 是否为 0；
- $\lambda$ 是否为 1。

### 19.3 运行目录布局

启动脚本假定仓库根目录下的布局为：

```text
scripts/gift/main        # 本方法：gift_runtime、YAML、run/verify/test
scripts/gift/main        # GIFT：gift_runtime、core_runtime、YAML、run/verify/test
```

`run_gift.sh` 与 `verify_gift_ray_workers.py` 均按此相对布局拼接 `PYTHONPATH`，各 Ray 节点需使用同一份仓库副本（或同一共享文件系统）。

### 19.4 排序没有跨 rank 通信

每个 actor rank 独立构造 microbatch Top-K。唯一相关的 distributed all-reduce 是 selected token 数量指标，不会改变 mask。因此不能把该方法描述为 global-batch ranking。

---

## 20. 验证和测试覆盖了什么

### 20.1 单元测试

`test_gift.py` 当前覆盖 11 项：

1. advantage 正负号切换冲突/非冲突分支；
2. 同符号时改变 $\lvert A\rvert$ 不改变 GIFT score；
3. 解析 $a,b$ 与真实 logit autograd 一致；
4. $r=0$ 使用 neutral score 0；
5. $d=0$ 被排除；
6. PPO clip inactive 使用 neutral 回退；
7. selector 接口没有 entropy；
8. group scope 每题独立预算；
9. microbatch scope 可以跨题转移预算；
10. 分段排序会先移除强冲突；
11. full budget、精确 schedule 和固定 $\lambda$ 契约。

本次文档编写时实际运行结果：

```text
Ran 11 tests in 0.676s
OK
```

### 20.2 配置 preflight

`verify_gift.py` 会逐字段验证：

- 模型/训练集/验证集路径；
- group size、batch、长度、资源；
- top-k、KL 参数；
- step 1、10、11、44、77、78、96 的 retention；
- schedule 单调不增；
- 每一步 $\lambda=1$；
- `val_before_train=false`；
- save/eval 间隔为 10；
- 输出目录在新任务启动前不存在。

这类检查采用 fail-closed：配置不一致就拒绝启动，不静默使用默认值。

### 20.3 4B 和 1.7B 是否是另一套算法

不是。目录中的：

```text
qwen3_4b_gift_microbatch_token_anneal.yaml
qwen3_1_7b_gift_microbatch_token_anneal.yaml
```

复用同一 GIFT runtime。`verify_gift_model_scale_math.py` 会把它们与 8B reference config 逐字段比较，只允许以下模型身份字段变化：

- `model_path`；
- `output_dir`；
- `project_name`；
- `experiment_name`。

所以 4B/1.7B 是模型规模复现，不是 selector、scope、退火或 loss 的新变体。

---

## 21. 计算和显存成本

active 阶段相对普通 OPSD-GRPO 多出一个 no-grad scoring phase：

- 多一次 teacher forward；
- 多一次 student forward；
- 多一次 top-k + tail 解析几何和 CPU Top-K；
- 不增加 backward 次数；
- 不保留额外 full-vocabulary autograd graph；
- 只临时计算 $\sum p^2$，保存的是 `[B,L]` 充分统计量和 sparse top-k。

正常 training phase 本身仍需要 teacher + student forward。故活跃阶段的主要代价是 forward 计算，而不是参数梯度存储。

step 78–96 没有 teacher/scoring，成本回落到纯 GRPO 路径。

---

## 22. 理论边界：这套方法不能声称什么

### 22.1 output-logit 正交不等于参数空间正交

若当前 token 的 logit Jacobian 为：

\[
J_i=\frac{\partial z_i}{\partial\theta},
\]

参数梯度是：

\[
\nabla_\theta L_{R,i}=J_i^\top r_i,
\qquad
\nabla_\theta L_{D,i}=J_i^\top d_i.
\]

参数空间内积为：

\[
r_i^\top J_iJ_i^\top d_i,
\]

通常不等于 $r_i^\top d_i$。而且全 batch 还有不同 token 间的
$r_i^\top J_iJ_j^\top d_j$ 交叉项。因此这里只能称为
**token-local output-logit-space geometry proxy**。

### 22.2 非冲突不等于 teacher 正确

$a_i\ge0$ 只说明两个局部 loss gradient 在这一度量下没有反向投影，不证明：

- teacher 内容正确；
- 该 token 是 task-bearing token；
- 加 KL 后 reward 必然提高；
- 大 residual 必然比小 residual 更有价值。

这些都必须由消融和验证集结果回答。

### 22.3 早期选择中允许冲突

因为 step 1–10 是 full budget，所有 eligible 冲突 token 也会被蒸馏。因此不能写“GIFT 保证所有 selected token 与 GRPO 同向”。准确主张是：

> 随预算下降，GIFT 按局部冲突强度优先撤掉冲突监督，并在非冲突集合中优先保留正交残差较大的 token。

### 22.4 score 每一步重算

模型、ratio、clip 状态和 teacher/student 分布会变化，所以 token 当前退出不代表相同位置以后永不进入。所谓“强冲突先退出”是每一步当前候选池中的优先级。

### 22.5 microbatch 边界是算法的一部分

改变动态 batch 的 token 上限、teacher 输入长度或 workload partition，可能改变谁与谁竞争，即使 score 公式不变也会改变 mask。复现实验时必须同时固定 batching 参数。

---

## 23. 可执行的算法伪代码

```python
for actor_step in range(1, 97):
    kappa = retention(actor_step)       # 1 -> 0
    lam = 1.0                           # 始终不变

    if kappa == 0:
        # step 78..96
        loss = native_clipped_grpo(all_valid_tokens)
        backward_and_step(loss)
        continue

    for ppo_minibatch in actor_update_batch:
        microbatches = dynamic_partition(
            ppo_minibatch,
            student_teacher_workload,
            max_tokens=24576,
        )

        queue = []

        # Phase 1: detached scoring
        with no_grad():
            records = []
            for mb in microbatches:
                q_topk, topk_ids = teacher_forward_privileged(mb, k=100)
                p_topk, p_action, sum_p2 = student_forward_aligned(
                    mb, topk_ids
                )

                rho = exp(logp_action - old_logp_action)
                h = clipped_surrogate_active(advantage, rho)
                r_geometry = A * rho * h * rollout_is * (p - one_hot(action))
                d_geometry = rollout_is * grad_forward_kl_topk_plus_tail(q, p)

                if d_geometry is zero or any required value is nonfinite:
                    eligible = False
                elif r_geometry is zero:
                    eligible = True
                    score = 0
                else:
                    r_hat = r_geometry / norm(r_geometry)
                    a = dot(d_geometry, r_hat)
                    b = norm(d_geometry - a * r_hat)
                    score = a if a < 0 else b

                records.append(detach_to_cpu(score, eligible, diagnostics))

            for mb_record in records:  # micro_batch scope
                valid_ids = all_valid_response_tokens(mb_record)
                candidate_ids = valid_ids[eligible]
                K = floor(kappa * len(valid_ids))
                K_actual = min(K, len(candidate_ids))
                mask = topk_mask(score[candidate_ids], K_actual)
                queue.append(mask)

        # Phase 2: ordinary differentiable training
        zero_grad()
        for mb in microbatches:
            mask = queue.pop_left_with_checks(mb.response_mask)
            q_topk, topk_ids = teacher_forward_privileged(mb, k=100)
            p_topk, logp_action = student_forward_with_grad(mb, topk_ids)

            grpo = native_clipped_grpo(all_valid_tokens)
            opsd = sum(mask * rollout_is * forward_kl(q_topk_tail, p_topk_tail)) \
                   / global_valid_token_count
            loss = grpo + lam * opsd
            loss.backward()

        assert queue_is_empty()
        optimizer_step()
```

该伪代码省略了 FSDP/data-parallel scale、日志和 tensor 搬运，但保留了当前算法的全部决定性语义。

---

## 24. 一个小型排序例子

假设某个实际 microbatch 有 6 个 valid、eligible token，当前
$\kappa=0.5$，所以：

\[
K=\lfloor0.5\times6\rfloor=3.
\]

其几何为：

| token | $a$ | $b$ | GIFT score | 解释 |
|---|---:|---:|---:|---|
| 1 | -0.90 | 1.20 | -0.90 | 强冲突；大 residual 不能救它 |
| 2 | -0.10 | 0.80 | -0.10 | 弱冲突 |
| 3 | 0 | 0 | 0 | GRPO 零方向的 neutral 回退 |
| 4 | +0.20 | 0.20 | 0.20 | 非冲突，小补充 |
| 5 | +0.30 | 0.80 | 0.80 | 非冲突，大补充 |
| 6 | +1.10 | 0.40 | 0.40 | 强同向但补充中等 |

降序为：

```text
token 5 (0.80) > token 6 (0.40) > token 4 (0.20)
> token 3 (0) > token 2 (-0.10) > token 1 (-0.90)
```

最终选 5、6、4。注意 token 6 虽然平行同向分量很大，最终仍按 $b=0.40$ 排，而不是按 $a=1.10$ 排；这是当前 GIFT 定义的有意选择。

在 $\kappa=1$ 时，6 个都会被选；冲突只决定后续谁先退出。

---

## 25. 复现与自查命令

从仓库根目录运行 CPU 单元测试：

```bash
PYTHONPATH="scripts/gift/main:\
third_party/verl:." \
python3 scripts/gift/main/test_gift.py
```

只做 YAML 合同校验：

```bash
PYTHONPATH="scripts/gift/main:\
third_party/verl:." \
python3 scripts/gift/main/verify_gift.py \
  --config scripts/gift/main/\
qwen3_8b_gift_microbatch_token_anneal.yaml
```

启动训练（每个节点各执行一次；多节点变量见脚本头部注释）：

```bash
bash scripts/gift/main/run_gift.sh \n  scripts/gift/main/qwen3_8b_gift_microbatch_token_anneal.yaml
```

启动脚本会先校验输入与新输出目录，再启动 Ray 与训练。

---

## 26. 最后用人话复述一遍

可以把每个 token 想象成 teacher 提出的一条“纠偏建议”。GRPO 当前也有一条自己的更新箭头。

- teacher 箭头若明显顶着 GRPO 走，就给负分；顶得越厉害，负分越低；
- teacher 箭头若没有顶着 GRPO 走，就看它侧向带来了多少新东西；侧向分量越大，分数越高；
- 开始 10 步预算足，几乎所有有用 teacher 建议都接受；
- 接下来不断缩小名额，先删强冲突，再删弱冲突，再从补充小到补充大逐渐删；
- 到 step 78，一个 teacher token 都不再用，完全回到 GRPO；
- 整个过程中 teacher 建议本身的权重不变，改变的只是还有多少 token 能获得这条建议。

而“Microbatch”表示：每次是在一个真实动态 training microbatch 内统一竞争这些名额。它允许不同问题互相抢预算，但不会跨 microbatch 或跨设备做全局排名。

这就是当前 **GIFT Microbatch token-only** 的完整实现语义。
