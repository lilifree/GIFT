# GIFT Microbatch Token-only：完整人话版说明

> 本文只解释方法思想、数学含义和训练流程，不展开代码实现。
>
> 对应方法：**GIFT 分段梯度排序 + microbatch 内竞争 + 只减少蒸馏 token 数量**。

---

## 1. 一句话概括

Student 仍然照常用 GRPO 学习；与此同时，privileged teacher 在每个 response 位置给出一份 OPSD 监督。GIFT 不把所有 teacher 监督一直保留到训练结束，而是：

1. 先判断 teacher 在这个位置的梯度与 GRPO 是冲突还是不冲突；
2. 冲突位置按“冲突有多强”排序；
3. 不冲突位置按“teacher 还能提供多少 GRPO 没覆盖的垂直信息”排序；
4. 在每个真实 training microbatch 内统一竞争蒸馏名额；
5. 训练开始时基本全选，随后逐步减少被蒸馏的位置，最后完全退出 teacher，只剩纯 GRPO。

最简洁的口号是：

> **强冲突先退出，弱冲突随后退出，非冲突且补充信息大的 token 最晚退出。**

这里的 token 指的是 **response 中的位置**，不是词表中的 100 个候选词。

---

## 2. 先分清三种容易混淆的“token”

这个方法里同时出现三种 token 概念，如果混在一起，后面的 top-100、Top-K 和退火都会很难理解。

### 2.1 Rollout 中实际生成的 token

Student 先生成一条 response，例如：

> Therefore / the / answer / is / 42

这里一共有 5 个 response 位置。GRPO 会对这些实际采样动作计算 policy-gradient loss。

### 2.2 某个位置上的词表候选

在“answer”这个位置，模型不是只知道最终采样出的一个词。它在整个词表上都有一份概率分布，例如：

- `answer`：0.30；
- `result`：0.12；
- `number`：0.08；
- 其他数万个词也各有概率。

这些是词表坐标，不是多个 response 位置。

### 2.3 Teacher top-100

Teacher top-100 指：在每一个 response 位置，teacher 从完整词表中找出自己概率最高的 100 个词表 ID。

它的作用是压缩一份巨大词表分布，方便保存和计算 OPSD。它不表示：

- 最多只能蒸馏 100 个 response 位置；
- teacher 生成了 100 个 token；
- student 的 top-100 必须和 teacher 自己的 top-100 相同；
- GIFT 每个 microbatch 只能选 100 个位置。

GIFT 后面做 Top-K 时，选的是“哪些 response 位置接受 OPSD”，与这里的词表 top-100 是两个完全不同的维度。

---

## 3. 整个训练过程在做什么

可以把一轮训练理解成下面八步。

### 第一步：Student 生成 rollout

Student 根据题目生成若干条 response。同一道题的多条 rollout 构成 GRPO 的 group。

### 第二步：GRPO 计算每条 rollout 的相对好坏

同题 group 内的 response 根据 reward 形成相对 advantage：

\[
A_{g,r}
=
\frac{R_{g,r}-\operatorname{mean}_{r'}R_{g,r'}}
{\operatorname{std}_{r'}R_{g,r'}+\epsilon}.
\]

因此：

- 高于同组平均水平的 response 通常有正 advantage；
- 低于同组平均水平的 response 通常有负 advantage；
- 一条 response 内，advantage 通常复制到它的每个 valid token 位置。

### 第三步：Privileged teacher 阅读同一条 rollout

Teacher 不需要重新生成另一条 response。它获得题目和标准 solution/answer 等 privileged context，然后对 student 已经生成的 rollout 做对齐 forward。

所以 student 和 teacher 比较的是：

- 同一道题；
- 同一条 rollout 历史；
- 同一个输出位置；
- 但 teacher 多看到了标准解答上下文。

Teacher 给出的不是“这个位置唯一正确的词”，而是一整份条件概率分布。

### 第四步：构造 top-100 + tail 的 OPSD 目标

Teacher 在每个位置选自己的 top-100 词表坐标。Student 随后在完全相同的 100 个词表 ID 上读取自己的概率。

“相同坐标”只表示比较对象一一对应，不表示概率相同，也不表示双方最终生成相同 token。

### 第五步：分别得到 GRPO 和 OPSD 的局部梯度

在每个 response 位置，GIFT 构造两个 output-logit-space 的 loss gradient：

- GRPO 梯度 \(r_i\)：RL 在这个位置希望怎样改变分布；
- OPSD 梯度 \(d_i\)：teacher 蒸馏在这个位置希望怎样改变分布。

### 第六步：根据两个梯度的几何关系给位置打分

GIFT 先以 GRPO 梯度为参考方向，把 OPSD 梯度分成：

- 沿 GRPO 方向的平行分量；
- 垂直于 GRPO 方向的正交残差。

冲突位置和不冲突位置采用不同的评分规则。

### 第七步：在每个 microbatch 内选 Top-K response 位置

每个真实 training microbatch 单独拥有一份预算。同一 microbatch 内的不同题目、不同 rollout、不同 response 位置共同竞争。

### 第八步：选中位置使用原始 OPSD KL，所有位置继续使用 GRPO

GIFT 的分解只负责决定 mask，不会把训练梯度真的投影掉。训练时：

- GRPO 仍作用于全部 valid response token；
- 只有被 mask 选中的位置额外加入完整的 OPSD loss；
- 预算最终降到 0 后，不再执行 teacher 路径，只训练 GRPO。

---

## 4. OPSD 的 top-100 + tail 到底是什么

### 4.1 Teacher 和 student 在同一组词表坐标上比较

设位置 \(i\) 上：

- student 完整词表分布为 \(p_i\)；
- teacher 完整词表分布为 \(q_i\)；
- teacher 选出的 top-100 词表 ID 集合为 \(T_i\)。

对每个 \(j\in T_i\)，比较的是：

\[
p_{ij}\quad\text{和}\quad q_{ij}.
\]

例如 teacher 的 top-100 中含有词表 ID `42`，那么就比较：

- teacher 给 `42` 的概率；
- student 给同一个词表 ID `42` 的概率。

Student 自己概率最高的 100 个词可以是另一组。这不影响比较，因为坐标集合由 teacher 确定。

### 4.2 Tail 的作用

Teacher top-100 之外仍有大量概率，不能直接丢掉，否则压缩后的概率总和不再是 1。

所以把 top-100 之外的全部词表坐标合并成一个 tail bucket：

\[
P_{\mathrm{tail}}
=1-\sum_{j\in T_i}p_{ij},
\qquad
Q_{\mathrm{tail}}
=1-\sum_{j\in T_i}q_{ij}.
\]

压缩后的 student 和 teacher 分布分别是：

\[
\tilde p_i=
\bigl(\{p_{ij}\}_{j\in T_i},P_{\mathrm{tail}}\bigr),
\]

\[
\tilde q_i=
\bigl(\{q_{ij}\}_{j\in T_i},Q_{\mathrm{tail}}\bigr).
\]

直观上，tail 只回答一个问题：

> 双方一共愿意给“teacher top-100 之外的所有词”多少概率？

它不再区分 tail 内究竟是哪一个具体词。

### 4.3 OPSD 使用 forward KL

每个位置的 OPSD loss 是：

\[
\ell_{\mathrm{OPSD},i}
=D_{\mathrm{KL}}(\tilde q_i\|\tilde p_i).
\]

展开后：

\[
\ell_{\mathrm{OPSD},i}
=
\sum_{j\in T_i}
q_{ij}\log\frac{q_{ij}}{p_{ij}}
+Q_{\mathrm{tail}}
\log\frac{Q_{\mathrm{tail}}}{P_{\mathrm{tail}}}.
\]

它表达的是：让 student 的压缩分布向 privileged teacher 的压缩分布靠近。

必须准确地说，这是 **top-100 + tail 压缩空间中的 forward KL**，不是完整词表 KL。它保留了 teacher 高概率词的细粒度信息，以及剩余词表的总概率质量，但丢掉了 tail 内部的细粒度差异。

### 4.4 Rollout 实际 token 不在 teacher top-100 时怎么办

对 OPSD 压缩目标而言，这个 token 位于 tail 中，不作为单独类别比较；但它没有从 GRPO 中消失：GRPO 仍然把它当作本位置真实采样的动作 \(u_i\)，精确计算其 policy gradient。

所以：

- OPSD 在此处看到的是 tail 总质量；
- GRPO 仍知道 student 实际采样了哪个具体 token。

---

## 5. 两个局部梯度分别表示什么

本文统一使用“loss gradient”记号，优化器实际沿负梯度方向更新。

### 5.1 GRPO 梯度

位置 \(i\) 的解析形式可以写成：

\[
\boxed{
r_i=A_i\rho_i h_iw_i(p_i-e_{u_i})
}
\]

其中：

- \(A_i\)：这条 response 的 GRPO advantage；
- \(\rho_i\)：current policy 与 old policy 在已采样 token 上的概率比；
- \(h_i\in\{0,1\}\)：当前 PPO/dual-clip surrogate 是否仍有梯度；
- \(w_i\)：rollout importance-sampling 修正；
- \(u_i\)：student 在该位置实际采样到的 token；
- \(e_{u_i}\)：该 token 对应的 one-hot 向量；
- \(p_i-e_{u_i}\)：action log-prob loss 在 logit 空间的基本几何方向。

人话解释：\(r_i\) 表示，在当前局部 PPO surrogate 下，GRPO 希望怎样调整这个位置的 student logits。

### 5.2 PPO clipping 在这里的作用

Clipping 不会连续地把 advantage 数值改小。它决定当前 surrogate 分支还是否有梯度：

- active 时，\(h_i=1\)；
- 被 clip/dual-clip 截成局部常数时，\(h_i=0\)，当前 GRPO 梯度为零。

因此“clip 是否激活”和“advantage 大小”是两件不同的事。

### 5.3 OPSD 梯度

\[
\boxed{
d_i=w_i\nabla_{z_i}
D_{\mathrm{KL}}(\tilde q_i\|\tilde p_i)
}
\]

人话解释：\(d_i\) 表示，若在这个位置施加 teacher 的 forward KL，student logits 会收到怎样的 loss gradient。

GIFT 不把 \(d_i\) 单位化。因此 teacher 梯度在这个位置本来有多大，会进入后续的平行分量和垂直分量大小。

---

## 6. GIFT 如何把 OPSD 梯度拆开

当 \(r_i\neq0\) 时，先只对 GRPO 梯度归一化：

\[
\hat r_i=\frac{r_i}{\|r_i\|}.
\]

这里 \(\hat r_i\) 只代表 GRPO 的参考方向，不再携带 GRPO 梯度强度。

### 6.1 有符号平行分量

\[
a_i=\langle d_i,\hat r_i\rangle.
\]

它回答的是：teacher 梯度沿 GRPO 参考轴投了多少，以及投在轴的哪一侧。

- \(a_i>0\)：两个 loss gradient 在该轴上同向；
- \(a_i<0\)：两个 loss gradient 在该轴上反向，存在局部冲突；
- \(a_i=0\)：OPSD 梯度在该位置与 GRPO 方向正交。

平行向量为：

\[
d_i^{\parallel}=a_i\hat r_i.
\]

### 6.2 正交残差

把平行分量从完整 OPSD 梯度中减掉：

\[
d_i^{\perp}=d_i-a_i\hat r_i.
\]

它的长度为：

\[
b_i=\|d_i^{\perp}\|.
\]

人话解释：\(b_i\) 衡量 teacher 在这个位置提供了多少“不沿当前 GRPO 局部方向”的信息。

这里必须保持克制：

- 它是 output-logit 空间中的局部几何量；
- 它可以被解释成一种“非共线补充量”；
- 它不能直接证明这些信息一定正确；
- 更不能证明它一定提高最终 reward。

---

## 7. 最终排序分数为什么是分段的

GIFT 的最终分数为：

\[
\boxed{
s_i=
\begin{cases}
a_i=-|a_i|, & a_i<0,\\[4pt]
b_i, & a_i\ge0.
\end{cases}
}
\]

然后按 \(s_i\) 从大到小选择。

### 7.1 冲突 token：按冲突程度降级

当 \(a_i<0\) 时，分数就是负数 \(a_i\)。

- 冲突越强，\(|a_i|\) 越大，分数越负；
- 分数越负，预算减少时越早退出；
- 冲突较弱的 token 会比强冲突 token 多保留一段时间。

例如：

| Token | \(a_i\) | 分数 | 退出优先级 |
|---|---:|---:|---|
| A | -0.90 | -0.90 | 很早退出 |
| B | -0.20 | -0.20 | 比 A 晚退出 |

这里故意没有把冲突 token 直接设成 \(-\infty\)。因此训练早期预算充足时，部分甚至全部冲突位置仍然可以接受 OPSD；退火开始后，再按冲突强弱逐渐撤掉。

### 7.2 非冲突 token：按垂直补充量晋级

当 \(a_i\ge0\) 时，使用 \(b_i\) 排序。

- \(b_i\) 越大，teacher 与 GRPO 不共线的分量越大；
- 这种 token 的分数越高，预算缩小时越晚退出；
- 若 teacher 梯度与 GRPO 完全平行，则 \(b_i=0\)，它排在正残差 token 后面，但仍排在所有负分冲突 token 前面。

### 7.3 整体顺序

从最晚保留到最早退出，整体优先级是：

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

这是一种“监督撤离优先级”，不是 token 正确率，也不是对真实 reward 增益的直接估计。

---

## 8. 排序时到底只用了方向，还是也用了大小

最准确的回答是：

> **GRPO 一侧只使用方向；OPSD 一侧保留实际梯度大小。**

### 8.1 GRPO 一侧

GIFT 使用的是 \(\hat r_i=r_i/\|r_i\|\)，所以对 active 且非零的位置：

\[
\hat r_i
=
\operatorname{sign}(A_i)
\frac{p_i-e_{u_i}}
{\|p_i-e_{u_i}\|}.
\]

因此以下正数尺度会在归一化中约掉：

- advantage 的绝对值 \(|A_i|\)；
- 正的 PPO ratio 幅度 \(\rho_i\)；
- 正的 rollout-IS 尺度 \(w_i\)。

但是它们并非全部“毫无作用”：

- \(\operatorname{sign}(A_i)\) 会翻转 GRPO 参考方向；
- 方向翻转可能把同一个位置从冲突分支切换到非冲突分支；
- \(\rho_i\) 与 advantage 符号共同决定 PPO clipping 是否让 \(h_i=0\)；
- 一旦 \(h_i=0\)，GRPO 方向就不再定义。

### 8.2 OPSD 一侧

\(d_i\) 没有除以 \(\|d_i\|\)，所以：

\[
a_i=\|d_i\|\cos\theta_i,
\qquad
b_i=\|d_i\||\sin\theta_i|.
\]

这说明分数不仅看角度，还保留 teacher 梯度大小：

- 同样的角度下，OPSD 梯度更大的 token 通常有更大的 \(|a_i|\) 或 \(b_i\)；
- 它不是纯 cosine 排序；
- 它也不是两个原始梯度的 raw inner product 排序。

### 8.3 Advantage 大小没有被整个训练丢掉

这里只是 **OPSD token selector** 不使用 \(|A_i|\) 排名。完整 GRPO loss 仍然使用原始 advantage，包括它的大小。

所以应该说：

> GIFT 的选点规则只保留 advantage 的方向作用；GRPO 本身仍保留完整 advantage 强度。

不能说“这个训练不考虑 advantage 大小”。

---

## 9. 为什么不用纯 cosine

如果两边都归一化：

\[
\cos(r_i,d_i)
=
\frac{\langle r_i,d_i\rangle}
{\|r_i\|\|d_i\|},
\]

那么 OPSD 梯度有多大也会被除掉。一个几乎为零的 teacher 梯度和一个很强的 teacher 梯度，只要角度相同，就会得到相同 cosine。

GIFT 只归一化 GRPO 参考方向：

\[
a_i=\langle d_i,\hat r_i\rangle,
\qquad
b_i=\|d_i-a_i\hat r_i\|.
\]

这样做的目的，是把两种职责分开：

- GRPO 提供“参考轴朝哪里”；
- OPSD 提供“沿这条轴有多少、垂直于这条轴有多少”。

所以 GIFT 比纯 cosine 多保留了 teacher 梯度的实际幅度。

---

## 10. 为什么不用原始梯度内积直接排序

原始内积是：

\[
\langle r_i,d_i\rangle.
\]

它同时混入：

- \(|A_i|\)；
- \(\rho_i\)；
- GRPO 基础梯度范数；
- OPSD 梯度范数；
- 两者夹角。

它适合回答“两个完整局部梯度的有符号耦合有多强”，但不容易区分：

1. teacher 是在重复 GRPO 已经会做的方向；
2. teacher 是否还带有明显的非共线部分。

此外，在 microbatch 跨 rollout 排序时，大 \(|A|\) 的 response 可能仅凭尺度占据大量预算。

GIFT 的选择是：

- 用归一化后的 GRPO 方向判断冲突类别；
- 用未归一化 OPSD 分量衡量 teacher 强度；
- 不让 \(|A|\) 直接控制蒸馏名额。

这不是数学上证明一定优于 raw inner product，而是一项清晰的设计取舍，需要用实验比较。

---

## 11. Microbatch 排序范围是什么意思

### 11.1 它不是“同一道题的 group”

GRPO advantage 确实是在同题多条 rollout 的 group 内计算的，但 GIFT 的 token 竞争范围不是 group。

这里的 microbatch 指：actor 在一次 PPO mini-batch 内，根据实际序列长度和 student/teacher workload 动态切出的训练小批次。

一个 microbatch 可能包含：

- 多道不同题目；
- 每道题的若干 rollout；
- 每条 rollout 的许多 response token。

### 11.2 谁和谁竞争

设第 \(m\) 个 microbatch 有 \(N_m\) 个 valid response token，当前预算比例为 \(\kappa_t\)：

\[
K_m(t)=\left\lfloor\kappa_tN_m\right\rfloor.
\]

然后在该 microbatch 的有效候选中，选择分数最大的 \(K_m(t)\) 个位置。

因此：

- 同一条 rollout 可以获得很多 OPSD token，也可以一个都没有；
- 同一 microbatch 内，不同题目可以互相争夺预算；
- 不设置每道题必须拿多少名额；
- 不设置每条 rollout 必须拿多少名额；
- 不同 microbatch 之间不能转移预算；
- 不同 actor rank 之间也不做全局 Top-K。

### 11.3 一个直观例子

某个 microbatch 有 1000 个 valid response token，当前 \(\kappa_t=0.30\)：

\[
K_m=\lfloor0.30\times1000\rfloor=300.
\]

如果其中一道题贡献了 250 个最高分位置，它可以拿走 250 个名额；其他题共享剩下的 50 个。这就是 microbatch 排序允许的预算转移。

### 11.4 候选不足时不会硬凑

以下位置不作为正常候选：

- padding；
- 数值为 NaN/Inf 的异常位置；
- OPSD 梯度近似为零的位置。

如果预算是 300，但只有 270 个有效候选，就只选择 270 个，不会用无效位置补满。

---

## 12. 零梯度和边界情况怎么处理

### 12.1 OPSD 梯度近似为零

当 \(\|d_i\|\) 太小时，teacher 在当前压缩目标上几乎没有更新信号。该位置被排除，不浪费蒸馏名额。

### 12.2 GRPO 梯度近似为零

可能原因包括：

- advantage 为 0；
- PPO/dual-clip 使当前分支 inactive；
- 数值饱和。

此时没有可用的 GRPO 参考方向，不能把它误称为“真正正交”。当前策略是：

- 只要 OPSD 梯度有效，仍允许它作为候选；
- 给它中性分数 \(s_i=0\)；
- 不猜测它究竟同向还是冲突。

### 12.3 真正的正交位置

如果 \(r_i\) 非零且 \(a_i=0\)，这才是几何意义上的正交。此时：

\[
s_i=b_i=\|d_i\|.
\]

所以 teacher 梯度较大的真正正交位置会得到较高分数。

### 12.4 “前 10 步全选”的准确含义

全选是指选择所有 valid、数值正常且 OPSD 梯度非零的候选位置。padding、异常值和零 OPSD 梯度位置仍不会被选中。

---

## 13. Token-only 退火是怎样进行的

当前实验总共训练 96 个 actor step。覆盖率预算为：

\[
\boxed{
\kappa_t=
\begin{cases}
1, & t\le10,\\[4pt]
\dfrac{78-t}{68}, & 10<t<78,\\[8pt]
0, & t\ge78.
\end{cases}
}
\]

OPSD 总体系数始终固定：

\[
\boxed{\lambda_t=1.}
\]

### 13.1 三个阶段

#### Step 1–10：基本全选

\[
\kappa_t=1.
\]

所有有效 OPSD 候选都被选中。此阶段分数已经可以计算，但只要没有候选不足，排序不会改变成员集合。

#### Step 11–77：逐步减少 response token

\(\kappa_t\) 每步线性下降。排序开始真正决定谁退出：

1. 强冲突 token 最先掉出 Top-K；
2. 然后是弱冲突 token；
3. 再到零分和补充量较小的位置；
4. 非冲突且正交残差大的位置保留最久。

#### Step 78–96：完全退出 teacher

\[
\kappa_t=0.
\]

此时：

- 没有 OPSD token；
- 不再进行 teacher forward；
- 不再计算 GIFT 分数；
- 训练目标在 objective 层面变成纯 GRPO。

优化器之前积累的历史状态不会重置，所以它不是重新启动一个全新的 GRPO 训练。

### 13.2 关键步数

| Step | \(\kappa_t\) | 最大预算比例 |
|---:|---:|---:|
| 1–10 | 1.0000 | 100% |
| 11 | 0.9853 | 约 98.53% |
| 20 | 0.8529 | 约 85.29% |
| 30 | 0.7059 | 约 70.59% |
| 40 | 0.5588 | 约 55.88% |
| 44 | 0.5000 | 50% |
| 50 | 0.4118 | 约 41.18% |
| 60 | 0.2647 | 约 26.47% |
| 70 | 0.1176 | 约 11.76% |
| 77 | 0.0147 | 约 1.47% |
| 78–96 | 0 | 纯 GRPO |

实际选择数还要取整：

\[
K_m(t)=\lfloor\kappa_tN_m\rfloor.
\]

因此短 microbatch 可能比比例曲线稍早出现 \(K=0\)。

### 13.3 为什么叫 token-only

因为退火过程中只改变二值 mask 中 1 的数量：

\[
m_i(t)\in\{0,1\}.
\]

被选中的 token 始终使用相同的 OPSD 系数 \(\lambda=1\)。没有：

- 后半程再降低 OPSD 权重；
- 给高分 token 更大的连续 loss 权重；
- 先减少 token，再单独做 weight annealing。

所以它与 staged token→weight 版本不同。

---

## 14. 最终训练目标

可以把目标写成：

\[
\boxed{
\mathcal L_t
=
\mathcal L_{\mathrm{GRPO}}(\text{全部 valid token})
+
\frac{\lambda}{N_{\mathrm{valid}}}
\sum_i m_i(t)w_i
D_{\mathrm{KL}}(\tilde q_i\|\tilde p_i)
}
\]

其中 \(\lambda=1\)。最重要的是分母使用全部 valid response token 数，而不是当前 selected token 数。

### 14.1 为什么分母不能改成 selected-token mean

假设从 1000 个位置减少到 100 个位置：

- 若除以 100，剩余每个位置会被放大 10 倍；
- 若仍除以全部 1000 个 valid token，每个选中位置的系数不变，OPSD 总体影响才会随选择数量下降。

本方法希望“逐 token 撤走 teacher 监督”，所以使用全 valid-token 分母更符合这个含义。

### 14.2 排名只产生 mask，不改变 KL 的方向

GIFT 计算了 \(d_i^\parallel\) 和 \(d_i^\perp\)，但训练时不会只反传 \(d_i^\perp\)。

被选中后，反传的是完整的：

\[
D_{\mathrm{KL}}(\tilde q_i\|\tilde p_i).
\]

因此：

- 投影只用于排序；
- mask 是 stop-gradient 的离散选择；
- GIFT 不做 PCGrad 式梯度手术；
- 不修改 GRPO 梯度；
- 不把负分变成负 KL 权重。

一个负分 token 如果在早期仍被选中，它接受的仍是正常、正权重的 forward KL。

---

## 15. 用一个小例子理解“逐渐退出”

假设一个 microbatch 只有 6 个有效候选位置，它们当前的几何量如下：

| Token | \(a_i\) | \(b_i\) | 最终分数 \(s_i\) | 含义 |
|---|---:|---:|---:|---|
| 1 | -0.80 | 1.20 | -0.80 | 强冲突 |
| 2 | -0.15 | 0.90 | -0.15 | 弱冲突 |
| 3 | — | — | 0.00 | GRPO 零方向的中性回退 |
| 4 | 0.60 | 0.00 | 0.00 | 同向但没有垂直补充 |
| 5 | 0.20 | 0.35 | 0.35 | 非冲突、补充较小 |
| 6 | 0.10 | 0.95 | 0.95 | 非冲突、补充较大 |

排序为：

\[
6\succ5\succ\{3,4\}\succ2\succ1.
\]

- \(\kappa=1\) 时，6 个位置全部选择；
- \(\kappa=0.5\) 时，预算为 3，优先保留 6、5，以及零分边界中的一个；
- \(\kappa\) 很小时，通常只剩 6 这类高残差位置；
- \(\kappa=0\) 时，一个 OPSD 位置都不保留。

这里的分数每一步都会重算。某个具体位置本步退出后，下步如果模型分布和几何关系改变，理论上可能重新进入 Top-K。因此“先退出”描述的是每个训练步当前排名下的优先级，不是永久冻结的静态名单。

---

## 16. 这套方法为什么这样设计

### 16.1 先全选，再逐渐撤掉

训练早期 student 较弱，teacher 可能在很多位置都提供有价值的密集监督。直接从低覆盖率起步，可能过早丢掉信息。

所以前 10 步先让 OPSD 广泛参与，随后再让几何排序决定谁值得保留更久。

### 16.2 冲突不立刻一刀切

局部 logit 梯度冲突不等于 teacher 一定错误。尤其训练早期，GRPO 信号本身可能噪声较大，或者 teacher 的完整 KL 中仍有其他有用结构。

因此 GIFT 没有采用 hard gate，而是让：

- 强冲突先撤；
- 弱冲突后撤；
- 早期预算足够时仍允许冲突 OPSD 存在。

### 16.3 不冲突时优先保留非共线信息

如果 teacher 梯度几乎完全重复 GRPO 当前方向，它可能有帮助，但从“额外信息”的角度看比较冗余。

若 teacher 与 GRPO 不冲突，同时还有较大垂直分量，说明 teacher 在当前局部 logit 空间提供了 GRPO 方向之外的结构。GIFT 让这种位置更晚退出。

### 16.4 只退火 token 数，让实验归因更清楚

若同时减少 token 数量和 KL 权重，很难判断性能变化来自哪一项。

Token-only 版本固定 \(\lambda=1\)，只改变覆盖位置数量，因此更容易回答：

> 按 GIFT 排名逐步撤走 OPSD 位置，本身是否有效？

---

## 17. 这套方法能主张什么

比较稳妥的主张是：

1. 它用 token-local output-logit 几何组织 teacher 监督的退出顺序；
2. 它显式区分冲突平行分量和非冲突正交残差；
3. 它不让 GRPO advantage 绝对值直接主导 microbatch 蒸馏预算；
4. 它只用投影结果做 token 选择，不修改原始训练梯度；
5. 它通过覆盖率退火，最终平滑回到 objective-level pure GRPO。

---

## 18. 这套方法不能证明什么

### 18.1 Logit 空间同向不等于参数空间同向

参数梯度还要经过模型 Jacobian。不同 token 之间也会共享参数并发生交叉作用。因此单个位置上：

\[
\langle r_i,d_i\rangle>0
\]

不保证全模型参数梯度一定同向。

### 18.2 大正交残差不等于更正确

\(b_i\) 只表示 teacher 梯度偏离 GRPO 参考轴的程度和大小。它是几何属性，不是 teacher 正确性的证明，也不是 reward gain 的无偏估计。

### 18.3 早期所选监督不保证全部无冲突

前 10 步基本全选，之后预算仍可能大于非冲突候选数，所以一些负分 token 仍会被蒸馏。本方法的主张是“冲突按程度优先退出”，不是“永远不使用冲突监督”。

### 18.4 Top-100 + tail 不是完整词表 KL

Tail 只保存外部词表的总质量，无法保留其中每个具体词的差异。这是效率和精细度之间的取舍。

### 18.5 Microbatch 边界会影响结果

谁和谁竞争由实际动态 microbatch 决定。改变动态 batching、token 上限或 teacher workload 可能改变竞争集合，即使评分公式完全不变，也可能改变最终 mask。

---

## 19. 与几种相近方案的区别

| 方法 | 排序依据 | 是否使用 \(|A|\) 排序 | 是否投影实际梯度 | 退火方式 |
|---|---|---:|---:|---|
| 纯 cosine | 两梯度夹角 | 否 | 否 | 可自行配置 |
| Raw inner product | \(\langle r,d\rangle\) | 是 | 否 | 可自行配置 |
| Hard compatibility gate + residual | 冲突位置直接排除；其余按 \(b\) | 否 | 否 | 候选内退火 |
| GIFT Microbatch token-only | 冲突按 \(a<0\)，非冲突按 \(b\) | 否，只保留符号作用 | 否 | \(\kappa:1\to0\)，\(\lambda=1\) |
| GIFT Microbatch staged | 与本文相同 | 否，只保留符号作用 | 否 | 先减 token，后减 KL 权重 |
| 熵 + KL 排序 | student 熵和 teacher/student divergence | 不使用 GRPO 几何 | 否 | token 覆盖率退火 |

本文版本最独特的地方不是“做了投影”，而是：

> **用投影几何形成分段排序，却仍对选中位置反传完整 KL；并让这个排序控制 teacher 监督随 token 预算逐步退出。**

---

## 20. 建议重点观察的诊断量

为了判断方法是否按预期工作，至少应观察：

1. 每一步目标覆盖率 \(\kappa_t\) 与真实选中比例；
2. 每个 microbatch 的 valid token、eligible token 和 selected token 数；
3. 被选 token 中正/负 advantage 的比例；
4. 被选 token 的平均 cosine、平均平行分量和平均正交残差；
5. GRPO 零方向中性回退的占比；
6. OPSD 零梯度被排除的占比；
7. 每条 rollout 实际得到的 OPSD token 数分布；
8. 完全没有得到 OPSD token 的 rollout 比例；
9. OPSD loss、GRPO loss 以及两者参数梯度范数的训练曲线；
10. step 78 后 teacher 计算是否确实停止。

这些量能帮助区分：

- 排名确实筛出了不同 token；
- 还是大多数 token 分数接近，实际近似随机；
- 预算是否过度集中在少数 rollout；
- 零 GRPO 梯度是否多到影响方法语义；
- 退火后辅助监督是否真的随覆盖率减弱。

---

## 21. 常见问题速答

### Q1：Teacher top-100 是否表示最多蒸馏 100 个 token？

不是。它是每个 response 位置上的 100 个词表坐标。蒸馏多少个 response 位置由 microbatch 的 \(K_m(t)\) 决定。

### Q2：Student 和 teacher 是否选择了相同 token？

只是使用相同词表 ID 作为概率比较坐标。双方概率可以不同，student 自己的 top-100 也可以不同。

### Q3：排序是不是只看方向？

不是。GRPO 侧只提供归一化方向；OPSD 侧保留梯度大小。最终分数同时含方向关系与 OPSD 分量强度。

### Q4：排序是否考虑 advantage 大小？

不考虑 \(|A|\)，但考虑 \(\operatorname{sign}(A)\)。完整 GRPO loss 仍考虑 advantage 大小。

### Q5：这是两个原始梯度的内积吗？

不是。原始内积只是中间量。冲突分支使用 \(\langle r,d\rangle/\|r\|\)，非冲突分支使用正交残差范数。

### Q6：选中后只训练 OPSD 的垂直分量吗？

不是。垂直分量只用于排名；选中后训练完整 forward KL。

### Q7：负分是否意味着使用负 KL 权重？

不是。负分只表示排名较低。只要被选中，KL 权重仍为正常正数。

### Q8：为什么前 10 步评分好像没作用？

因为 \(\kappa=1\) 时所有有效候选都被选中。评分从预算开始低于 100% 后才决定成员取舍。

### Q9：最后是不是完全没有 teacher？

是。step 78–96 的覆盖率为 0，不再执行 teacher forward，目标只剩 GRPO。

### Q10：Microbatch 是否等于一道题的 8 条 rollout？

不是。Group 用于计算相对 advantage；microbatch 是实际训练时的动态计算分块，可能同时包含多道题。

---

## 22. 最终总结

GIFT Microbatch token-only 可以理解为一种 **teacher 监督的渐进式撤离策略**：

1. privileged teacher 先通过 top-100 + tail forward KL 给每个 response 位置提供 OPSD 信号；
2. GRPO 梯度只作为归一化参考方向，advantage 的正负号参与冲突判断，但 \(|A|\) 不参与排名；
3. OPSD 梯度不归一化，因此 teacher 分量的实际大小被保留；
4. 冲突 token 按冲突强度得到负分，越冲突越早退出；
5. 非冲突 token 按正交残差得到非负分，补充量越大越晚退出；
6. 所有候选在各自真实 microbatch 内竞争，不设置 group 或 rollout 固定配额；
7. 前 10 步基本全选，step 11–77 逐步减少 response token，step 78 起完全退出 OPSD；
8. 排名只决定二值 mask，被选中位置仍训练完整 forward KL；
9. GRPO 始终作用于全部 valid token，并始终保留完整 advantage 强度；
10. 这是一种 output-logit-space 的局部预算代理，而不是参数空间安全性或最终 reward 提升的数学保证。

如果只记一句话，可以记成：

> **GRPO 告诉 GIFT“参考方向朝哪”，teacher 梯度告诉 GIFT“冲突多强、额外分量多大”，退火预算决定这些 teacher 监督何时逐步退出。**
