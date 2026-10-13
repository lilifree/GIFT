# GIFT Microbatch Token-only：4B/1.7B 数学规模实验

这两组实验是现有 Qwen3-8B `GIFT Microbatch token-only` 的严格模型规模复现。
除模型身份和独立输出目录外，所有 YAML 参数保持与 8B 完全一致。

## 固定设置

- 数据：DeepMath level 7–10；验证集为 AMC23、AIME24、AIME25。
- 总训练步数：96；`val_before_train: false`，因此不做 step 0 评测。
- 保存与评测：每 10 个 actor step 一次。
- 排序范围：同一 actor microbatch 内的所有 valid response token。
- 排序分数：GIFT 分段分数；冲突分支为 `-|parallel|`，非冲突分支为正交残差范数。
- 退火：step 1–10 全选；step 11–77 将 token 覆盖率线性降至接近 0；step 78–96 完全退出 OPSD。
- OPSD 权重始终为 1，只减少 token 数，不做权重退火。
- 拓扑：论文实验为 2 节点 × 16 GPU，由配置中的 `nnodes` / `n_gpus_per_node` 决定，4B/1.7B 须与 8B 配置一致。全局 batch 不随 GPU 数变化，训练预算和批量语义可比。

## 两个实验

| 模型 | 配置 | 独立输出目录 |
|---|---|---|
| Qwen3-4B | `qwen3_4b_gift_microbatch_token_anneal.yaml` | `GIFT-MICROBATCH-TOKEN-ANNEAL-FULLSTART-Qwen3-4B-level7-10` |
| Qwen3-1.7B | `qwen3_1_7b_gift_microbatch_token_anneal.yaml` | `GIFT-MICROBATCH-TOKEN-ANNEAL-FULLSTART-Qwen3-1.7B-level7-10` |

## 隔离与复用

本次只新增文件，不修改原有 8B 配置、运行时或启动脚本。新入口先执行模型感知的
fail-closed 校验，再复用已经训练验证过的 GIFT runtime。验证器还会逐字段比较小模型
配置与 8B 配置，除 `model_path`、`output_dir`、`project_name`、`experiment_name` 外，
任何配置漂移都会导致启动失败。
