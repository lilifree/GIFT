<div align="center">

# GIFT: Gradient-Informed FilTering for Distillation-Augmented Reasoning RL

[![Dataset](https://img.shields.io/badge/Dataset-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Leyiii/RLCSD)

<p>
  <a href="#-overview">Overview</a> •
  <a href="#-results">Results</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 📖 Overview

Reasoning LLMs are trained with two complementary signals: **sparse, outcome-grounded RL** (GRPO) and **dense, token-level self-distillation** (OPSD). The two signals can conflict, because the teacher's imitation gradient can work against the verifier's reward. **GIFT** decides token by token which teacher updates to apply. It splits each token's distillation gradient into a component **parallel** to the RL gradient, which measures conflict, and a component **orthogonal** to it, which measures novel guidance. A **coverage-annealing** schedule then shrinks the set of distilled tokens until training is pure GRPO.

<p align="center">
  <img src="assets/gift_overview.png" width="95%" alt="GIFT overview: the distillation gradient d_i is decomposed against the GRPO gradient r_i into a conflict component a_i and a complement component b_i, tokens are ranked by a priority score, and the selection budget is annealed from all tokens to none"/>
</p>

<table>
<tr>
<td width="25%" align="center"><b>🛡️ Safety filter</b><br/><sub>The sign of the parallel component <i>a<sub>i</sub></i> detects first-order conflict with the RL update</sub></td>
<td width="25%" align="center"><b>🧭 Value measure</b><br/><sub>The orthogonal norm <i>b<sub>i</sub></i> measures guidance that rescaling the RL gradient cannot reproduce</sub></td>
<td width="25%" align="center"><b>⚡ Closed-form scoring</b><br/><sub>Computed per token from forward-pass statistics, with no extra backward pass</sub></td>
<td width="25%" align="center"><b>📉 Coverage annealing</b><br/><sub>The token budget shrinks while λ stays at 1. Conflicting tokens are dropped first</sub></td>
</tr>
</table>

## 📊 Results

**Mathematical reasoning** across Qwen3 scales (mean@12, %). Training data is DeepMath-103K. Avg. is the mean over AMC23, AIME24 and AIME25.

| Size | Bench. | BASE | GRPO | OPSD | SDPO | RLSD | RLCSD | OPDVR | LUFFY | UFT | **GIFT** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1.7B | AMC23 | 75.50 | 75.70 | 75.60 | 75.30 | 75.30 | <u>75.80</u> | **76.51** | 75.40 | 74.60 | <u>75.80</u> <sub>(+0.40%)</sub> |
| | AIME24 | 44.44 | 45.28 | 45.56 | 44.72 | 44.72 | 46.94 | 46.67 | <u>48.33</u> | <u>48.33</u> | **48.61** <sub>(+9.38%)</sub> |
| | AIME25 | 35.00 | 35.56 | 33.33 | <u>36.67</u> | 34.72 | 34.72 | 35.83 | 34.17 | <u>36.67</u> | **38.33** <sub>(+9.51%)</sub> |
| | Avg. | 51.65 | 52.18 | 51.50 | 52.23 | 51.58 | 52.49 | 53.00 | 52.63 | <u>53.20</u> | **54.25** <sub>(+5.03%)</sub> |
| 4B | AMC23 | 88.45 | 89.16 | 88.25 | 88.96 | 88.86 | **90.26** | 88.45 | 89.16 | <u>89.66</u> | 88.55 <sub>(+0.11%)</sub> |
| | AIME24 | 71.67 | <u>74.17</u> | 73.06 | 70.83 | 73.61 | 72.22 | 73.61 | **76.11** | 70.56 | 73.33 <sub>(+2.32%)</sub> |
| | AIME25 | 63.89 | 63.61 | <u>67.22</u> | <u>67.22</u> | 63.61 | 64.17 | 64.72 | 65.28 | 64.72 | **69.72** <sub>(+9.13%)</sub> |
| | Avg. | 74.67 | 75.65 | 76.18 | 75.67 | 75.36 | 75.55 | 75.59 | <u>76.85</u> | 74.98 | **77.20** <sub>(+3.39%)</sub> |
| 8B | AMC23 | 89.16 | 89.36 | 88.76 | 89.56 | 88.76 | 89.26 | **89.96** | **89.96** | 88.86 | <u>89.66</u> <sub>(+0.56%)</sub> |
| | AIME24 | 75.00 | 74.72 | **77.22** | 75.83 | 75.28 | <u>76.11</u> | 74.44 | 73.33 | <u>76.11</u> | <u>76.11</u> <sub>(+1.48%)</sub> |
| | AIME25 | 65.83 | 65.83 | 66.39 | 66.39 | <u>69.72</u> | 68.06 | <u>69.72</u> | 67.50 | 66.39 | **71.11** <sub>(+8.02%)</sub> |
| | Avg. | 76.67 | 76.64 | 77.46 | 77.26 | 77.92 | 77.81 | <u>78.04</u> | 76.93 | 77.12 | **78.96** <sub>(+2.99%)</sub> |

**Logical reasoning** on Knights-and-Knaves (pass@1, %). The in-distribution (ID) split has 4–8 roles. The out-of-distribution (OOD) splits have 9, 10 and 11 roles.

| Benchmark | BASE | GRPO | OPSD | SDPO | RLSD | RLCSD | OPDVR | LUFFY | UFT | **GIFT** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ID · 4–8 roles | 72.40 | 73.40 | 72.80 | **75.20** | 73.60 | 74.00 | 73.00 | <u>74.60</u> | 74.20 | 73.40 <sub>(+1.38%)</sub> |
| OOD · 9 roles | 69.00 | 72.00 | **75.00** | <u>73.00</u> | 71.00 | 69.00 | 71.00 | 67.00 | 71.00 | <u>73.00</u> <sub>(+5.80%)</sub> |
| OOD · 10 roles | <u>59.00</u> | 56.00 | <u>59.00</u> | 56.00 | 55.00 | <u>59.00</u> | 57.00 | <u>59.00</u> | 56.00 | **61.00** <sub>(+3.39%)</sub> |
| OOD · 11 roles | 42.00 | **46.00** | 41.00 | 40.00 | 41.00 | <u>44.00</u> | <u>44.00</u> | 43.00 | 40.00 | <u>44.00</u> <sub>(+4.76%)</sub> |
| Avg. | 60.60 | 61.85 | <u>61.95</u> | 61.05 | 60.15 | 61.50 | 61.25 | 60.90 | 60.30 | **62.85** <sub>(+3.71%)</sub> |

In both tables, **bold** marks the best score, <u>underline</u> the second best, and parentheses give GIFT's relative gain over BASE.

## 🚀 Getting Started

### Requirements

- Linux with NVIDIA GPUs and CUDA 12.x. The launcher defaults `CUDA_HOME` to `/usr/local/cuda-12.6`.
- Python ≥ 3.10, as required by the vendored verl.
- **32 GPUs** for the paper setting. The launcher reads the GPU layout from `nnodes` and `n_gpus_per_node` in the config. `ppo_mini_batch_size` must be divisible by the total GPU count.
- Main dependencies: PyTorch `>=2.5,<2.10`, vLLM, FlashAttention, Ray and Hydra. A modified copy of [verl](https://github.com/volcengine/verl) `0.8.0.dev` is vendored in [`third_party/verl`](third_party/verl).

### Installation

```bash
pip install -r requirements.txt
pip install -r third_party/verl/requirements.txt
```

`requirements.txt` keeps torch below 2.10, because torch 2.10+ ships CUDA 13 wheels. To match CUDA 12.6 exactly, install torch from the PyTorch index:

```bash
pip install "torch>=2.5.0,<2.10" --index-url https://download.pytorch.org/whl/cu126
```

You do not need to install `third_party/verl`. The launcher adds it and the GIFT runtime to `PYTHONPATH`.

### Data and models

The training and evaluation parquets are hosted at [Leyiii/RLCSD](https://huggingface.co/datasets/Leyiii/RLCSD). Download all of them with:

```bash
python scripts/download_data.py --all

# or a single split
python scripts/download_data.py --dataset kk_4to8 --split train
```

The files are written to `data/gift/<dataset>/{train,val}.parquet`.

| Dataset | Used by |
| :--- | :--- |
| `deepmath_filtered_level5_7` | Qwen3-1.7B (math train) |
| `deepmath_filtered_level6_8` | Qwen3-4B (math train) |
| `deepmath_filtered_level7_10` | Qwen3-8B + OLMo-3-7B-Think (math train) |
| `amc23+aime24+aime25` | math eval |
| `kk_4to8` | logic train (Knights-and-Knaves, 4–8 roles) |
| `kk_4to8_test+kk_9+kk_10+kk_11` | logic eval (ID 4–8 + OOD 9/10/11 roles) |

The training subsets are difficulty bands filtered from [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) ([He et al., 2025](https://arxiv.org/abs/2504.11456)). The Knights-and-Knaves generator follows [Logic-RL](https://arxiv.org/abs/2502.14768) (Xie et al., 2025).

Place each Hugging Face checkpoint in a directory that contains `config.json` and `tokenizer_config.json`, and set `model_path` in your config to that directory.

### Train

Run the same command on every node, changing only `NODE_RANK`. Node 0 hosts the Ray head and the trainer.

```bash
NODE_RANK=0 MASTER_ADDR=<head-node-ip> \
bash scripts/gift/main/run_gift.sh <your_config>.yaml
```

Hydra overrides placed after the config path are passed through to verl. Checkpoints are written to `output_dir`. Logs go to the console, TensorBoard and files, and also to SwanLab when `use_swanlab: true`.

## 🧪 Testing

The unit tests run on CPU and need no models or data:

```bash
PYTHONPATH=scripts/gift/main:third_party/verl:. python scripts/gift/main/test_gift.py
```

They check that the closed-form $a_i$ and $b_i$ match logit-space autograd, that the strongest conflicts are dropped first, that tokens with a zero GRPO gradient score as neutral, and that the coverage schedule and the fixed $\lambda$ follow the paper.

## 🗂️ Repository Layout

```text
.
├── assets/                       # README figures
├── scripts/
│   ├── download_data.py          # fetch parquet datasets into data/gift/
│   ├── _gpu_run_verl.sh          # YAML → verl/Hydra launcher
│   └── gift/main/
│       ├── gift_runtime/         # GIFT: piecewise geometry scoring + Top-K masks
│       ├── core_runtime/         # shared overlay: sparse-tail geometry, schedule, actor hooks
│       ├── gift_bootstrap/       # sitecustomize that installs the overlay in Ray workers
│       ├── run_gift.sh           # multi-node launcher
│       ├── verify_*.py           # fail-closed config and Ray-worker validators
│       └── test_gift.py          # unit tests
├── src/                          # self-distillation trainer, reward function, prompts, baseline losses
└── third_party/verl/             # vendored verl 0.8.0.dev with GIFT hooks and baselines
```

GIFT leaves verl's source files unchanged. At import time it patches the policy loss and the actor in memory. [`third_party/verl/MODIFICATIONS.md`](third_party/verl/MODIFICATIONS.md) lists the hooks it uses and the baseline losses from the paper (`opsd`, `sdpo`, `rlsd`, `rlcsd`, `tip_*`).

## 📝 Citation

```bibtex
@misc{gift2026,
  title  = {{GIFT}: Gradient Informed Filtering for Distillation Augmented Reasoning {RL}},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review as a conference paper at ICLR 2027}
}
```

## 🙏 Acknowledgements

- [verl](https://github.com/volcengine/verl) is the training framework. The vendored copy keeps its Apache-2.0 license and notices ([`LICENSE`](third_party/verl/LICENSE), [`Notice.txt`](third_party/verl/Notice.txt)).
- [Leyiii/RLCSD](https://huggingface.co/datasets/Leyiii/RLCSD) provides the processed training and evaluation datasets.
- [DeepMath-103K](https://arxiv.org/abs/2504.11456), [Logic-RL](https://arxiv.org/abs/2502.14768) and the [AMC23](https://huggingface.co/datasets/math-ai/amc23), [AIME24](https://huggingface.co/datasets/math-ai/aime24) and [AIME25](https://huggingface.co/datasets/math-ai/aime25) benchmarks provide the source data.
- [Qwen3](https://arxiv.org/abs/2505.09388) and [OLMo 3](https://arxiv.org/abs/2512.13961) are the base models.
