# PiT-PO for 1 GPU

**Single-GPU edition of Physics-informed Token-regularized Policy Optimization for LLM-based scientific equation discovery.**

> 本目录是根据用户对“只使用一张 GPU，并消除在线训练后重新部署大模型等待时间”的需求，从论文对应的 PiT-PO 代码整理出的单显卡工程版本。它实现的是同一个 PiT-PO 方法，不是新的算法分支；主要变化是训练与搜索的运行方式。

目录名为 `PiT-PO_for_1_GPU`，下文命令均以该目录为项目根目录。

## 单显卡版本做了什么

传统流程是：

```text
搜索 → 保存训练结果 → 停止推理服务 → 重新部署模型 → 继续搜索
```

本版本参考 CALM（https://github.com/whxru/CALM）的 colocated runtime 思路，将流程改为：

```text
vLLM 生成 → 方程评估 → 单卡 LoRA/GRPO 训练
          → 内存同步最新 LoRA → 同一 vLLM 运行时继续生成
```

具体特性：

- 训练模型、LoRA 和 vLLM 推理引擎位于同一张 GPU。
- 搜索与训练顺序执行，不会同时争用显存。
- 训练后直接把 LoRA tensor 在内存中交给 vLLM。
- 搜索关键路径上不保存/合并完整模型，不启动 HTTP 模型服务，也不重启 vLLM。
- 只在周期 checkpoint、异常退出或实验结束时保存小型 LoRA adapter。
- 单卡本地模式固定使用一个 sampler 和一个 evaluator。

PiT-PO 的物理约束、token 级冗余惩罚、GRPO 奖励以及 multi-island 搜索逻辑保持不变。

## 目录结构

```text
PiT-PO_for_1_GPU/
├── launch_grpo.py            # 单卡 Python 主入口
├── run_experiment.sh         # 后台启动脚本
├── stop_experiment.sh        # 优雅停止并触发恢复 checkpoint
├── grpo_config.py            # 在线训练策略
├── environment.yml
├── pitpo/
│   ├── pipeline.py           # 单进程搜索/评估/训练流水线
│   ├── sampler.py            # colocated vLLM sampler
│   ├── evaluator.py          # 方程评估与训练触发
│   ├── grpo_trainer.py       # Unsloth、LoRA、GRPO 与内存同步
│   ├── buffer.py             # multi-island experience buffer
│   └── ...
├── specs/                    # 问题定义
├── data/                     # train/test 数据
└── tests/
```

## 环境与硬件

```bash
cd /data/home/zdhs0037/PiT-PO_for_1_GPU
conda env create -f environment.yml
conda activate pitpo
```

默认配置面向一张 NVIDIA GPU，并使用：

- `vLLM 0.7.3`
- `Unsloth 2025.3.18`
- 4-bit BitsAndBytes backbone
- BF16 LoRA 计算（GPU 支持时）
- 4096-token context、每个 prompt 生成 4 个候选

该配置已在单张 A100 40GB 上运行。使用 `--no-load_in_4bit` 可以测试非量化 BF16 backbone，但所需显存明显更高，能否同时容纳训练模型和 vLLM 取决于 GPU 容量及 `--vllm_gpu_memory_utilization`。

## 快速开始

### 后台启动脚本

```bash
cd /data/home/zdhs0037/PiT-PO_for_1_GPU
conda activate pitpo
bash run_experiment.sh oscillator1 0
```

脚本按如下格式接收参数：

```text
bash run_experiment.sh [problem] [gpu] [额外的 launch_grpo.py 参数]
```

例如运行 2500 个候选：

```bash
MAX_SAMPLES=2500 bash run_experiment.sh oscillator1 0
```

测试非量化模式：

```bash
bash run_experiment.sh oscillator1 0 --no-load_in_4bit
```

替换模型或生成配置：

```bash
MODEL_PATH=/path/to/model \
MAX_NEW_TOKENS=1024 \
SAMPLES_PER_PROMPT=4 \
bash run_experiment.sh oscillator1 0
```

查看脚本支持的全部环境变量：

```bash
bash run_experiment.sh --help
```

启动后会创建：

```text
runs/1gpu_<problem>_<timestamp>/
├── train.out       # 完整标准输出
├── command.txt     # 可复现的实际启动命令
└── pids.txt        # stop_experiment.sh 使用的 PID
```

### 直接调用 Python

```bash
python -u launch_grpo.py \
    --problem oscillator1 \
    --model_path /data/home/zdhs0037/zdhs0037_data/meta-llama/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct \
    --gpu 0 \
    --max_samples 2500 \
    --samples_per_prompt 4 \
    --max_new_tokens 1024 \
    --grpo_lr 1e-6 \
    --grpo_batch_size 4 \
    --buffer_size 100 \
    --grpo_train_every 32 \
    --training_strategy continuous \
    --load_in_4bit \
    --fast_inference \
    --vllm_gpu_memory_utilization 0.6
```

`--max_new_tokens` 是每个候选的最大 completion 长度，并不表示每次一定生成满该长度。一次 vLLM 调用会并行返回 `--samples_per_prompt` 个候选。

## 常用参数

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--problem` | `data/` 与 `specs/` 下的问题名 | `oscillator1` |
| `--max_samples` | 整个搜索生成的候选总数 | `3000` |
| `--samples_per_prompt` | 一次 vLLM 调用并行生成的候选数 | `4` |
| `--max_new_tokens` | 每个候选的最大生成长度 | `1024` |
| `--max_seq_length` | prompt 与 completion 的总上下文上限 | `4096` |
| `--grpo_lr` | 在线 LoRA 学习率 | `1e-6` |
| `--grpo_batch_size` | GRPO micro-batch size | `4` |
| `--buffer_size` | experience buffer 容量 | `500` |
| `--grpo_train_every` | 每累计 N 个有效候选触发一次训练 | `64` |
| `--training_strategy` | `conservative/adaptive/aggressive/continuous` | `continuous` |
| `--load_in_4bit` | 使用 4-bit backbone | 开启 |
| `--no-load_in_4bit` | 使用非量化 backbone | 关闭 |
| `--fast_inference` | 使用 colocated vLLM | 开启 |
| `--vllm_gpu_memory_utilization` | Unsloth/vLLM 可使用的显存比例 | `0.6` |
| `--disable_grpo` | 只搜索，不在线训练 LoRA | 关闭 |
| `--checkpoint_every` | 每 N 次模型更新保存恢复 adapter | `100` |
| `--export_merged_model` | 搜索结束后额外导出 merged model | 关闭 |

`run_experiment.sh` 为单卡实验选择了更适合当前 oscillator 配置的 `buffer_size=100` 和 `grpo_train_every=32`；直接执行 `launch_grpo.py` 时使用表中的通用默认值。

## 日志、checkpoint 与停止方式

内部训练日志和 TensorBoard 数据分别写入：

```text
logs/grpo_<problem>_<timestamp>/
reward_curves/grpo_<problem>_<timestamp>/
```

关键输出包括：

- `training.log`：在线训练触发、loss、buffer 状态。
- `time_summary.json`：生成、评估以及训练到下一次生成的耗时。
- `checkpoints/`：周期 LoRA 恢复点。
- `final_adapter/`：正常完成后的最终 LoRA。
- `emergency_adapter/`：异常或终止时的恢复 LoRA。

停止最近一次由脚本启动的实验：

```bash
bash stop_experiment.sh
```

停止指定实验：

```bash
bash stop_experiment.sh runs/1gpu_oscillator1_YYYYMMDD_HHMMSS
```

停止脚本发送 `SIGTERM`，程序会尝试写入 emergency adapter。不要直接使用 `kill -9`。

> 不要在实验运行过程中移动或重命名项目目录。日志、checkpoint 和 Slurm 输出可能保存了绝对路径；应先停止实验，重命名后再从新目录重新提交。

## 添加新问题

1. 添加 `data/<task>/train.csv`，最后一列为目标值。
2. 添加 `specs/specification_<task>_numpy.txt`。
3. 在 spec 中用 `@equation.evolve` 标记待搜索函数；可以用 `@evaluate.run` 提供自定义 evaluator。
4. 运行 `bash run_experiment.sh <task> 0`。

## 测试

不加载 GPU 模型的单元测试：

```bash
python -m pytest -q
```

GPU 上的训练后立即生成 smoke test 位于 `tests/gpu_smoke_train_then_generate.py`。

## Citation

```bibtex
@misc{wang2026llmbasedscientificequationdiscovery,
      title={LLM-Based Scientific Equation Discovery via Physics-Informed Token-Regularized Policy Optimization},
      author={Boxiao Wang and Kai Li and Tianyi Liu and Chen Li and Junzhe Wang and Yifan Zhang and Jian Cheng},
      year={2026},
      eprint={2602.10576},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.10576}
}
```

## Acknowledgements

本项目基于 LLM-SR、LLM-SRBench，以及 CALM 的训练/搜索 colocated runtime 思路。感谢相关工作公开代码与数据。
