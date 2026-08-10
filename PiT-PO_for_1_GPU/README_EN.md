# PiT-PO for 1 GPU

**Single-GPU edition of Physics-informed Token-regularized Policy Optimization for LLM-based scientific equation discovery.**

> This directory provides a single-GPU engineering edition of the code accompanying PiT-PO. It was prepared in response to requests from users who need to run the method on only one GPU while eliminating the delay caused by redeploying the large language model after online training. It implements the same PiT-PO method rather than a new algorithmic branch; the main change is how training and search are executed.

The directory is named `PiT-PO_for_1_GPU`. All commands below assume that this directory is the project root.

## What the Single-GPU Edition Changes

The conventional workflow is:

```text
Search → Save training results → Stop the inference service → Redeploy the model → Resume search
```

Inspired by the colocated runtime design in CALM (https://github.com/whxru/CALM), this edition changes the workflow to:

```text
vLLM generation → Equation evaluation → Single-GPU LoRA/GRPO training
                → Synchronize the latest LoRA in memory → Continue generation with the same vLLM runtime
```

Key features:

- The training model, LoRA adapter, and vLLM inference engine reside on the same GPU.
- Search and training run sequentially, so they do not compete for GPU memory at the same time.
- After training, the LoRA tensors are handed directly to vLLM in memory.
- The search-critical path does not save or merge the full model, start an HTTP model service, or restart vLLM.
- Small LoRA adapters are saved only for periodic checkpoints, abnormal termination, or experiment completion.
- Local single-GPU mode uses exactly one sampler and one evaluator.

The physical constraints, token-level redundancy penalty, GRPO reward, and multi-island search logic of PiT-PO remain unchanged.

## Repository Layout

```text
PiT-PO_for_1_GPU/
├── launch_grpo.py            # Main single-GPU Python entry point
├── run_experiment.sh         # Background launcher
├── stop_experiment.sh        # Graceful stop and recovery-checkpoint trigger
├── grpo_config.py            # Online training strategies
├── environment.yml
├── pitpo/
│   ├── pipeline.py           # Single-process search/evaluation/training pipeline
│   ├── sampler.py            # Colocated vLLM sampler
│   ├── evaluator.py          # Equation evaluation and training triggers
│   ├── grpo_trainer.py       # Unsloth, LoRA, GRPO, and in-memory synchronization
│   ├── buffer.py             # Multi-island experience buffer
│   └── ...
├── specs/                    # Problem definitions
├── data/                     # Training and test data
└── tests/
```

## Environment and Hardware

```bash
cd /data/home/zdhs0037/PiT-PO_for_1_GPU
conda env create -f environment.yml
conda activate pitpo
```

The default configuration targets one NVIDIA GPU and uses:

- `vLLM 0.7.3`
- `Unsloth 2025.3.18`
- A 4-bit BitsAndBytes backbone
- BF16 LoRA computation when supported by the GPU
- A 4096-token context and four candidates per prompt

This configuration has been run on a single 40 GB A100. Use `--no-load_in_4bit` to test an unquantized BF16 backbone. This mode requires substantially more GPU memory, and whether the training model and vLLM can coexist depends on GPU capacity and `--vllm_gpu_memory_utilization`.

## Quick Start

### Background Launcher

```bash
cd /data/home/zdhs0037/PiT-PO_for_1_GPU
conda activate pitpo
bash run_experiment.sh oscillator1 0
```

The launcher accepts arguments in the following format:

```text
bash run_experiment.sh [problem] [gpu] [additional launch_grpo.py arguments]
```

For example, to run a search with 2,500 candidates:

```bash
MAX_SAMPLES=2500 bash run_experiment.sh oscillator1 0
```

To test unquantized mode:

```bash
bash run_experiment.sh oscillator1 0 --no-load_in_4bit
```

To replace the model or generation settings:

```bash
MODEL_PATH=/path/to/model \
MAX_NEW_TOKENS=1024 \
SAMPLES_PER_PROMPT=4 \
bash run_experiment.sh oscillator1 0
```

To list all environment variables supported by the launcher:

```bash
bash run_experiment.sh --help
```

Each launch creates:

```text
runs/1gpu_<problem>_<timestamp>/
├── train.out       # Complete standard output
├── command.txt     # Reproducible launch command
└── pids.txt        # PID used by stop_experiment.sh
```

### Direct Python Invocation

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

`--max_new_tokens` is the maximum completion length for each candidate; it does not mean that every candidate will necessarily use the full limit. Each vLLM call returns `--samples_per_prompt` candidates in parallel.

## Common Options

| Option | Description | Default |
|---|---|---:|
| `--problem` | Problem name under `data/` and `specs/` | `oscillator1` |
| `--max_samples` | Total number of candidates generated during the search | `3000` |
| `--samples_per_prompt` | Number of candidates generated in parallel by one vLLM call | `4` |
| `--max_new_tokens` | Maximum generation length for each candidate | `1024` |
| `--max_seq_length` | Combined context limit for the prompt and completion | `4096` |
| `--grpo_lr` | Online LoRA learning rate | `1e-6` |
| `--grpo_batch_size` | GRPO micro-batch size | `4` |
| `--buffer_size` | Experience-buffer capacity | `500` |
| `--grpo_train_every` | Trigger training after every N valid candidates | `64` |
| `--training_strategy` | `conservative/adaptive/aggressive/continuous` | `continuous` |
| `--load_in_4bit` | Use a 4-bit backbone | enabled |
| `--no-load_in_4bit` | Use an unquantized backbone | disabled |
| `--fast_inference` | Use colocated vLLM | enabled |
| `--vllm_gpu_memory_utilization` | Fraction of GPU memory available to Unsloth/vLLM | `0.6` |
| `--disable_grpo` | Run search without online LoRA training | disabled |
| `--checkpoint_every` | Save a recovery adapter every N model updates | `100` |
| `--export_merged_model` | Export an additional merged model after search completion | disabled |

For the current oscillator configuration, `run_experiment.sh` selects the more suitable single-GPU values `buffer_size=100` and `grpo_train_every=32`. Direct invocation of `launch_grpo.py` uses the general defaults shown in the table.

## Logs, Checkpoints, and Stopping a Run

Internal training logs and TensorBoard data are written to:

```text
logs/grpo_<problem>_<timestamp>/
reward_curves/grpo_<problem>_<timestamp>/
```

Key outputs include:

- `training.log`: online training triggers, losses, and buffer state.
- `time_summary.json`: generation, evaluation, and training-to-next-generation latency.
- `checkpoints/`: periodic LoRA recovery checkpoints.
- `final_adapter/`: final LoRA adapter after normal completion.
- `emergency_adapter/`: recovery LoRA adapter after interruption or abnormal termination.

To stop the most recent experiment launched by the script:

```bash
bash stop_experiment.sh
```

To stop a specific experiment:

```bash
bash stop_experiment.sh runs/1gpu_oscillator1_YYYYMMDD_HHMMSS
```

The stop script sends `SIGTERM`, allowing the program to attempt to write an emergency adapter. Do not use `kill -9`.

> Do not move or rename the project directory while an experiment is running. Logs, checkpoints, and Slurm output may contain absolute paths. Stop the experiment first, rename the directory, and then resubmit it from the new location.

## Adding a New Problem

1. Add `data/<task>/train.csv`, with the target values in the final column.
2. Add `specs/specification_<task>_numpy.txt`.
3. Mark the function to be searched with `@equation.evolve` in the specification. You may provide a custom evaluator with `@evaluate.run`.
4. Run `bash run_experiment.sh <task> 0`.

## Testing

Run the unit tests without loading a GPU model:

```bash
python -m pytest -q
```

The post-training immediate-generation GPU smoke test is located at `tests/gpu_smoke_train_then_generate.py`.

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

This project builds on LLM-SR, LLM-SRBench, and CALM's colocated runtime design for training and search. We thank the authors of these projects for making their code and data publicly available.
