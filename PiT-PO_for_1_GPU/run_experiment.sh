#!/usr/bin/env bash
# PiT-PO-for-1-GPU launcher.
#
# Usage:
#   bash run_experiment.sh [problem] [gpu] [additional launch_grpo.py options]
#
# Examples:
#   bash run_experiment.sh oscillator1 0
#   MAX_SAMPLES=2500 bash run_experiment.sh oscillator1 0
#   bash run_experiment.sh oscillator1 0 --no-load_in_4bit

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_PATH="/data/home/zdhs0037/zdhs0037_data/meta-llama/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct"

show_help() {
    cat <<'EOF'
PiT-PO-for-1-GPU

Usage:
  bash run_experiment.sh [problem] [gpu] [additional launch_grpo.py options]

Positional arguments:
  problem    Task name under data/ and specs/ (default: oscillator1)
  gpu        Physical GPU index (default: 0)

Environment overrides:
  MODEL_PATH                 Local/Hugging Face model path
  MAX_SAMPLES                Total generated candidates (default: 3000)
  SAMPLES_PER_PROMPT         Candidates generated per vLLM call (default: 4)
  MAX_NEW_TOKENS             Completion-token limit (default: 1024)
  GRPO_LR                    Online LoRA learning rate (default: 1e-6)
  GRPO_BATCH_SIZE            GRPO micro-batch size (default: 4)
  BUFFER_SIZE                Experience-buffer capacity (default: 100)
  GRPO_TRAIN_EVERY           Train every N valid candidates (default: 32)
  LOAD_IN_4BIT               1 for 4-bit, 0 for BF16 (default: 1)
  FAST_INFERENCE             1 for colocated vLLM, 0 for HF generate (default: 1)
  VLLM_GPU_MEMORY_UTILIZATION vLLM memory fraction (default: 0.6)
  PYTHON_BIN                 Python executable (default: python)
  HF_HUB_OFFLINE             1 for local/offline loading (default: 1)

Additional options are appended to launch_grpo.py. Because they are appended,
they can override defaults above, for example: --buffer_size 500.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

PROBLEM="${1:-oscillator1}"
if (( $# > 0 )); then shift; fi
GPU="${1:-0}"
if (( $# > 0 )); then shift; fi
EXTRA_ARGS=("$@")

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GRPO_LR="${GRPO_LR:-1e-6}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-4}"
BUFFER_SIZE="${BUFFER_SIZE:-100}"
GRPO_TRAIN_EVERY="${GRPO_TRAIN_EVERY:-32}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
FAST_INFERENCE="${FAST_INFERENCE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"

case "${LOAD_IN_4BIT}" in
    1|true|TRUE|yes|YES) QUANTIZATION_FLAG="--load_in_4bit" ;;
    0|false|FALSE|no|NO) QUANTIZATION_FLAG="--no-load_in_4bit" ;;
    *) echo "[ERROR] LOAD_IN_4BIT must be 1/0 or true/false" >&2; exit 2 ;;
esac

case "${FAST_INFERENCE}" in
    1|true|TRUE|yes|YES) INFERENCE_FLAG="--fast_inference" ;;
    0|false|FALSE|no|NO) INFERENCE_FLAG="--no-fast_inference" ;;
    *) echo "[ERROR] FAST_INFERENCE must be 1/0 or true/false" >&2; exit 2 ;;
esac

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${WORK_DIR}/runs/1gpu_${PROBLEM}_${TIMESTAMP}"
TRAIN_OUT="${RUN_DIR}/train.out"
PID_FILE="${RUN_DIR}/pids.txt"
COMMAND_FILE="${RUN_DIR}/command.txt"
mkdir -p "${RUN_DIR}"

COMMAND=(
    "${PYTHON_BIN}" -u "${WORK_DIR}/launch_grpo.py"
    --problem "${PROBLEM}"
    --gpu "${GPU}"
    --model_path "${MODEL_PATH}"
    --max_samples "${MAX_SAMPLES}"
    --samples_per_prompt "${SAMPLES_PER_PROMPT}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --grpo_lr "${GRPO_LR}"
    --grpo_batch_size "${GRPO_BATCH_SIZE}"
    --buffer_size "${BUFFER_SIZE}"
    --reward_scaling 10.0
    --grpo_train_every "${GRPO_TRAIN_EVERY}"
    --training_strategy continuous
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    "${QUANTIZATION_FLAG}"
    "${INFERENCE_FLAG}"
    --log_level INFO
)
COMMAND+=("${EXTRA_ARGS[@]}")

{
    printf 'cd %q\n' "${WORK_DIR}"
    printf 'HF_HUB_OFFLINE=%q TRANSFORMERS_OFFLINE=%q ' \
        "${HF_HUB_OFFLINE}" "${TRANSFORMERS_OFFLINE}"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "${COMMAND_FILE}"

echo "============================================================"
echo "  PiT-PO-for-1-GPU"
echo "  Problem:             ${PROBLEM}"
echo "  Model:               ${MODEL_PATH}"
echo "  Physical GPU:        ${GPU}"
echo "  Quantization:        $([[ ${QUANTIZATION_FLAG} == --load_in_4bit ]] && echo '4-bit' || echo 'BF16')"
echo "  Candidates / prompt: ${SAMPLES_PER_PROMPT}"
echo "  Max new tokens:      ${MAX_NEW_TOKENS}"
echo "  Output directory:    ${RUN_DIR}"
echo "============================================================"

cd "${WORK_DIR}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
nohup "${COMMAND[@]}" > "${TRAIN_OUT}" 2>&1 &

TRAIN_PID=$!
printf 'PiT-PO-for-1-GPU %s\n' "${TRAIN_PID}" > "${PID_FILE}"
printf '%s\n' "${RUN_DIR}" > "${WORK_DIR}/last_run.txt"

echo "Started PID ${TRAIN_PID}"
echo "Log:     ${TRAIN_OUT}"
echo "Command: ${COMMAND_FILE}"
echo "Watch:   tail -f ${TRAIN_OUT}"
echo "Stop:    bash ${WORK_DIR}/stop_experiment.sh ${RUN_DIR}"
