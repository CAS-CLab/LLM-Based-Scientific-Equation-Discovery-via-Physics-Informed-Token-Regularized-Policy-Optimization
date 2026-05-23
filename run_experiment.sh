#!/bin/bash
# ============================================================
# PiT-PO one-click experiment launcher (EvoTune-style: vLLM managed by Python)
# Usage:   bash run_experiment.sh [problem] [vLLM_GPU] [GRPO_GPU] [port]
# Example: bash run_experiment.sh oscillator1 0 1 6000
# ============================================================

PROBLEM=${1:-"oscillator1"}
VLLM_GPU=${2:-"0"}
GRPO_GPU=${3:-"1"}
PORT=${4:-"6000"}

MODEL_PATH="/mnt/finder/wangboxiao/LLM-Research/Meta-Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3.1-8B-Instruct"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${WORK_DIR}/runs/${PROBLEM}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

TRAIN_OUT="${RUN_DIR}/train.out"
PID_FILE="${RUN_DIR}/pids.txt"

echo "=============================================="
echo "  PiT-PO (EvoTune-style vLLM)"
echo "  Problem:    ${PROBLEM}"
echo "  Model:      ${MODEL_PATH}"
echo "  vLLM GPU:   ${VLLM_GPU}  Port: ${PORT}"
echo "  GRPO GPU:   ${GRPO_GPU}"
echo "  Output dir: ${RUN_DIR}"
echo "=============================================="

echo "[1/1] Starting PiT-PO (vLLM will be launched by Python)..."
cd "${WORK_DIR}" || exit 1
CUDA_VISIBLE_DEVICES=${GRPO_GPU} nohup python launch_grpo.py --problem "${PROBLEM}" --max_samples 3000 --port "${PORT}" --grpo_lr 1e-6 --grpo_batch_size 4 --buffer_size 100 --reward_scaling 10.0 --grpo_train_every 32 --device_id 0 --vllm_gpu "${VLLM_GPU}" --training_strategy continuous --log_level INFO > "${TRAIN_OUT}" 2>&1 &

TRAIN_PID=$!
echo "Train PID: ${TRAIN_PID}" | tee -a "${PID_FILE}"
echo "  Log -> ${TRAIN_OUT}"

echo ""
echo "=============================================="
echo "  Experiment launched!"
echo ""
echo "  Watch log:  tail -f ${TRAIN_OUT}"
echo "  Stop:       kill ${TRAIN_PID}"
echo "=============================================="
echo "${RUN_DIR}" > "${WORK_DIR}/last_run.txt"
