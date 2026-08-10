#!/usr/bin/env bash
# Gracefully stop a PiT-PO-for-1-GPU process launched by run_experiment.sh.

set -u

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-$(cat "${WORK_DIR}/last_run.txt" 2>/dev/null || true)}"

if [[ -n "${RUN_DIR}" && "${RUN_DIR}" != /* ]]; then
    RUN_DIR="${WORK_DIR}/${RUN_DIR}"
fi

if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/pids.txt" ]]; then
    echo "[ERROR] pids.txt not found. Pass a run directory, for example:" >&2
    echo "  bash stop_experiment.sh runs/1gpu_oscillator1_YYYYMMDD_HHMMSS" >&2
    exit 1
fi

echo "Stopping PiT-PO-for-1-GPU run: ${RUN_DIR}"
while IFS= read -r line; do
    PID="${line##* }"
    NAME="${line% *}"
    if [[ "${PID}" =~ ^[0-9]+$ ]] && kill -0 "${PID}" 2>/dev/null; then
        kill -TERM "${PID}"
        echo "  Sent SIGTERM to ${NAME}(${PID}); an emergency LoRA checkpoint will be attempted."
    else
        echo "  ${NAME}(${PID}) is not running."
    fi
done < "${RUN_DIR}/pids.txt"
