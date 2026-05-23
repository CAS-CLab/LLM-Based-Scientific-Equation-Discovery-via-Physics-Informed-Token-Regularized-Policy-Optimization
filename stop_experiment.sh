#!/bin/bash
# Stop an experiment: read pids.txt and terminate all associated processes (file must use Unix LF endings)
# Usage: bash stop_experiment.sh [runs/xxx]   or run with no args to stop the most recent run

RUN_DIR=${1:-"$(cat "$(dirname "$0")/last_run.txt" 2>/dev/null)"}

if [ -z "${RUN_DIR}" ] || [ ! -f "${RUN_DIR}/pids.txt" ]; then
    echo "[ERROR] pids.txt not found; please pass a valid runs/xxx directory"
    exit 1
fi

echo "Stopping experiment: ${RUN_DIR}"
while IFS= read -r line; do
    PID=$(echo "${line}" | grep -oP '\d+$')
    NAME=$(echo "${line}" | grep -oP '^\S+')
    if [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; then
        kill "${PID}"
        echo "  Terminated ${NAME} (PID ${PID})"
    else
        echo "  ${NAME} (PID ${PID}) is no longer running"
    fi
done < "${RUN_DIR}/pids.txt"
echo "Done."
