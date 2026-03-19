#!/usr/bin/env bash
# NB 07: DB error experiment, add method (25 sampled runs)
# Usage: ./scripts/05_db_error_add.sh [--model qwen3:32b]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source "$SCRIPT_DIR/_parse_args.sh" "$@"

echo "=== DB Error + Add ==="

if [[ -z "${DB_SETUP_DONE:-}" ]]; then
    echo "Setting up clean template databases..."
    python run_experiment.py --setup-all-dbs
fi

source "$SCRIPT_DIR/_wait_for_server.sh"
source "$SCRIPT_DIR/_ollama_model.sh"
source "$SCRIPT_DIR/_kill_port.sh"

echo "Starting servers..."
kill_port 8000
kill_port 8001
python -m servers.process_server --experiment-type exception_handling_db_error --port 8000 &
PID_PROCESS=$!
trap "kill $PID_PROCESS 2>/dev/null || true; wait $PID_PROCESS 2>/dev/null || true" EXIT
wait_for_server http://localhost:8000/docs

python -m servers.frame_server --experiment-type exception_handling_db_error --port 8001 $MODEL_ARGS &
PID_FRAME=$!
trap "kill $PID_PROCESS $PID_FRAME 2>/dev/null || true; wait $PID_PROCESS $PID_FRAME 2>/dev/null || true" EXIT

wait_for_server http://localhost:8001/docs

warmup_ollama_model

echo "Running 25 sampled add-method runs..."
python run_experiment.py --experiment-type exception_handling_db_error --method add --sample 25 $ALL_ARGS

unload_ollama_model

echo "Evaluating..."
python run_experiment.py --evaluate-only --experiment-type exception_handling_db_error --method add --sample 25 $ALL_ARGS

echo "=== Done ==="
