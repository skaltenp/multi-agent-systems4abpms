#!/usr/bin/env bash
# Exception handling experiment, classic agent method (25 sampled runs)
# Usage: ./scripts/08_exception_handling_classic.sh [--model qwen3:32b]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source "$SCRIPT_DIR/_parse_args.sh" "$@"

echo "=== Exception Handling + Classic ==="

if [[ -z "${DB_SETUP_DONE:-}" ]]; then
    echo "Setting up clean template databases..."
    python run_experiment.py --setup-all-dbs
fi

source "$SCRIPT_DIR/_wait_for_server.sh"
source "$SCRIPT_DIR/_ollama_model.sh"
source "$SCRIPT_DIR/_kill_port.sh"

echo "Starting classic agent server..."
kill_port 8002
python -m servers.classic_agent_server --experiment-type exception_handling --port 8002 &
PID_CLASSIC=$!
trap "kill $PID_CLASSIC 2>/dev/null || true; wait $PID_CLASSIC 2>/dev/null || true" EXIT
wait_for_server http://localhost:8002/docs

warmup_ollama_model

echo "Running 25 sampled classic-method runs..."
python run_experiment.py --experiment-type exception_handling --method classic --sample 25 $ALL_ARGS

unload_ollama_model

echo "Evaluating..."
python run_experiment.py --evaluate-only --experiment-type exception_handling --method classic --sample 25 $ALL_ARGS

echo "=== Done ==="
