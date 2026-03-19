#!/usr/bin/env bash
# Base experiment, classic agent method (140 runs)
# Usage: ./scripts/07_base_classic.sh [--model qwen3:32b]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source "$SCRIPT_DIR/_parse_args.sh" "$@"

echo "=== Base + Classic ==="

# 1. Setup DBs
if [[ -z "${DB_SETUP_DONE:-}" ]]; then
    echo "Setting up clean template databases..."
    python run_experiment.py --setup-all-dbs
fi

# 2. Start classic agent server (no frame server needed)
source "$SCRIPT_DIR/_wait_for_server.sh"
source "$SCRIPT_DIR/_ollama_model.sh"
source "$SCRIPT_DIR/_kill_port.sh"

echo "Starting classic agent server..."
kill_port 8002
python -m servers.classic_agent_server --experiment-type base --port 8002 &
PID_CLASSIC=$!
trap "kill $PID_CLASSIC 2>/dev/null || true; wait $PID_CLASSIC 2>/dev/null || true" EXIT
wait_for_server http://localhost:8002/docs

warmup_ollama_model

# 3. Run experiment
echo "Running 140 classic-method runs..."
python run_experiment.py --experiment-type base --method classic $ALL_ARGS

unload_ollama_model

# 4. Evaluate
echo "Evaluating..."
python run_experiment.py --evaluate-only --experiment-type base --method classic $ALL_ARGS

echo "=== Done ==="
