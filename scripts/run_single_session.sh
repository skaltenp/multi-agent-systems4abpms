#!/usr/bin/env bash
# Run a single experiment session end-to-end (db setup, servers, run, cleanup).
#
# Usage:
#   ./scripts/run_single_session.sh <session_id> --model qwen3.5:9b [--force]
#
# Example:
#   ./scripts/run_single_session.sh \
#     processadaptation_base_rule_processadaptationmethod_add_tour_J09B_seed_42 \
#     --model qwen3.5:9b --force
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <session_id> --model <model> [--force] [--experiment-type <type>]"
    exit 1
fi

SESSION_ID="$1"
shift

# Parse --experiment-type from args (default: base)
EXPERIMENT_TYPE="base"
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--experiment-type" ]]; then
        EXPERIMENT_TYPE="$arg"
    fi
    prev="$arg"
done

# Split args: MODEL_ARGS (server-safe), ALL_ARGS (includes --seed etc.)
source "$SCRIPT_DIR/_parse_args.sh" "$@"

echo "=== Running single session: $SESSION_ID ==="

# 1. Setup DBs
echo "Setting up clean template databases..."
python run_experiment.py --setup-all-dbs

# 2. Start servers
source "$SCRIPT_DIR/_wait_for_server.sh"
source "$SCRIPT_DIR/_ollama_model.sh"
source "$SCRIPT_DIR/_kill_port.sh"

echo "Starting servers..."
kill_port 8000
kill_port 8001
python -m servers.process_server --experiment-type "$EXPERIMENT_TYPE" --port 8000 &
PID_PROCESS=$!
trap "kill $PID_PROCESS 2>/dev/null || true; wait $PID_PROCESS 2>/dev/null || true" EXIT
wait_for_server http://localhost:8000/docs

python -m servers.frame_server --experiment-type "$EXPERIMENT_TYPE" --port 8001 $MODEL_ARGS &
PID_FRAME=$!
trap "kill $PID_PROCESS $PID_FRAME 2>/dev/null || true; wait $PID_PROCESS $PID_FRAME 2>/dev/null || true" EXIT

wait_for_server http://localhost:8001/docs

warmup_ollama_model

# 3. Run single session
echo "Running session..."
python run_single_session.py "$SESSION_ID" $ALL_ARGS

unload_ollama_model

echo "=== Done ==="
