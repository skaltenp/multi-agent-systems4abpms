#!/usr/bin/env bash
# Smoke test: classic agent only — 1 adaptation x 1 tour x 1 seed across 3 experiment types.
# Total: 3 runs.
# Usage: ./scripts/test_classic.sh [--model gpt-5.1]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source "$SCRIPT_DIR/_wait_for_server.sh"
source "$SCRIPT_DIR/_kill_port.sh"

source "$SCRIPT_DIR/_parse_args.sh" "$@"

source "$SCRIPT_DIR/_ollama_model.sh"
TEST="--test"

echo "========================================="
echo "  TEST CLASSIC — 3 runs"
echo "  Args: ${MODEL_ARGS:-default (gpt-5.1)}"
echo "========================================="

separator() {
    echo ""
    echo "========================================="
    echo "  $1"
    echo "========================================="
}

start_classic_server() {
    local exp_type="$1"
    kill_port 8002
    separator "Starting classic agent server (${exp_type})"
    python -m servers.classic_agent_server --experiment-type "$exp_type" --port 8002 &
    PID_C=$!
    trap "kill $PID_C 2>/dev/null || true; wait $PID_C 2>/dev/null || true" EXIT
    wait_for_server http://localhost:8002/docs
}

stop_classic_server() {
    separator "Stopping classic agent server"
    kill $PID_C 2>/dev/null || true
    wait $PID_C 2>/dev/null || true
}

# --- Create clean template databases once ---
separator "Setting up clean template databases"
python run_experiment.py --setup-all-dbs

# --- Base + Classic ---

start_classic_server base
warmup_ollama_model

separator "[1/3] Base + Classic — running experiment"
python run_experiment.py --experiment-type base --method classic $TEST $ALL_ARGS
separator "[1/3] Base + Classic — evaluating"
python run_experiment.py --evaluate-only --experiment-type base --method classic $TEST $ALL_ARGS

unload_ollama_model
stop_classic_server

# --- Exception Handling + Classic ---
start_classic_server exception_handling
warmup_ollama_model

separator "[2/3] Exception Handling + Classic — running experiment"
python run_experiment.py --experiment-type exception_handling --method classic $TEST $ALL_ARGS
separator "[2/3] Exception Handling + Classic — evaluating"
python run_experiment.py --evaluate-only --experiment-type exception_handling --method classic $TEST $ALL_ARGS

unload_ollama_model
stop_classic_server

# --- DB Error + Classic ---
start_classic_server exception_handling_db_error
warmup_ollama_model

separator "[3/3] DB Error + Classic — running experiment"
python run_experiment.py --experiment-type exception_handling_db_error --method classic $TEST $ALL_ARGS
separator "[3/3] DB Error + Classic — evaluating"
python run_experiment.py --evaluate-only --experiment-type exception_handling_db_error --method classic $TEST $ALL_ARGS

unload_ollama_model
stop_classic_server

echo ""
echo "========================================="
echo "  TEST CLASSIC COMPLETE — 3 runs"
echo "========================================="
