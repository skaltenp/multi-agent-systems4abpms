#!/usr/bin/env bash
# Smoke test: runs 1 adaptation x 1 tour x 1 seed across all 6 experiment configs.
# Total: 6 runs (1 per config) instead of 380.
# Usage: ./scripts/test_all.sh [--model qwen3:32b]
#        ./scripts/test_all.sh --evaluate-only [--model qwen3:32b]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source "$SCRIPT_DIR/_wait_for_server.sh"
source "$SCRIPT_DIR/_kill_port.sh"

# Parse --evaluate-only flag, then split remaining into MODEL_ARGS vs EXTRA_ARGS
EVAL_ONLY=false
_REMAINING=()
for arg in "$@"; do
    if [[ "$arg" == "--evaluate-only" ]]; then
        EVAL_ONLY=true
    else
        _REMAINING+=("$arg")
    fi
done
source "$SCRIPT_DIR/_parse_args.sh" "${_REMAINING[@]}"

source "$SCRIPT_DIR/_ollama_model.sh"
TEST="--test"

if $EVAL_ONLY; then
    echo "========================================="
    echo "  EVALUATE ONLY — 9 configs"
    echo "  Args: ${MODEL_ARGS:-default (gpt-5.1)}"
    echo "========================================="

    for exp_type in base exception_handling exception_handling_db_error; do
        for method in add generate_bpmn classic; do
            echo ""
            echo ">>> Evaluating: ${exp_type} + ${method}"
            python run_experiment.py --evaluate-only --experiment-type "$exp_type" --method "$method" $TEST $ALL_ARGS
        done
    done

    echo ""
    echo "========================================="
    echo "  EVALUATE COMPLETE — all 6 configs"
    echo "========================================="
    exit 0
fi

echo "========================================="
echo "  TEST MODE — 9 runs total"
echo "  Args: ${MODEL_ARGS:-default (gpt-5.1)}"
echo "========================================="

separator() {
    echo ""
    echo "========================================="
    echo "  $1"
    echo "========================================="
}

start_servers() {
    local exp_type="$1"
    shift
    # Start process server first — frame server connects to it during startup
    kill_port 8000
    kill_port 8001
    separator "Starting process server (${exp_type})"
    python -m servers.process_server --experiment-type "$exp_type" --port 8000 &
    PID_P=$!
    trap "kill $PID_P 2>/dev/null || true; wait $PID_P 2>/dev/null || true" EXIT
    wait_for_server http://localhost:8000/docs

    separator "Starting frame server (${exp_type})"
    python -m servers.frame_server --experiment-type "$exp_type" --port 8001 "$@" &
    PID_F=$!
    trap "kill $PID_P $PID_F 2>/dev/null || true; wait $PID_P $PID_F 2>/dev/null || true" EXIT
    wait_for_server http://localhost:8001/docs
}

stop_servers() {
    separator "Stopping servers"
    kill $PID_P $PID_F 2>/dev/null || true
    wait $PID_P $PID_F 2>/dev/null || true
}

# --- Create clean template databases once ---
separator "Setting up clean template databases"
python run_experiment.py --setup-all-dbs

# --- Base + Add ---
start_servers base $MODEL_ARGS
warmup_ollama_model

separator "[1/6] Base + Add — running experiment"
python run_experiment.py --experiment-type base --method add $TEST $ALL_ARGS
separator "[1/6] Base + Add — evaluating"
python run_experiment.py --evaluate-only --experiment-type base --method add $TEST $ALL_ARGS

unload_ollama_model
stop_servers

# --- Base + BPMN ---
start_servers base $MODEL_ARGS
warmup_ollama_model

separator "[2/6] Base + BPMN — running experiment"
python run_experiment.py --experiment-type base --method generate_bpmn $TEST $ALL_ARGS
separator "[2/6] Base + BPMN — evaluating"
python run_experiment.py --evaluate-only --experiment-type base --method generate_bpmn $TEST $ALL_ARGS

unload_ollama_model
stop_servers

# --- Exception Handling + Add ---
start_servers exception_handling $MODEL_ARGS
warmup_ollama_model

separator "[3/6] Exception Handling + Add — running experiment"
python run_experiment.py --experiment-type exception_handling --method add $TEST $ALL_ARGS
separator "[3/6] Exception Handling + Add — evaluating"
python run_experiment.py --evaluate-only --experiment-type exception_handling --method add $TEST $ALL_ARGS

unload_ollama_model
stop_servers

# --- Exception Handling + BPMN ---
start_servers exception_handling $MODEL_ARGS
warmup_ollama_model

separator "[4/6] Exception Handling + BPMN — running experiment"
python run_experiment.py --experiment-type exception_handling --method generate_bpmn $TEST $ALL_ARGS
separator "[4/6] Exception Handling + BPMN — evaluating"
python run_experiment.py --evaluate-only --experiment-type exception_handling --method generate_bpmn $TEST $ALL_ARGS

unload_ollama_model
stop_servers

# --- DB Error + Add ---
start_servers exception_handling_db_error $MODEL_ARGS
warmup_ollama_model

separator "[5/6] DB Error + Add — running experiment"
python run_experiment.py --experiment-type exception_handling_db_error --method add $TEST $ALL_ARGS
separator "[5/6] DB Error + Add — evaluating"
python run_experiment.py --evaluate-only --experiment-type exception_handling_db_error --method add $TEST $ALL_ARGS

unload_ollama_model
stop_servers

# --- DB Error + BPMN ---
start_servers exception_handling_db_error $MODEL_ARGS
warmup_ollama_model

separator "[6/6] DB Error + BPMN — running experiment"
python run_experiment.py --experiment-type exception_handling_db_error --method generate_bpmn $TEST $ALL_ARGS
separator "[6/6] DB Error + BPMN — evaluating"
python run_experiment.py --evaluate-only --experiment-type exception_handling_db_error --method generate_bpmn $TEST $ALL_ARGS

unload_ollama_model
stop_servers

# === Classic agent runs (only need classic_agent_server) ===

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

# --- Base + Classic ---
start_classic_server base
warmup_ollama_model

separator "[7/9] Base + Classic — running experiment"
python run_experiment.py --experiment-type base --method classic $TEST $ALL_ARGS
separator "[7/9] Base + Classic — evaluating"
python run_experiment.py --evaluate-only --experiment-type base --method classic $TEST $ALL_ARGS

unload_ollama_model
stop_classic_server

# --- Exception Handling + Classic ---
start_classic_server exception_handling
warmup_ollama_model

separator "[8/9] Exception Handling + Classic — running experiment"
python run_experiment.py --experiment-type exception_handling --method classic $TEST $ALL_ARGS
separator "[8/9] Exception Handling + Classic — evaluating"
python run_experiment.py --evaluate-only --experiment-type exception_handling --method classic $TEST $ALL_ARGS

unload_ollama_model
stop_classic_server

# --- DB Error + Classic ---
start_classic_server exception_handling_db_error
warmup_ollama_model

separator "[9/9] DB Error + Classic — running experiment"
python run_experiment.py --experiment-type exception_handling_db_error --method classic $TEST $ALL_ARGS
separator "[9/9] DB Error + Classic — evaluating"
python run_experiment.py --evaluate-only --experiment-type exception_handling_db_error --method classic $TEST $ALL_ARGS

unload_ollama_model
stop_classic_server

echo ""
echo "========================================="
echo "  TEST COMPLETE — all 9 configs ran"
echo "========================================="
