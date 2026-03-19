#!/usr/bin/env bash
# Run all 3 "add" experiments sequentially (base + exception_handling + db_error).
# Usage: ./scripts/run_all_add.sh [--model qwen3:32b] [--seed 42]
#        ./scripts/run_all_add.sh --evaluate-only [--model qwen3:32b]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

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

if $EVAL_ONLY; then
    echo "========================================="
    echo "  EVALUATE ONLY — add method, 3 configs"
    echo "  Args: ${ALL_ARGS:-default (gpt-5.1)}"
    echo "========================================="

    python run_experiment.py --evaluate-only --experiment-type base --method add $ALL_ARGS
    python run_experiment.py --evaluate-only --experiment-type exception_handling --method add --sample 25 $ALL_ARGS
    python run_experiment.py --evaluate-only --experiment-type exception_handling_db_error --method add --sample 25 $ALL_ARGS

    echo ""
    echo "========================================="
    echo "  EVALUATE COMPLETE — add method"
    echo "========================================="
    exit 0
fi

echo "========================================="
echo "  Running all add experiments"
echo "  Args: ${ALL_ARGS:-default (gpt-5.1)}"
echo "========================================="

# Create clean template databases once for all scripts
python run_experiment.py --setup-all-dbs

# Warmup once for all scripts; OLLAMA_MANAGED tells subscripts to skip their own
warmup_ollama_model
export OLLAMA_MANAGED=1
export DB_SETUP_DONE=1

cleanup_ollama() { unload_ollama_model; }
trap cleanup_ollama EXIT

bash "$SCRIPT_DIR"/01_base_add.sh $ALL_ARGS
bash "$SCRIPT_DIR"/03_exception_handling_add.sh $ALL_ARGS
bash "$SCRIPT_DIR"/05_db_error_add.sh $ALL_ARGS

echo "========================================="
echo "  All add experiments complete"
echo "========================================="
