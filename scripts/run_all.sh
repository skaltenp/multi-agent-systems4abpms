#!/usr/bin/env bash
# Run all 9 experiments sequentially.
# Usage: ./scripts/run_all.sh [--model qwen3:32b]
#        ./scripts/run_all.sh --evaluate-only [--model qwen3:32b]
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
    echo "  EVALUATE ONLY — 9 configs"
    echo "  Args: ${MODEL_ARGS:-default (gpt-5.1)}"
    echo "========================================="

    # Base (all 140 runs per method)
    python run_experiment.py --evaluate-only --experiment-type base --method add $ALL_ARGS
    python run_experiment.py --evaluate-only --experiment-type base --method generate_bpmn $ALL_ARGS
    python run_experiment.py --evaluate-only --experiment-type base --method classic $ALL_ARGS

    # Exception handling (25 sampled runs per method)
    python run_experiment.py --evaluate-only --experiment-type exception_handling --method add --sample 25 $ALL_ARGS
    python run_experiment.py --evaluate-only --experiment-type exception_handling --method generate_bpmn --sample 25 $ALL_ARGS
    python run_experiment.py --evaluate-only --experiment-type exception_handling --method classic --sample 25 $ALL_ARGS

    # DB error (25 sampled runs per method)
    python run_experiment.py --evaluate-only --experiment-type exception_handling_db_error --method add --sample 25 $ALL_ARGS
    python run_experiment.py --evaluate-only --experiment-type exception_handling_db_error --method generate_bpmn --sample 25 $ALL_ARGS
    python run_experiment.py --evaluate-only --experiment-type exception_handling_db_error --method classic --sample 25 $ALL_ARGS

    # Print summary table
    python -c "
import glob, os
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

files = sorted(glob.glob('experiment_results/process_adaptation_results_*.csv'))
if not files:
    print('No result CSVs found.')
    exit()

frames = []
for f in files:
    df = pd.read_csv(f)
    fname = os.path.basename(f)
    if '_exception_handling_db_error' in fname:
        exp = 'db_error'
    elif '_exception_handling' in fname:
        exp = 'exc_handling'
    else:
        exp = 'base'
    for m in ['add', 'generate_bpmn', 'classic']:
        if f'_{m}' in fname.replace('_exception_handling', '').replace('_db_error', ''):
            method = m
            break
    else:
        method = '?'
    df['experiment'] = exp
    df['method'] = method
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

limit_cols = [c for c in df.columns if c.startswith(('recursion_limit_', 'timeout_', 'error_'))]

# Shorten column names for display
short = {
    'recursion_limit_frame': 'rec_frame',
    'recursion_limit_process': 'rec_proc',
    'timeout_frame': 'to_frame',
    'timeout_process': 'to_proc',
    'error_frame': 'err_frame',
    'error_process': 'err_proc',
}
short_cols = [short.get(c, c) for c in limit_cols]

# Per-model summaries
for model, mdf in df.groupby('model_name'):
    summary = mdf.groupby(['experiment', 'method']).agg(
        runs=('all_correct', 'size'),
        ok=('all_correct', 'sum'),
        rate=('all_correct', 'mean'),
        **{c: (c, 'sum') for c in limit_cols},
    ).reset_index()

    summary['ok'] = summary['ok'].astype(int)
    summary['rate'] = (summary['rate'] * 100).round(1).astype(str) + '%'
    for c in limit_cols:
        summary[c] = summary[c].astype(int)
    summary = summary.rename(columns=short)

    total_runs = summary['runs'].sum()
    total_ok = summary['ok'].sum()
    total_rate = f'{total_ok / total_runs * 100:.1f}%' if total_runs else '0%'

    print()
    print('=' * 130)
    print(f'  {model}')
    print('=' * 130)
    print(summary.to_string(index=False))
    print(f'  TOTAL: {total_runs} runs, {total_ok} ok ({total_rate})  |  {\"  \".join(f\"{c}={summary[c].sum()}\" for c in short_cols)}')
    print('=' * 130)

# Grand total
print()
total_runs = len(df)
total_ok = int(df['all_correct'].sum())
total_rate = f'{total_ok / total_runs * 100:.1f}%' if total_runs else '0%'
print(f'  GRAND TOTAL: {total_runs} runs, {total_ok} ok ({total_rate}) across {df[\"model_name\"].nunique()} model(s)')
"

    echo ""
    echo "========================================="
    echo "  EVALUATE COMPLETE — all 9 configs"
    echo "========================================="
    exit 0
fi

echo "========================================="
echo "  Running all experiments sequentially"
echo "  Args: ${MODEL_ARGS:-default (gpt-5.1)}"
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
bash "$SCRIPT_DIR"/02_base_bpmn.sh $ALL_ARGS
bash "$SCRIPT_DIR"/03_exception_handling_add.sh $ALL_ARGS
bash "$SCRIPT_DIR"/04_exception_handling_bpmn.sh $ALL_ARGS
bash "$SCRIPT_DIR"/05_db_error_add.sh $ALL_ARGS
bash "$SCRIPT_DIR"/06_db_error_bpmn.sh $ALL_ARGS
bash "$SCRIPT_DIR"/07_base_classic.sh $ALL_ARGS
bash "$SCRIPT_DIR"/08_exception_handling_classic.sh $ALL_ARGS
bash "$SCRIPT_DIR"/09_db_error_classic.sh $ALL_ARGS

echo "========================================="
echo "  All experiments complete"
echo "========================================="
