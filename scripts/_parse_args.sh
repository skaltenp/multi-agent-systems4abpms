#!/usr/bin/env bash
# Shared argument parser for experiment scripts.
# Separates server-compatible args (MODEL_ARGS) from run_experiment-only args (EXTRA_ARGS).
#
# Usage: source "$SCRIPT_DIR/_parse_args.sh" "$@"
#
# After sourcing:
#   MODEL_ARGS  — safe to pass to servers AND run_experiment.py (e.g. --model, --base-url)
#   EXTRA_ARGS  — only for run_experiment.py (e.g. --seed)
#   ALL_ARGS    — MODEL_ARGS + EXTRA_ARGS (for run_experiment.py)

_MODEL_PARTS=()
_EXTRA_PARTS=()
_skip_next=false

for _arg in "$@"; do
    if $_skip_next; then
        # This is the value for the previous flag
        if [[ "${_current_flag}" == "--seed" ]]; then
            _EXTRA_PARTS+=("${_current_flag}" "$_arg")
        else
            _MODEL_PARTS+=("${_current_flag}" "$_arg")
        fi
        _skip_next=false
        continue
    fi

    case "$_arg" in
        --seed)
            _current_flag="$_arg"
            _skip_next=true
            ;;
        --model|--base-url)
            _current_flag="$_arg"
            _skip_next=true
            ;;
        *)
            # Pass-through (e.g. --evaluate-only handled separately by callers)
            _MODEL_PARTS+=("$_arg")
            ;;
    esac
done

MODEL_ARGS="${_MODEL_PARTS[*]:-}"
EXTRA_ARGS="${_EXTRA_PARTS[*]:-}"
ALL_ARGS="${MODEL_ARGS:+$MODEL_ARGS }${EXTRA_ARGS}"
ALL_ARGS="${ALL_ARGS% }"  # trim trailing space
