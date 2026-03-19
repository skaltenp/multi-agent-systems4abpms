#!/usr/bin/env bash
# Helper: extract model name from args and provide Ollama warmup/unload functions.
# Source this from experiment scripts after setting MODEL_ARGS.

_OLLAMA_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load Ollama env vars (base URL, API key) so scripts work regardless of shell state / nohup
if [[ -f "$_OLLAMA_HELPER_DIR/../.env.ollama" ]]; then
    set -a
    source "$_OLLAMA_HELPER_DIR/../.env.ollama"
    set +a
fi

# Extract --model value from MODEL_ARGS (empty if not set)
_extract_model() {
    local prev=""
    for arg in $MODEL_ARGS; do
        if [[ "$prev" == "--model" ]]; then
            echo "$arg"
            return
        fi
        prev="$arg"
    done
}

OLLAMA_MODEL="$(_extract_model)"

# Check if model is an Ollama model (not OpenAI/Anthropic)
_is_ollama_model() {
    [[ -n "$OLLAMA_MODEL" ]] && \
    [[ ! "$OLLAMA_MODEL" =~ ^gpt- ]] && \
    [[ ! "$OLLAMA_MODEL" =~ ^o[134]- ]] && \
    [[ ! "$OLLAMA_MODEL" =~ ^claude- ]]
}

# Unload all models, then load and pin the target model
warmup_ollama_model() {
    if _is_ollama_model && [[ -z "${OLLAMA_MANAGED:-}" ]]; then
        python "$_OLLAMA_HELPER_DIR/../experiment/ollama_model.py" warmup --model "$OLLAMA_MODEL"
    fi
}

# Unload all models
unload_ollama_model() {
    if _is_ollama_model && [[ -z "${OLLAMA_MANAGED:-}" ]]; then
        python "$_OLLAMA_HELPER_DIR/../experiment/ollama_model.py" unload_all
    fi
}
