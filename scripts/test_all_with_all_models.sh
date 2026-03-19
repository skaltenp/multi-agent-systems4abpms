#!/usr/bin/env bash
# Run test_all.sh sequentially for each model to verify everything works.
# Usage: ./scripts/test_all_with_all_models.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS=("gpt-5.1" "qwen3.5:35b" "qwen3.5:9b")

for model in "${MODELS[@]}"; do
    echo ""
    echo "######################################"
    echo "  Testing model: $model"
    echo "######################################"
    echo ""
    bash "$SCRIPT_DIR/test_all.sh" --model "$model"
done

echo ""
echo "######################################"
echo "  All models tested successfully"
echo "######################################"
