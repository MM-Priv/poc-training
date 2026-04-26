#!/bin/bash
# Run once on the login node from /mnt/data/poc-training/.
# Creates .venv on the shared jail filesystem — visible on all worker nodes.
# Also stages the Llama 3.1 tokenizer for TorchTitan (reuses the cached 8B
# tokenizer — identical to 70B, vocab 128256) so no extra HF download is needed.
set -euo pipefail

export PATH="/root/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "uv $(uv --version)"
cd "$SCRIPT_DIR"
uv sync

ASSETS="$SCRIPT_DIR/assets/hf/llama-3.1-tokenizer"
SNAPSHOT=$(find ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots \
              -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n1)
if [ -n "$SNAPSHOT" ]; then
    mkdir -p "$ASSETS"
    for f in tokenizer.json tokenizer_config.json special_tokens_map.json config.json; do
        cp -L "$SNAPSHOT/$f" "$ASSETS/$f"
    done
    echo "Tokenizer staged at $ASSETS"
else
    echo "Note: Llama 3.1 8B not cached — run tokenize_data.py first to populate tokenizer."
fi

echo ""
echo "Environment ready. Activate with:"
echo "  source $SCRIPT_DIR/.venv/bin/activate"
