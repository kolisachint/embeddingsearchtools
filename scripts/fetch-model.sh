#!/usr/bin/env bash
#
# Fetch the all-MiniLM-L6-v2 int8 ONNX weights + tokenizer and place them where
# the `onnx` build bundles them from (crates/core/models/).
#
# Run this on a machine with Hugging Face access, then build with `--features
# onnx`. Portable across Linux/macOS/Windows-bash (needs curl).
#
#   scripts/fetch-model.sh
#   cargo build --release --features onnx
#
set -euo pipefail

# Resolve the repo's models dir relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/../crates/core/models"

# Canonical int8 ONNX export of sentence-transformers/all-MiniLM-L6-v2.
MODEL_URL="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
TOKENIZER_URL="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

# Cross-encoder used for reranking a shortlist. A separate model from the
# embedder above and not interchangeable with it: this one scores a
# (query, passage) pair jointly and cannot produce a standalone vector.
RERANKER_DIR="${MODELS_DIR}/reranker"
RERANKER_MODEL_URL="https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quantized.onnx"
RERANKER_TOKENIZER_URL="https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json"

mkdir -p "$MODELS_DIR"

echo "Downloading model.onnx ..."
curl -fL --retry 3 -o "${MODELS_DIR}/model.onnx" "$MODEL_URL"

echo "Downloading tokenizer.json ..."
curl -fL --retry 3 -o "${MODELS_DIR}/tokenizer.json" "$TOKENIZER_URL"

mkdir -p "$RERANKER_DIR"

echo "Downloading reranker/model.onnx ..."
curl -fL --retry 3 -o "${RERANKER_DIR}/model.onnx" "$RERANKER_MODEL_URL"

echo "Downloading reranker/tokenizer.json ..."
curl -fL --retry 3 -o "${RERANKER_DIR}/tokenizer.json" "$RERANKER_TOKENIZER_URL"

# Sanity-check sizes so an HTML error page or truncated download is caught early.
model_size=$(wc -c < "${MODELS_DIR}/model.onnx")
tok_size=$(wc -c < "${MODELS_DIR}/tokenizer.json")
if [ "$model_size" -lt 1000000 ]; then
  echo "ERROR: model.onnx is only ${model_size} bytes — download likely failed." >&2
  exit 1
fi
if [ "$tok_size" -lt 10000 ]; then
  echo "ERROR: tokenizer.json is only ${tok_size} bytes — download likely failed." >&2
  exit 1
fi

rr_model_size=$(wc -c < "${RERANKER_DIR}/model.onnx")
rr_tok_size=$(wc -c < "${RERANKER_DIR}/tokenizer.json")
if [ "$rr_model_size" -lt 1000000 ]; then
  echo "ERROR: reranker/model.onnx is only ${rr_model_size} bytes — download likely failed." >&2
  exit 1
fi
if [ "$rr_tok_size" -lt 10000 ]; then
  echo "ERROR: reranker/tokenizer.json is only ${rr_tok_size} bytes — download likely failed." >&2
  exit 1
fi

echo
echo "Fetched into ${MODELS_DIR}:"
echo "  model.onnx               ${model_size} bytes"
echo "  tokenizer.json           ${tok_size} bytes"
echo "  reranker/model.onnx      ${rr_model_size} bytes"
echo "  reranker/tokenizer.json  ${rr_tok_size} bytes"
echo
# Record hashes so you can pin/verify them later if you want reproducibility.
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${MODELS_DIR}/model.onnx" "${MODELS_DIR}/tokenizer.json"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "${MODELS_DIR}/model.onnx" "${MODELS_DIR}/tokenizer.json"
fi
echo
echo "Done. Build the self-contained binary with:  cargo build --release --features onnx"
