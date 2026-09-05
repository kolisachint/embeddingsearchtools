#!/usr/bin/env bash
#
# Fetch an embedding model (ONNX weights + tokenizer) and write the matching
# `model.json` spec beside them.
#
# The spec is not optional decoration: `OnnxEmbedder` refuses a model directory
# without one, because pooling, token limit and query/document prefixes cannot
# be read off the weights and guessing them wrong is invisible in every output
# except search quality.
#
# Run this on a machine with Hugging Face access, then build with `--features
# onnx`. Portable across Linux/macOS/Windows-bash (needs curl).
#
#   scripts/fetch-model.sh                                  # bundled default
#   cargo build --release --features onnx
#
# Fetch a different model, or one into a standalone directory for `--model`:
#
#   scripts/fetch-model.sh --model bge-small --dest ./models-bge
#   embsearch serve --path ./store --model ./models-bge
#
# The cross-encoder reranker is a SEPARATE ~23 MB model and is NOT fetched by
# default. Bundling it tripled the released binary (26 MB -> 73 MB extracted)
# for a reranker that measured worse than the caller's own deterministic one on
# every query class but conceptual, so the release build ships without it. Pass
# --with-reranker to fetch it anyway, or hand the daemon a directory at run
# time with `--reranker-model <dir>`, which needs no rebuild.
#
#   scripts/fetch-model.sh --with-reranker
#
set -euo pipefail

WITH_RERANKER=0
MODEL_ID="minilm"
DEST=""

usage() {
  sed -n '2,30p' "$0"
  cat <<'EOF'

Options:
  --model <minilm|bge-small>  Which embedding model to fetch (default: minilm)
  --dest <dir>                Where to put it (default: crates/core/models)
  --with-reranker             Also fetch the cross-encoder reranker
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --with-reranker) WITH_RERANKER=1 ;;
    --model)
      [ $# -ge 2 ] || { echo "--model needs a value" >&2; exit 2; }
      MODEL_ID="$2"; shift
      ;;
    --model=*) MODEL_ID="${1#*=}" ;;
    --dest)
      [ $# -ge 2 ] || { echo "--dest needs a value" >&2; exit 2; }
      DEST="$2"; shift
      ;;
    --dest=*) DEST="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# Resolve the repo's models dir relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${DEST:-${SCRIPT_DIR}/../crates/core/models}"

# Per-model: weights URL, tokenizer URL, and the spec that describes how to run
# them. `pooling` and `max_tokens` come from each model's own reference
# pipeline (sentence-transformers config), not from a preference.
case "$MODEL_ID" in
  minilm)
    # Canonical int8 ONNX export of sentence-transformers/all-MiniLM-L6-v2.
    MODEL_URL="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
    TOKENIZER_URL="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
    read -r -d '' MODEL_SPEC <<'EOF' || true
{
  "name": "all-MiniLM-L6-v2-int8",
  "dim": 384,
  "pooling": "mean",
  "normalize": true,
  "max_tokens": 256
}
EOF
    ;;
  bge-small)
    # BAAI/bge-small-en-v1.5. CLS pooling, 512 tokens — both differ from
    # MiniLM, which is exactly why the spec travels with the weights.
    #
    # No query_prefix here. BGE's retrieval instruction ("Represent this
    # sentence for searching relevant passages: ") is optional for v1.5 and is
    # a separate eval arm, not a default: it belongs in a spec of its own so
    # the two are distinguishable by model_id.
    MODEL_URL="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx"
    TOKENIZER_URL="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/tokenizer.json"
    read -r -d '' MODEL_SPEC <<'EOF' || true
{
  "name": "bge-small-en-v1.5-int8",
  "dim": 384,
  "pooling": "cls",
  "normalize": true,
  "max_tokens": 512
}
EOF
    ;;
  bge-small-prefixed)
    # Same weights as bge-small, with the retrieval instruction on the query
    # side only. A distinct `name`, so the two produce different model ids and
    # a store built under one cannot be opened under the other.
    #
    # MEASURED WORSE THAN PLAIN `bge-small` — not a shipping candidate. Arm A2
    # existed to ask whether the prefix earns its place and the answer was no:
    # against A1 it cost MRR (-0.025 overall, -0.064 on the semantic subgroup)
    # to buy reach (R@50 +0.032). It helps find the right chunk somewhere in
    # the top 50 and hurts putting it near the top, which is the wrong trade
    # for a caller that reads a handful of results.
    #
    # Kept fetchable because that result is as underpowered as every other in
    # the run (p=0.454) and anyone enlarging the gold set should be able to
    # re-test it. See docs/embedder-strategy.md#results-a0a2.
    MODEL_URL="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx"
    TOKENIZER_URL="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/tokenizer.json"
    read -r -d '' MODEL_SPEC <<'EOF' || true
{
  "name": "bge-small-en-v1.5-int8-q",
  "dim": 384,
  "pooling": "cls",
  "normalize": true,
  "max_tokens": 512,
  "query_prefix": "Represent this sentence for searching relevant passages: "
}
EOF
    ;;
  *)
    echo "unknown --model '$MODEL_ID' (expected: minilm, bge-small, bge-small-prefixed)" >&2
    exit 2
    ;;
esac

# Cross-encoder used for reranking a shortlist. A separate model from the
# embedder above and not interchangeable with it: this one scores a
# (query, passage) pair jointly and cannot produce a standalone vector.
RERANKER_DIR="${MODELS_DIR}/reranker"
RERANKER_MODEL_URL="https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quantized.onnx"
RERANKER_TOKENIZER_URL="https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json"

mkdir -p "$MODELS_DIR"

echo "Fetching model '${MODEL_ID}' into ${MODELS_DIR}"

echo "Downloading model.onnx ..."
curl -fL --retry 3 -o "${MODELS_DIR}/model.onnx" "$MODEL_URL"

echo "Downloading tokenizer.json ..."
curl -fL --retry 3 -o "${MODELS_DIR}/tokenizer.json" "$TOKENIZER_URL"

echo "Writing model.json ..."
printf '%s\n' "$MODEL_SPEC" > "${MODELS_DIR}/model.json"

if [ "$WITH_RERANKER" = "1" ]; then
  mkdir -p "$RERANKER_DIR"

  echo "Downloading reranker/model.onnx ..."
  curl -fL --retry 3 -o "${RERANKER_DIR}/model.onnx" "$RERANKER_MODEL_URL"

  echo "Downloading reranker/tokenizer.json ..."
  curl -fL --retry 3 -o "${RERANKER_DIR}/tokenizer.json" "$RERANKER_TOKENIZER_URL"
fi

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

echo
echo "Fetched into ${MODELS_DIR}:"
echo "  model.onnx               ${model_size} bytes"
echo "  tokenizer.json           ${tok_size} bytes"
echo "  model.json               $(wc -c < "${MODELS_DIR}/model.json") bytes"

if [ "$WITH_RERANKER" = "1" ]; then
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
  echo "  reranker/model.onnx      ${rr_model_size} bytes"
  echo "  reranker/tokenizer.json  ${rr_tok_size} bytes"
else
  echo "  (reranker not fetched — pass --with-reranker if you want it bundled)"
fi
echo
# Record hashes so you can pin/verify them later if you want reproducibility.
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${MODELS_DIR}/model.onnx" "${MODELS_DIR}/tokenizer.json" "${MODELS_DIR}/model.json"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "${MODELS_DIR}/model.onnx" "${MODELS_DIR}/tokenizer.json" "${MODELS_DIR}/model.json"
fi
echo
echo "Done. Build the self-contained binary with:  cargo build --release --features onnx"
