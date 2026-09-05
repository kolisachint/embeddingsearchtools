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
  --model <id>                Which embedding model to fetch (default: minilm)
                              minilm | minilm-512 | bge-small |
                              bge-small-prefixed | nomic
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
  minilm-512)
    # Identical weights to `minilm`, with the token limit raised 256 -> 512.
    #
    # This exists to separate two things the A0/A1 comparison confounded.
    # MiniLM's reference pipeline sets max_seq_length 256, and the consuming
    # chunker caps chunks at 1000 characters believing that to be about 256
    # tokens. Measured against this tokenizer on a real code corpus it is 313
    # tokens at the median, so `minilm` truncates 85.8% of chunks and drops
    # 20.3% of the corpus's tokens, always from the tail of a chunk. bge-small
    # at 512 drops 0.1%. So A1 was never the isolated "model effect" it was
    # billed as: it changed the model AND gave the dense leg a fifth more of
    # the corpus to read.
    #
    # BERT position embeddings go to 512, so this is a supported length rather
    # than an extrapolation — but it is past what the model was tuned for, and
    # that is the point: it prices the coverage half on its own. Run it as an
    # arm, not as a default.
    MODEL_URL="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
    TOKENIZER_URL="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
    read -r -d '' MODEL_SPEC <<'EOF' || true
{
  "name": "all-MiniLM-L6-v2-int8-512",
  "dim": 384,
  "pooling": "mean",
  "normalize": true,
  "max_tokens": 512
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
  nomic)
    # nomic-ai/nomic-embed-text-v1.5. The ceiling arm: 768-d and ~137 MB
    # quantized against bge-small's 384-d and 34 MB, so it is priced as "what
    # does a much larger bi-encoder buy", not as a shipping candidate.
    #
    # Every field below is from the model's own config rather than inferred:
    # `1_Pooling/config.json` sets mean pooling (NOT cls — it differs from
    # bge-small here), `word_embedding_dimension` is 768, and `config.json`
    # gives `max_position_embeddings` 2048 with no rotary scaling. The card's
    # 8192 `max_seq_length` needs that scaling to be real, so 2048 is the
    # honest cap; it costs nothing either way, since chunks reach ~313 tokens
    # and the tokenizer pads each batch to its own longest member.
    #
    # Both prefixes are mandatory for this model, not optional flavouring:
    # it was trained with asymmetric task instructions, and omitting them
    # silently degrades retrieval rather than failing.
    MODEL_URL="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_quantized.onnx"
    TOKENIZER_URL="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json"
    read -r -d '' MODEL_SPEC <<'EOF' || true
{
  "name": "nomic-embed-text-v1.5-int8",
  "dim": 768,
  "pooling": "mean",
  "normalize": true,
  "max_tokens": 2048,
  "query_prefix": "search_query: ",
  "document_prefix": "search_document: "
}
EOF
    ;;
  *)
    echo "unknown --model '$MODEL_ID' (expected: minilm, minilm-512, bge-small, bge-small-prefixed, nomic)" >&2
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
