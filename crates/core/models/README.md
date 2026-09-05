# Bundled model weights

When built with `--features onnx`, the crate embeds these files into the binary
via `include_bytes!` (see `crates/core/src/embed.rs`):

| File | What it is |
|------|-----------|
| `model.onnx` | `all-MiniLM-L6-v2`, **int8-quantized** ONNX export |
| `tokenizer.json` | the matching WordPiece tokenizer |
| `model.json` | the [`ModelSpec`] — pooling, dim, token limit, prefixes, identity |

`model.json` is small and always committed (never a placeholder). It is not
optional metadata: pooling and the token limit cannot be read off the weights,
and a wrong guess produces degraded vectors with no error anywhere. So
`OnnxEmbedder::from_dir` **refuses a model directory without one**.

The `model_id` it yields is `<name>.<8 hex of a hash over every field>`.
Consumers invalidate their caches by comparing that string and nothing else, so
folding the whole spec into it is what makes "same name, different pooling"
impossible to ship.

The versions committed here are **empty placeholders** so the `onnx` feature
compiles before the real weights are available. With placeholders in place,
`OnnxEmbedder::from_bundled()` fails at ONNX session-build time with a clear
runtime error — it does **not** break compilation.

These two files are also listed under `exclude` in `../Cargo.toml`, so they are
never shipped in the crates.io package (which keeps it under the 10 MB cap) even
if you've fetched the real ~23 MB weights locally. Because an excluded file is
absent from a crate built off crates.io, `../build.rs` recreates an empty
placeholder when either file is missing and prints a `cargo:warning` — so the
`onnx` feature still compiles everywhere, just non-functionally until real
weights are supplied.

## Supplying the real weights

Drop the two real files in this directory (same names) and rebuild:

```bash
cargo build --release --features onnx
```

The canonical source is Hugging Face — the int8 ONNX export from
`Xenova/all-MiniLM-L6-v2`:

- `onnx/model_quantized.onnx`  → save here as `model.onnx`
- `tokenizer.json`

The easiest path is `scripts/fetch-model.sh`, which downloads both files into
this directory. The release workflow runs it before building the self-contained
`onnx` binaries, and CI-free local builds can run it too.

## Using an external model dir instead of bundling

To avoid embedding the weights, point the CLI at a directory holding
`model.onnx`, `tokenizer.json` and `model.json` at runtime:

```bash
embsearch query --path ./store --model /path/to/model-dir "your query"
```

`scripts/fetch-model.sh` builds such a directory for you, spec included:

```bash
scripts/fetch-model.sh --model bge-small --dest ./models-bge
embsearch serve --path ./store --model ./models-bge
```

Known model ids are `minilm` (the bundled default), `bge-small`, and
`bge-small-prefixed` (the same weights with BGE's retrieval instruction on the
query side — a distinct `name`, so the two cannot share a store).

`bge-small-prefixed` **measured worse than plain `bge-small`** and is not a
shipping candidate; it stays fetchable so the result can be re-tested on a
larger gold set. `bge-small` itself beat the bundled MiniLM on every endpoint
without reaching significance on any of them, so the bundled default has not
changed. See [`docs/embedder-strategy.md`](../../../docs/embedder-strategy.md#results-a0a2).

A store records the model that built it, and opening it with a different one is
refused rather than silently producing nonsense. Switching models means
re-indexing.
