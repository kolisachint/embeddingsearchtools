# Bundled model weights

When built with `--features onnx`, the crate embeds these files into the binary
via `include_bytes!` (see `crates/core/src/embed.rs`):

| File | What it is |
|------|-----------|
| `model.onnx` | `all-MiniLM-L6-v2`, **int8-quantized** ONNX export |
| `tokenizer.json` | the matching WordPiece tokenizer |

The versions committed here are **empty placeholders** so the `onnx` feature
compiles before the real weights are available. With placeholders in place,
`MiniLmEmbedder::from_bundled()` fails at ONNX session-build time with a clear
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

> This repository's build environment blocks outbound access to
> `huggingface.co`, so the weights cannot be fetched during the build here. Fetch
> them in an environment that permits Hugging Face (or download once and copy the
> files in), then rebuild with `--features onnx`.

## Using an external model dir instead of bundling

To avoid embedding the weights, point the CLI at a directory holding `model.onnx`
+ `tokenizer.json` at runtime:

```bash
embsearch query --path ./store --model /path/to/model-dir "your query"
```
