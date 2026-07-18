//! Ensures the bundled-model paths exist before the `onnx` feature's
//! `include_bytes!` runs.
//!
//! The real weights (`models/model.onnx`, `models/tokenizer.json`) are excluded
//! from the packaged crate (see `Cargo.toml` `exclude`) so crates.io stays under
//! its 10 MB cap. That means a crate built from crates.io with `--features onnx`
//! has no weight files at all — and `include_bytes!` would fail to compile.
//!
//! To keep the `onnx` feature compilable everywhere, this script creates an
//! **empty placeholder** for either file when it is missing. An empty file makes
//! `MiniLmEmbedder::from_bundled()` fail at ONNX session-build time with a clear
//! runtime error (not a compile error), while `--model <dir>` still works. When
//! a placeholder is synthesized we emit a `cargo:warning` so a developer who
//! forgot to run `scripts/fetch-model.sh` gets a visible nudge rather than a
//! silently non-functional binary.

use std::path::Path;

fn main() {
    // Only relevant when the model weights are actually bundled.
    if std::env::var_os("CARGO_FEATURE_ONNX").is_none() {
        return;
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    for name in ["models/model.onnx", "models/tokenizer.json"] {
        let path = Path::new(&manifest_dir).join(name);
        // Re-run if the file appears/changes (e.g. after fetch-model.sh).
        println!("cargo:rerun-if-changed={}", path.display());

        if !path.exists() {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(&path, b"").unwrap_or_else(|e| {
                panic!("failed to create model placeholder {}: {e}", path.display())
            });
            println!(
                "cargo:warning=bundling EMPTY MiniLM placeholder for {name} \
                 — run scripts/fetch-model.sh for real weights, or use --model <dir> at runtime"
            );
        }
    }
}
