# End-to-end setup runbook

This is the complete, self-contained checklist to finish the project from a
machine with full access (Hugging Face + a GitHub login that has the `workflow`
scope). Nothing here depends on the original build environment — everything you
need is committed in this repo.

## What's already done (on the branch)

- Rust workspace: `embsearch-core` (library) + `embsearch-cli` (`embsearch` binary).
- Embedder / Index / Store abstractions, `Database` API, CLI, NDJSON daemon,
  zero-dependency TypeScript client (`ts/client.ts`).
- Default build uses a deterministic `MockEmbedder`; real MiniLM is behind the
  `onnx` feature.
- `fmt` + `clippy -D warnings` clean; full test suite passes.
- Crates are publish-ready (verified with `cargo publish --dry-run`).

## What this runbook completes

1. Bundle the **real MiniLM int8 weights** and build/test the `onnx` backend.
2. Install the **CI + release workflows** (couldn't be pushed from the original
   environment — that login lacked GitHub's `workflow` scope).
3. Publish to **crates.io** and cut a **release** with self-contained binaries.

---

## 1. Bundle the real model and build the onnx backend

The `onnx` build bundles `crates/core/models/{model.onnx,tokenizer.json}` into
the binary via `include_bytes!`. Those files are committed as **empty
placeholders**; replace them with the real weights:

```bash
# Downloads all-MiniLM-L6-v2 int8 ONNX + tokenizer into crates/core/models/
scripts/fetch-model.sh

# Build the self-contained binary (ort auto-downloads ONNX Runtime on first build)
cargo build --release --features onnx

# Smoke-test real embeddings end-to-end
printf '%s\n' \
  '{"id":"a","text":"the quick brown fox"}' \
  '{"id":"b","text":"machine learning vector search"}' \
  | ./target/release/embsearch index --path ./store --input -
./target/release/embsearch query --path ./store "fast animal" -k 2
```

> **Do not commit the real weights.** They're ~23 MB and would bloat git and blow
> past the crates.io 10 MB package cap. Keep the empty placeholders tracked:
> ```bash
> git update-index --skip-worktree crates/core/models/model.onnx crates/core/models/tokenizer.json
> ```
> (Undo later with `--no-skip-worktree` if you ever need to.)

## 2. Install the CI + release workflows

The workflow files live in [`docs/workflows/`](workflows/) because the original
environment's git token was refused write access to `.github/workflows/`. Copy
them into place from a login that has the `workflow` scope:

```bash
mkdir -p .github/workflows
cp docs/workflows/ci.yml docs/workflows/release.yml .github/workflows/
git add .github/workflows/
git commit -m "Add CI and release workflows"
git push
```

- **`ci.yml`** — on every push/PR: `fmt`, `clippy -D warnings`, `test`, plus
  default-backend release-binary builds for linux/macOS/Windows as artifacts.
- **`release.yml`** — on a `v*.*.*` tag: builds **self-contained MiniLM binaries**
  (fetches weights, `--features onnx`) into a GitHub Release, and publishes both
  crates to crates.io.

## 3. crates.io

The `CRATES_IO_TOKEN` repository secret is already set and is what
`release.yml` reads. Before the first real publish, sanity-check locally:

```bash
cargo publish -p embsearch-core --dry-run
```

Notes:
- The publish job intentionally does **not** fetch weights, so the packaged crate
  stays tiny (empty placeholders). Consumers who enable `onnx` supply weights via
  `embsearch --model <dir>` or by dropping files into `crates/core/models/` and
  building locally.
- If you ever want the weights bundled *inside* the published crate, you'd need to
  request a crates.io package-size-limit increase (default is 10 MB).
- Crate names `embsearch-core` / `embsearch-cli` must be available on crates.io.
  If taken, rename in the `[package]` names and the CLI's dependency, then retry.

## 4. Cut a release

```bash
git tag v0.1.0
git push origin v0.1.0
```

That fires `release.yml`: cross-platform self-contained binaries attach to the
GitHub Release for tag `v0.1.0`, and `embsearch-core` then `embsearch-cli`
publish to crates.io.

## 5. (Optional) also run onnx checks in CI

`ci.yml` builds the default (mock) backend for speed. To also exercise the real
backend in CI, add a job that runs `scripts/fetch-model.sh` then
`cargo build --features onnx` / `cargo test --features onnx`. Left out by default
so CI stays fast and doesn't depend on Hugging Face for every push.

---

## Reference

- Footprint: default (mock) binary ~1.1 MB; onnx binary ~35–45 MB total
  (~23 MB model + ~10–15 MB ONNX Runtime + binary).
- Model: `all-MiniLM-L6-v2`, int8 ONNX from `Xenova/all-MiniLM-L6-v2` on
  Hugging Face (see `scripts/fetch-model.sh` for exact URLs).
- Swap the model dir at runtime without rebuilding: `embsearch --model <dir> ...`
  (dir holds `model.onnx` + `tokenizer.json`).
