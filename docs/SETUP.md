# Setup & release runbook

Status of this project: **the build + release pipeline is live and has shipped.**
This document is now a reference for *how it works* and *how to cut future
releases*, not a list of pending blockers. The historical "couldn't push from the
original environment" caveats have been resolved — see the changelog below.

## Current state (done)

- Rust workspace: `embsearch-core` (library) + `embsearch-cli` (`embsearch` binary).
- Embedder / Index / Store abstractions, `Database` API, CLI, NDJSON daemon,
  zero-dependency TypeScript client (`ts/client.ts`).
- Default build uses a deterministic `MockEmbedder`; real MiniLM is behind the
  `onnx` feature.
- `fmt` + `clippy -D warnings` clean; full test suite passes.
- **CI + release workflows are installed and tracked** in `.github/workflows/`
  (`ci.yml`, `release.yml`).
- **Releases have shipped**: tags `v0.1.1` and `v0.1.2` are pushed to origin;
  the crates.io publish + GitHub Release + cross-platform onnx binaries run
  automatically from the release workflow.

> `docs/workflows/` retains the original standalone copies of the workflow files
> for historical reference. The authoritative, running versions are the ones in
> `.github/workflows/`; they have since diverged (the live release workflow is
> PR-label-driven — see below).

---

## Cutting a release (the live flow)

Releases are **automated and driven by PR labels** — you do *not* tag by hand.

1. Open a PR with your changes.
2. Add one of these labels before merging:
   - `cargo:patch` → x.y.**z+1**
   - `cargo:minor` → x.**y+1**.0
   - `cargo:major` → **x+1**.0.0
3. Merge the PR into `main`.

On merge, `release.yml` automatically:

1. **Bumps** the workspace version in `Cargo.toml` (single source of truth;
   member crates inherit via `version.workspace = true`) and every intra-workspace
   pinned dependency, then commits `release: vX.Y.Z` and pushes an annotated tag.
2. **Publishes** `embsearch-core` then `embsearch-cli` to crates.io
   (`CRATES_IO_TOKEN` secret). Already-published versions are skipped, not failed.
3. **Creates a GitHub Release** for the tag with generated notes.
4. **Builds self-contained MiniLM binaries** (`--features onnx`, real ~23 MB
   int8 weights fetched via `scripts/fetch-model.sh`) for linux/macOS
   (x86_64 + arm64)/Windows and uploads them with per-asset SHA256.
5. **Aggregates** a `SHA256SUMS` file onto the release.

Merging without a `cargo:*` label makes no release — the workflow is a no-op.

---

## Working with the onnx backend locally

The `onnx` build bundles `crates/core/models/{model.onnx,tokenizer.json}` into
the binary via `include_bytes!`. In git these are **empty placeholders** so the
feature compiles before weights are present; supply real weights locally to run
real embeddings:

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

> **Don't commit the real weights.** They're ~23 MB and would bloat git. Keep the
> empty placeholders tracked:
> ```bash
> git update-index --skip-worktree crates/core/models/model.onnx crates/core/models/tokenizer.json
> ```
> (Undo with `--no-skip-worktree`.) The crates.io 10 MB cap is enforced by the
> `exclude` in `crates/core/Cargo.toml`, so `cargo package`/`publish` drops these
> files even when the real weights are present locally — you can publish from a
> weights-populated checkout safely.

Swap the model dir at runtime without rebuilding — `--model` is a per-command
flag (holds `model.onnx` + `tokenizer.json`):

```bash
embsearch query --path ./store --model <dir> "your query"
```

---

## crates.io notes

- `release.yml` publishes automatically; to sanity-check locally first:
  ```bash
  cargo publish -p embsearch-core --dry-run
  ```
- The packaged crate stays tiny regardless of working tree: real weights are under
  `exclude` in `crates/core/Cargo.toml`, so they never ship even after
  `fetch-model.sh`. A crate built from crates.io with `--features onnx` still
  compiles — `build.rs` synthesizes empty placeholders for the excluded files and
  prints a `cargo:warning` pointing at `scripts/fetch-model.sh`. Consumers who
  enable `onnx` supply weights via `embsearch <cmd> --model <dir>` or by dropping
  files into `crates/core/models/` and building locally.
- To bundle weights *inside* the published crate you'd need a crates.io
  package-size-limit increase (default 10 MB).

## (Optional) onnx checks in CI

`ci.yml` builds the default (mock) backend for speed. To also exercise the real
backend, add a job that runs `scripts/fetch-model.sh` then
`cargo build --features onnx` / `cargo test --features onnx`. Left out by default
so CI stays fast and doesn't depend on Hugging Face for every push.

---

## Reference

- Footprint: default (mock) binary ~1.1 MB; onnx binary ~35–45 MB total
  (~23 MB model + ~10–15 MB ONNX Runtime + binary).
- Model: `all-MiniLM-L6-v2`, int8 ONNX from `Xenova/all-MiniLM-L6-v2` on
  Hugging Face (see `scripts/fetch-model.sh` for exact URLs).
