# Workflows pending installation

These are complete, ready-to-run workflow files that could not be pushed from
the session that wrote them: GitHub refuses `.github/workflows/*` writes from an
OAuth app without the `workflow` scope, the same constraint recorded in
[`../../SETUP.md`](../SETUP.md). They live here so they are reviewable and
version-controlled; they do **nothing** until copied into `.github/workflows/`.

```bash
cp docs/workflows/pending/eval-build.yml docs/workflows/pending/ci.yml .github/workflows/
git add .github/workflows/ && git commit -m "ci: install eval-build workflow, compile onnx on PRs"
git push   # needs a token with `workflow` scope
```

## `eval-build.yml` (new)

`workflow_dispatch` only. Builds one `--features onnx` binary plus a
self-contained directory per model (`model.onnx` + `tokenizer.json` +
`model.json`) and attaches them to a **prerelease**.

Prerelease is the safety property, not a formality: consumers resolve the
binary through `/releases/latest`, and GitHub excludes prereleases from that
endpoint, so nothing this workflow publishes can reach a user. It exists so a
retrieval eval can compare models with one binary and `--model <dir>` per arm
rather than one build per arm.

It also **smoke-tests every model pack** — indexes two documents, queries them,
and fails if the semantically matching one does not win. A wrong pooling, token
limit or ONNX input name compiles, links and runs perfectly; behaviour is the
only check that catches it.

Inputs: `models` (comma-separated ids from `scripts/fetch-model.sh`) and `tag`
(the prerelease tag, reused and clobbered across runs).

## `ci.yml` (modified)

Adds an `onnx` job that runs clippy over the ONNX backend. The existing `test`
job builds only the default mock backend, and the `onnx` module cannot be
compiled at all without network access to ONNX Runtime — so on a restricted
machine it is never compiled locally, and its first compile happened during a
release. The job needs no model weights: `build.rs` synthesizes empty
placeholders, which is a link-time concern rather than a compile-time one.
