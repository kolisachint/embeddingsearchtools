# CI / Release workflows

The GitHub Actions workflows live in `.github/workflows/` (`ci.yml` and
`release.yml`). There is no manual activation step.

## Workflow details

### `ci.yml`

Runs on pushes to `main` and on PRs:
- `cargo fmt --all --check` — formatting
- `cargo clippy --workspace --all-targets -- -D warnings` — lints
- `cargo test --workspace` — all tests (default mock backend)

### `release.yml`

A single workflow triggered when a PR with a `cargo:patch`, `cargo:minor`, or
`cargo:major` label is merged. Runs five jobs:

1. **bump-and-tag** — reads the current `[workspace.package].version`, bumps it
   based on the label, updates intra-workspace path-dependency versions, commits
   to `main`, pushes, and creates an annotated `v*` tag
2. **publish** — publishes crates to crates.io in dependency order
   (`embsearch-core` → `embsearch-cli`), skipping any version already on the
   index so a partial run can be retried (needs the `CRATES_IO_TOKEN` secret).
   The publish uses the default backend — the empty model placeholders keep the
   packaged crate under the crates.io 10 MB cap
3. **create-release** — creates the GitHub release with auto-generated notes
   (runs in parallel with publish)
4. **build** — builds a **self-contained MiniLM binary** (`--features onnx`,
   real int8 weights fetched via `scripts/fetch-model.sh`) for four targets
   (Linux gnu x86_64, macOS x86_64 + aarch64, Windows x86_64) and attaches each
   archive plus a per-asset `.sha256`. GitHub Releases have no 10 MB cap, so
   bundling the ~23 MB model here is fine
5. **checksums** — aggregates a combined `SHA256SUMS` manifest for downloaders

## PR-based release flow

The recommended release process uses the `/pr` command (see
`.agents/commands/pr.md`):

1. **Agent runs `/pr patch`** (or `minor`/`major`) → Creates PR with
   `cargo:<bump>` label
2. **PR gets merged** → Triggers `release.yml`
3. **Release workflow** → Bumps version, tags, publishes crates, builds
   cross-platform binaries, and uploads checksums — all in one workflow

This ensures version bumps are reviewable and tied to specific changes.

## Why a single workflow?

Splitting version bumping and releasing across two workflows (bump → tag push →
release) does not work cleanly: tags pushed by the `GITHUB_TOKEN` do not trigger
other workflows (a GitHub Actions safety measure), so every release would need a
manual tag re-push. Combining both into a single workflow eliminates this.

## First-time setup

- Add the `CRATES_IO_TOKEN` repository secret (crates.io API token).
- Create the three release labels: `cargo:patch`, `cargo:minor`, `cargo:major`.
- Crate names `embsearch-core` / `embsearch-cli` must be available on crates.io.
