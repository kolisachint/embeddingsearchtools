---
name: pr
description: |
  Open a release PR: create a branch, stage only your changes, rebase on
  origin/main, push, and open a PR labeled cargo:<bump> so the release
  workflow publishes on merge.
  Usage: /pr <patch|minor|major>
argument-hint: patch|minor|major
---
Open a release PR with bump type: **$1** (default to `patch` if empty or invalid).

Goal: get only the changes YOU made in this session onto a feature branch,
rebased on top of `origin/main`, then open a PR labeled `cargo:$1`. When that PR
is merged, `.github/workflows/release.yml` bumps the workspace version in
Cargo.toml, tags, publishes the crates, and builds the release binaries.

Follow these steps exactly. Stop and report if any step fails.

1. Resolve the bump type:
   - Use `$1`. If it is not one of `patch`, `minor`, or `major`, use `patch`.

2. Identify ONLY the files you changed in this session. Do NOT use
   `git add -A` or `git add .` (other agents may have uncommitted work in this
   worktree). List the specific paths you created, modified, or deleted.
   Confirm with `git status` that those are the files you intend to stage.

3. Sync with main before branching:
   ```bash
   git fetch origin main
   ```
   Determine if local `main` (or your base) is behind `origin/main`. If behind,
   you must rebase your work onto `origin/main` later (step 7). Do not run
   `git reset --hard`, `git checkout .`, `git clean`, or `git stash` — those can
   destroy other agents' uncommitted work.

4. Create a feature branch from the current state (named for the work, e.g.
   `pr/<short-slug>`):
   ```bash
   git switch -c pr/<short-slug>
   ```

5. Stage ONLY your files and commit:
   ```bash
   git add -- <your-file-1> <your-file-2> ...
   git commit -m "<concise, descriptive message>"
   ```
   Include `fixes #<number>` or `closes #<number>` in the commit body if there
   is a related issue or PR.

6. Run cargo checks to ensure code quality:
   ```bash
   cargo fmt --all --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```
   These run against the default (mock) backend, matching CI. Do not commit the
   real model weights in `crates/core/models/` — keep the empty placeholders.

7. Rebase your branch onto the latest `origin/main`:
   ```bash
   git rebase origin/main
   ```
   Resolve conflicts only in YOUR files. If a conflict appears in a file you did
   not modify, run `git rebase --abort` and stop and ask the user. Never force
   push over shared history beyond your own branch.

8. Push the branch:
   ```bash
   git push -u origin pr/<short-slug>
   ```

9. Open the PR with a body summarizing the changes (commit messages, affected
   crates, and any `fixes #N` references), and apply the bump label:
   ```bash
   gh pr create --title "<title>" --body-file <tmp-body.md> --label "cargo:$1"
   ```

10. Report the PR URL and the applied `cargo:$1` label.

On merge, the labeled PR triggers the release workflow which bumps the workspace
version in Cargo.toml, updates the lockfile, creates a git tag, and publishes the
crates and release binaries — so do not run the release process yourself here.
