# Runbook: measuring an embedding model change

How to run the model comparison that
[`embedder-strategy.md`](embedder-strategy.md) specifies. Everything below is
built and pushed; the steps are what a machine with Hugging Face and ONNX
Runtime access needs to do to produce numbers.

## Where this was left

Written in an environment that could not finish it, for three reasons worth
knowing before you start — none of them apply on a normal laptop:

| Blocked | Why | Consequence | Status |
|---|---|---|---|
| `huggingface.co` | egress policy | no model weights could be fetched | cleared |
| `cdn.pyke.io` | egress policy | `--features onnx` could not be compiled *at all* | cleared |
| `.github/workflows/*` | OAuth token lacks `workflow` scope | the eval workflow could not be installed | **done** — installed |

So the `onnx` code path was never compiled locally. It **has** been executed —
the released v0.3.2 binary was downloaded and run (see
[Verified](#verified--not-verified) below) — but the eval itself has not been
run, and no model has been compared to another.

### Step 0: install the eval workflow — done

`eval-build.yml` and the `onnx` CI job now live in `.github/workflows/`, and
`docs/workflows/pending/` is gone with them. Nothing to copy.

The caution that used to sit here — take only the `onnx` job, because
`pending/ci.yml` predates the `as_chunks` clippy fix — turned out to be moot:
that fix was to `simd.rs`, and by merge time the two `test` jobs were
byte-identical, so both files were installed as they stood.

To run it: **Actions → Eval build → Run workflow**, models
`minilm,bge-small,bge-small-prefixed`.

It publishes to a **prerelease** tag (`eval-latest`). That is a safety
property, not a formality: consumers resolve the binary through
`/releases/latest` (`hoocode`'s `tools-manager.ts:221`), and GitHub excludes
prereleases from that endpoint, so nothing the workflow publishes can reach a
user.

It also smoke-tests every model pack — indexes two documents, queries them,
fails if the semantically matching one loses. A wrong pooling or token limit
compiles, links and runs perfectly, so behaviour is the only check that catches
one.

**Building locally instead**, if you would rather not wait on CI:

```bash
scripts/fetch-model.sh --model minilm                       # bundled default
scripts/fetch-model.sh --model bge-small --dest ./pack/bge-small
scripts/fetch-model.sh --model bge-small-prefixed --dest ./pack/bge-small-prefixed
cargo build --release -p embsearch-cli --features onnx
```

If `cdn.pyke.io` is unreachable, see the `ORT_LIB_LOCATION` recipe in
[`../README.md`](../README.md).

## Running an arm

One binary serves every arm; the model is a flag.

```bash
cd hoocode/packages/coding-agent

bun run search-eval -- \
  --corpus-ref <SHA> \
  --embsearch-binary /path/to/embsearch \
  --model-dir /path/to/pack/bge-small \
  --daemon-hybrid \
  --out runs/a1-bge-small.json
```

Omit `--model-dir` for the MiniLM control: it uses the model bundled in the
binary.

`--model-dir` suffixes the store directory, so arms do not collide and each
pays its own indexing cost once (~17k chunks). The suffix is empty without the
flag, so an ordinary run reuses the store it always did.

### Arms

Hold everything else fixed: same corpus SHA, exclusions, fusion `k`, rerank
weights, index kind.

| arm | `--model-dir` | chunk cap | purpose |
|---|---|---|---|
| A0 | *(none — bundled MiniLM)* | 1000 | control, re-measured |
| A1 | `pack/bge-small` | 1000 | model effect, isolated |
| A2 | `pack/bge-small-prefixed` | 1000 | does the query instruction earn its place |
| A3 | winner of A1/A2 | 2000 | the window a 512-token model unlocks |
| A4 | winner of A1/A2 | 1500 | brackets A3 if A3 moves |

A3/A4 need `CHUNK_MAX_CHARS` raised in `hoocode`'s `chunker.ts` and
`CHUNKER_VERSION` bumped. That changes `retrievalSourceHash`, which is correct
and visible — it means A3 is not comparable to A0–A2 on anything but the
metrics themselves.

**A0 must be re-measured.** The committed
`test/fixtures/search-eval-baseline.json` cannot serve as the control: it is
the contaminated corpus `ffeaad9` on embsearch 0.2.0 with `corpusExcluded`
absent, i.e. taken before the gold fixture was removed from the corpus it
searches. Pick a corpus SHA at or after `hoocode` `419ba17` — earlier trees
lack the harness changes, and the gold set is anchored into two files those
changes moved.

### Comparing

```bash
bun run search-eval:compare -- "auto +rr" --against runs/a0-minilm.json runs/a1-bge-small.json
```

Paired sign test over the same 62 queries. Pre-registered endpoints:

- **Primary** — MRR on `auto +rr`, all 62 queries, p ≤ 0.05.
- **Secondary** — R@10 and MRR on the 24-query semantic subgroup (conceptual +
  cross-file + boundary). Declared in advance so it is not a post-hoc slice.
- **Guardrail** — exact-symbol and error-fragment must not regress. Both sit at
  100% R@10 (MRR 0.888 / 0.950), carried by BM25; a dense change that costs
  them is a net loss whatever it wins.
- **Reach** — R@50, which is what the chunk-window arms are actually about.

At n=4 (boundary) and n=6 (cross-file) nothing is ever significant. Those are
descriptive; the 24-query subgroup is the smallest unit any claim gets made
about.

**What would falsify the premise.** If A1/A2 do not beat A0 on the semantic
subgroup, the embedder hypothesis in `hoocode`'s
`hybrid-retrieval-design.md:853` is wrong: the remaining failures are not the
model, and the next lever is the cross-encoder, not a bigger bi-encoder. Report
that as a result rather than reaching for a third model.

## Verified / not verified

Measured against the released v0.3.2 binary, downloaded and run:

- ✅ `ModelSpec` parses from the bundled `model.json`; the daemon reports
  `all-MiniLM-L6-v2-int8.1f635576`.
- ✅ Mean pooling produces real semantics — "wolf chasing a rabbit" ranks a fox
  document over an infrastructure one.
- ✅ The refactor is behaviour-preserving: short documents score bit-identical
  `0.38157177` on v0.3.1 and v0.3.2.
- ✅ Truncation changes long-document vectors as intended (cosine 0.230 → 0.192
  on a ~4,800-token document). It does **not** prevent a crash — v0.3.1 indexes
  the same document without error. The defect was a silently worse vector, not
  a failure.
- ✅ `store-info` reads a v0.3.1 store that the v0.3.2 binary refuses to open.
- ❌ **CLS pooling has never been executed.** No BGE weights were reachable.
  The eval-build smoke test is the first thing that will run it.
- ❌ **The query-prefix path has never been executed**, for the same reason.
- ❌ No model has been compared to another. Every number in
  `embedder-strategy.md` is a prior measurement of the *current* embedder.

`OnnxEmbedder` is now compiled on every PR (the `onnx` job in `ci.yml`), which
closes the gap where that code path got its first compile during a release.

## Open items

1. **Run the eval.** Everything above.
2. **A3/A4 need a chunker change** in `hoocode` — raise `CHUNK_MAX_CHARS`, bump
   `CHUNKER_VERSION`, and fold in the blank-line fix parked at
   `hybrid-retrieval-design.md:846` so users pay one rebuild rather than two.
3. **Nomic** stays documented-but-unbundled. `--model <dir>` reaches it; adding
   a `nomic` id to `fetch-model.sh` needs a spec with `search_query:` /
   `search_document:` prefixes and `dim` 768.
4. **Toolchain drift.** `dtolnay/rust-toolchain@stable` means a new clippy lint
   can turn `main` red without a commit — it did, via
   `chunks_exact_to_as_chunks`. Worth a `rust-toolchain.toml` if that recurs.
