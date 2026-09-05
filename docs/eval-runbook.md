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

### Screening runs: `--fast`

A full arm indexes 22k chunks — about 6 min on MiniLM, 22 on bge-small and
nearly two hours on nomic. `--fast <chunks>` cuts the corpus to a budget so a
whole sweep fits in a coffee break:

```bash
bun run search-eval -- --corpus-ref <SHA> --fast 6000 --daemon-hybrid \
  --embsearch-binary /path/to/embsearch --out runs/f1.json
```

It keeps **every gold-bearing file** and draws distractors with a fixed seed, so
all 62 queries stay answerable and two arms see byte-identical corpora. The
subsample is folded into the worktree path and the store key, so a `--fast` run
cannot collide with a full one.

**It is not a cheap version of the real eval.** A smaller distractor pool makes
retrieval easier: at 6,027 chunks the A0 control reads MRR 0.659 against its
full-corpus 0.595. More queries also tie, which costs the sign test power it was
already short of. Use it to rank arms against each other, never to state an
absolute number. Records carry `corpusSubsample` so the two kinds cannot be
mistaken for one another.

It does track the full corpus on the thing it is for — A0→A1 measured +0.059
MRR and +0.131 semantic MRR at 6k against +0.043 and +0.100 on the full corpus,
same direction and rough size at a quarter of the cost.

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
- ✅ **CLS pooling executes and is sane.** Ran locally against real
  `Xenova/bge-small-en-v1.5` int8 weights: on "wolf chasing a rabbit" the fox
  document scores 0.702 against 0.360 for an unrelated infrastructure one.
- ✅ **The query-prefix path executes.** `bge-small-prefixed` produces a
  distinct `model_id` (`…-int8-q.5e71728d`) and different scores (fox 0.661),
  confirming the prefix reaches the tokenizer rather than being dropped.
- ✅ **The onnx backend compiles locally**, in 49s once `cdn.pyke.io` is
  reachable. Its first compile is no longer during a release.
- ✅ **A0–A2 have run.** See
  [Results](embedder-strategy.md#results-a0a2). bge-small wins on every
  endpoint, nothing clears p≤0.05, the prefix loses to no prefix.
- ✅ **Indexing cost is now measured per model, not inferred.** Run records
  carry `timing.indexSeconds` / `timing.querySeconds`, and
  `scripts/bench-index.py` times the daemon's `bulk` path directly. The two
  agree within 4% (62 vs 64.3 chunks/s for MiniLM, 17.0 vs 16.9 for
  bge-small), which is the check that either one is measuring what it claims.
- ✅ **The 3.5× is decomposed**: ~1.8× window, ~2.1× model depth. See
  [the cost section](embedder-strategy.md#the-cost-nobody-priced).
- ✅ **The truncation confound is ruled out.** `minilm-512` was added as the
  missing control and shows the coverage difference is not what A1 won on.
- ✅ **`nomic` runs.** It needed no code change — 768 dims, mean pooling and
  both `search_*` prefixes are all data in `model.json`, and its ONNX export
  does carry `token_type_ids`, which was the one thing that could have forced
  one.
- ⚠️ **A3/A4 ran, but not as written.** Their premise — "the window the model
  unlocks" — is void: at cap 1000 only 0.5% of chunks exceed 512 tokens, so
  there is no headroom to unlock, and cap 2000 *overruns* the window (66% of
  chunks truncated, 17.6% of tokens dropped). Re-specified on the record as a
  descriptive "should the shipped cap go up" arm. Cap 2000 wins every metric
  and indexes 26% **faster** — the cost assumption in this runbook was
  backwards — but nothing is significant, the guardrail slips for the first
  time, and line-overlap scoring mechanically favours bigger chunks. Not
  shippable evidence; see
  [A3/A4](embedder-strategy.md#a3a4-the-premise-was-void-so-the-arm-was-re-specified).
- ❌ **N1 (nomic) is priced but not scored.** At 3.2 chunks/s it is 20× MiniLM
  — 114 min for a full corpus — and 137 MB quantized against bge-small's 34.
  Ceiling-pricing only, as intended; a candidate it is not. It does *run*
  correctly (768-d, mean pooling, both prefixes), so scoring it is a matter of
  patience, not plumbing.

  An attempt at the 6k screening arm was abandoned after 2h20m having indexed
  ~2,000 of 6,028 chunks — under 0.7 chunks/s against the 3.2 the standalone
  bench measured. **The gap is unexplained.** It is not the eval path in
  general: bge-small runs at 17.0 chunks/s under the harness against 16.9 in
  the bench, so only nomic diverges. The untested hypothesis is input shapes —
  `bench-index.py` sends uniform 48-record batches while the indexer batches
  *per file* (~16 records on average, highly variable), and ORT plans and
  allocates per distinct input shape, which a 137 MB model would feel far more
  than a 34 MB one. Worth confirming before anyone reads either nomic number as
  its real cost; bucketing batches by length would be the fix if it holds.

`OnnxEmbedder` is now compiled on every PR (the `onnx` job in `ci.yml`), which
closes the gap where that code path got its first compile during a release.

## Open items

1. **Decide on A1 — currently: not shipping, MiniLM stays.** It wins every
   endpoint and clears none of them, and the cost is concrete where the benefit
   is uncertain: 3.5× indexing time, paid by every user at once when `model_id`
   changes.

   The argument that settled it: **you get one free re-index and this would
   spend it.** A3/A4 force an invalidation too, and this note already suspects
   the chunk window "may matter more than the model's own retrieval delta" — so
   shipping the model alone burns the invalidation on the smaller, unproven
   half. If both land, they should land together.

   The binding constraint is the gold set, not the model: 42 of 62 queries tied,
   so the eval cannot resolve a difference this size. Enlarging the semantic
   classes (24 queries, where the whole effect lives) is far cheaper than a
   3.5× indexing regression and is what would actually let anyone decide.

   One caveat on that 3.5×: 17→60 min is *total eval wall time*, dominated by
   embedding ~40k chunks but not isolated from query work. A0→A1 is a clean
   comparison (both foreground, both timed); A2's ~86 min ran against other
   load and should not be trusted. Per-*query* latency was never measured — one
   forward pass on a short query against a warm daemon, likely single-digit ms
   for both, and almost certainly not the thing to worry about.

   **Resolved, 2026-09-05.** "~40k chunks" was the tell: the corpus is ~20k and
   the harness was indexing it twice per arm, into a dense store and a hybrid
   one, for identical vectors. That is fixed, so those totals were about double
   the real indexing cost. Both halves are now recorded separately as
   `timing.indexSeconds` and `timing.querySeconds`. The guess about query
   latency was right and can stop being a guess: ~17s for 62 queries across all
   16 configs, near-identical on every model, against 97s–354s of indexing.
   Query cost is not where a model choice is decided.
2. **A cap change needs a span-normalised metric first.** `--chunk-max-chars`
   now makes the sweep runnable without touching `CHUNK_MAX_CHARS`, and the
   sweep has been run. What it cannot currently settle is how much of cap
   2000's win is retrieval and how much is line-overlap scoring rewarding
   bigger spans; until a metric separates those, the arm describes rather than
   decides. Shipping a cap change would still need `CHUNKER_VERSION` bumped and
   a re-index for every user.

   The blank-line fix this item said to "fold in" is **already landed**
   (`chunker.ts:68-89`, `CHUNKER_VERSION` 2); the design note calling it parked
   is stale. Nothing to carry.
3. ~~**Drop `bge-small-prefixed`.**~~ Done, but softened on reflection: it is
   marked as measured-worse in `fetch-model.sh` and off the eval-build default
   list, **not deleted**. The risk was shipping it, not fetching it — and A2's
   negative result is as underpowered (p=0.454) as A1's positive one, so
   anyone enlarging the gold set should be able to re-test the prefix rather
   than find the option gone.
4. **Consider replacing `test/fixtures/search-eval-baseline.json` with A0.** The
   committed baseline is the contaminated `ffeaad9` corpus and is what
   `search-eval:compare` defaults to, so the default comparison is against a
   record the design note calls unusable. A0 is a clean control on a known SHA.
   Not done here because it changes the default target for every future run.
3. **Nomic** stays documented-but-unbundled. `--model <dir>` reaches it; adding
   a `nomic` id to `fetch-model.sh` needs a spec with `search_query:` /
   `search_document:` prefixes and `dim` 768.
4. **Toolchain drift.** `dtolnay/rust-toolchain@stable` means a new clippy lint
   can turn `main` red without a commit — it did, via
   `chunks_exact_to_as_chunks`. Worth a `rust-toolchain.toml` if that recurs.
