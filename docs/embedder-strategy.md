# Design Note: Embedder strategy — model spec, asymmetric queries, and what to bundle

Status: **still unmeasured; the groundwork is done.** Every rate quoted from
`hoocode` below is a past measurement of the *current* embedder — no model has
been compared to another yet. What has landed is what makes the comparison
runnable: a `ModelSpec` so a model is data rather than code, and fixes for the
two migration bugs that would have made any result unshippable.
[`eval-runbook.md`](eval-runbook.md) is how to run it, and what is verified
versus merely written.

Companion document: `hoocode/docs/hybrid-retrieval-design.md`, which owns the
retrieval pipeline, the gold set, and the eval harness. That document defers
"any Rust-side changes" here, and its last open finding is the question this
note answers.

## TL;DR

- The consumer's eval says the remaining headroom is **entirely in the dense
  leg**. BM25 has the lexical classes saturated; the semantic classes sit at
  MRR 0.041–0.250.
- Ship **`bge-small-en-v1.5` int8** as the bundled default, replacing
  `all-MiniLM-L6-v2`. Same 384 dims (no store format change), ~+10 MB download,
  512-token window instead of 256.
- **`nomic-embed-text-v1.5` is not a bundling candidate** — ~140 MB of weights
  against a 26 MB binary the consumer downloads on demand — and its one real
  advantage (8,192 tokens) is worth nothing in a pipeline that chunks at 60
  lines. Keep it reachable via `--model <dir>`, documented, unbundled.
- The swap is **not a weights swap**. The old `MiniLmEmbedder` hardcoded mean
  pooling and hardcoded `model_id`; BGE uses CLS pooling. Dropping BGE weights
  into `crates/core/models/` would have produced wrong vectors *and* a manifest
  still claiming MiniLM, defeating the consumer's index invalidation. The
  `ModelSpec` that fixes this was a prerequisite, not a refactor — it shipped
  in v0.3.2.
- Two bugs found while checking the migration path, both of which brick or
  corrupt an existing index on a model change — now **fixed**, and not a moment
  early: v0.3.2 shipped the `model_id` change that arms the first one. See
  [Migration was broken](#migration-was-broken--both-bugs-are-now-fixed).
- The 512-token window unlocks a chunk-size experiment the consumer cannot
  currently run. It may matter more than the model's own retrieval delta.

---

## Where the headroom is

`hoocode` measures retrieval over itself with a 62-query / 68-span gold set
across 28 files and 16 subsystems, run against a git worktree pinned to an
explicit SHA, with provenance capture and a paired sign test. This is a real
eval, and it is unusually specific about where the current system fails.

Recall@10 by query class on the pre-decontamination baseline
(corpus `ffeaad9`, embsearch 0.2.0, `all-MiniLM-L6-v2-int8`, 17,179 chunks):

| class | n | lexical | semantic | bm25+dense +rr |
|---|---|---|---|---|
| exact-symbol | 22 | 73% | 68% | **100%** |
| error-fragment | 10 | 100% | 50% | **100%** |
| path | 6 | 0% | 50% | 83% |
| conceptual | 14 | 0% | 36% | **36%** |
| cross-file | 6 | 0% | 25% | **25%** |
| boundary | 4 | 0% | 0% | **25%** |

Per-class MRR after decontamination, on the shipped `auto +rr` path:
exact-symbol 0.888, error-fragment 0.950, path 0.667 — against conceptual
**0.133**, cross-file **0.041**, boundary **0.250**.

Two things follow.

**The lexical leg is done.** BM25 puts the answer in the top 10 for every
exact-symbol and error-fragment query. There is nothing left to win there and
a model change must not lose it — see the guardrail in the protocol below.

**24 of 62 queries are carried by an embedder that is barely functioning on
them.** conceptual, cross-file and boundary are precisely the classes where a
bi-encoder is the only retriever that can help, and they are where the system
scores worst. That is what a better embedder is for, and it is why this
experiment is worth running rather than assuming.

The consumer's own conclusion, after ruling out chunk size and chunk metadata
(`hybrid-retrieval-design.md:853`):

> the bi-encoder simply does not place it near this query. That points at the
> embedder — a 384-dim MiniLM asked to relate "which protocol operation
> reports…" to a method signature — rather than at anything the retrieval
> pipeline controls. Testing it means swapping the embedding model, which is
> the first open question here that neither fusion, reranking, nor chunking
> can reach.

## Candidates

Public MTEB retrieval scores (NDCG@10), listed to size the bet, **not** to
decide it. The 62-query domain eval decides it; general-domain leaderboard
averages predict code search poorly and are used here only to rule things in
and out before spending a measurement.

| model | params | dim | ctx | pooling | query prefix | int8 weights | MTEB retr. |
|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` *(today)* | 22.7M | 384 | 256 eff. | mean | none | ~23 MB | ~41.9 |
| `bge-small-en-v1.5` | 33.4M | 384 | 512 | **CLS** | optional instruction | ~33 MB | ~51.7 |
| `gte-small` | 33.4M | 384 | 512 | mean | none | ~33 MB | ~49.5 |
| `e5-small-v2` | 33.4M | 384 | 512 | mean | **required** (`query:`/`passage:`) | ~33 MB | ~49.0 |
| `nomic-embed-text-v1.5` | 137M | 768 (MRL) | 8192 | mean | **required** (`search_query:`/`search_document:`) | ~137 MB | ~53.0 |

The three small models are all 384-dim and 512-token, so any of them is a
drop-in for the store format and none forces a `dim` migration. They differ on
exactly the two axes the current code hardcodes — pooling and prefixes — which
is why the `ModelSpec` below is worth building even though we intend to ship
one of them.

`bge-small-en-v1.5` is the primary candidate: best public retrieval of the
three, and the ~10-point gap to MiniLM is large enough that it should be
visible at n=62 if it transfers to code at all.

### Why Nomic is out as a default

Not close, on this consumer's constraints:

1. **Download size.** `hoocode` fetches a per-platform release archive on
   demand the first time semantic search is used
   (`packages/coding-agent/src/utils/tools-manager.ts:129`). Today that is a
   ~26 MB extracted binary. Nomic's weights alone are ~137 MB int8. The
   cross-encoder reranker was already rejected for tripling the download at
   ~23 MB (`scripts/fetch-model.sh:12`); this is four times that cost, for a
   terminal coding agent.
2. **The 8,192-token window buys nothing here.** The consumer chunks at 60
   lines / 1,000 chars and returns chunks, not documents. Its eval established
   that *reach at fixed top-k* is the binding constraint (R@50 85% against
   R@10 72%), and that shrinking chunks costs reach because "a fixed top-k over
   smaller chunks retrieves less content." Context length is not the lever;
   content-per-retrieved-chunk is, and that is capped by k and chunk size, not
   by 8,192.
3. **Quadratic cost at long sequences.** 137M parameters over long inputs on a
   single CPU core is not the "3–4x MiniLM" that parameter counts suggest.

None of this makes Nomic a bad model. It makes it the wrong *bundled* model
for this consumer. It stays reachable through `--model <dir>` for anyone
indexing long documents rather than code, and this note commits to documenting
that path rather than pretending it does not exist.

---

## Migration was broken — both bugs are now fixed

Both were live bugs, independent of which model we pick, and either one turned
a model upgrade into a support incident. Recorded here with their fixes,
because the reasoning is what stops them coming back.

**They stopped being hypothetical.** `model_id` now carries a spec hash, so
v0.3.2 reports `all-MiniLM-L6-v2-int8.1f635576` where v0.3.1 reported
`all-MiniLM-L6-v2-int8` — verified by running both binaries. That is Bug A's
exact trigger, shipped. Nobody broke on release day only because `ensureTool`
returns early when the binary already exists (`tools-manager.ts:467`), so
existing users kept v0.3.1 alongside their v0.3.1 store. It would have fired on
the next binary refresh.

### Bug A — a model change bricks the consumer's index instead of rebuilding it

`Database::open` refuses a store whose manifest `model_id` disagrees with the
embedder (`crates/core/src/lib.rs:166`), and `serve` reaches it through
`open_or_create_hybrid` → `open` (`crates/cli/src/main.rs:281`). That refusal is
correct: querying MiniLM vectors with BGE vectors is silent nonsense.

The consumer, however, discovers the model id *from the daemon it just
started*. `EmbsearchService.start` calls `openClient()` (spawn + `info`) and
only then consults the sidecar
(`packages/coding-agent/src/core/embsearch/embsearch-service.ts:305,328`). On a
model upgrade the daemon fails to open the existing store, `client.ready()`
rejects, and the service lands in `unavailable` — permanently. The comment on
line 327 ("Missing/stale sidecar (format, chunker, or model changed) → clean
rebuild") describes an intent the code cannot carry out: `emptyIndexMeta` resets
the *sidecar* only and never removes the store directory, and the line is
unreachable on a model change anyway.

The adjacent hybrid-mismatch branch (line 316) does the right thing — close,
`rmSync(storeDir)`, reopen. The model path needs the same treatment, and it
needs the model id *before* spawning against the store.

**Fixed (this repo):** `embsearch store-info --path <dir> [--json]` reads
`manifest.json` without constructing an embedder:

```
$ embsearch store-info --path ./store --json
{"format_version":1,"model_id":"all-MiniLM-L6-v2-int8","dim":384,
 "metric":"cosine","index":"flat","hybrid":false,"rows":3,"live":3}
```

It answers for a store no current binary can open, which is exactly the case
that matters — verified by reading a v0.3.1 store with a binary that refuses to
open it. A daemon-side `info` cannot do this: reaching `info` already requires
a successful open.

**Fixed (consumer):** `EmbsearchService.start` catches the daemon's refusal,
confirms via `store-info` that a readable store is what it choked on, then
wipes and reopens. Only a store that is *present and readable* is discarded —
if `store-info` cannot read it either, the problem is not a model mismatch and
the original failure stands, because destroying an index would be the wrong
answer to an unknown fault.

Rejected alternative: making `serve` auto-wipe a mismatched store. Deleting a
user's index as a side effect of starting a daemon is not a decision that layer
gets to make silently — the consumer knows whether it can rebuild, and the
library does not.

### Bug B — a chunker version bump leaks orphaned vectors

When only `CHUNKER_VERSION` changes, the daemon opens fine (the Rust side knows
nothing about chunking), `loadIndexMeta` returns `undefined`, and the service
continues with `emptyIndexMeta` — `meta.files` is now empty. In
`indexChangedFiles`, `oldChunkCount` is read from that empty map and is `0` for
every file, so the "remove old chunks beyond the new count" loop never runs
(`embsearch-service.ts:397`), and `toRemove` is empty for the same reason. Any
file that chunks into *fewer* pieces under the new strategy leaves its tail
vectors (`path#N`, `path#N+1`, …) in the store, holding text that no longer
exists anywhere, retrievable forever.

The store is upserted by id, so this is invisible in chunk counts and invisible
in the sidecar. It surfaces as occasional stale hits.

**Fixed:** a stale sidecar over a non-empty store now takes the store with it,
through the same `rebuildFrom` path as Bug A and the pre-existing hybrid
mismatch — three symptoms, one recovery. Resetting the sidecar alone was never
enough, because the sidecar is the only record of how many chunks each file
produced.

This matters here because the plan deliberately bumps `CHUNKER_VERSION`
alongside the model (see below), which is precisely the combination that
triggers it.

### Why the bugs come first

The migration has to be right before the model changes, not after, because both
failures are *silent to the person who caused them*: Bug A looks like "semantic
search stopped working on my machine", Bug B looks like nothing at all.

---

## Design: `ModelSpec`

### The hazard being designed against

`MiniLmEmbedder` hardcodes three things that belong to the weights, not to the
code: the pooling strategy (`crates/core/src/embed.rs:252`, mean over the
attention mask), the identity string (`crates/core/src/embed.rs:199`,
`"all-MiniLM-L6-v2-int8"`), and the absence of any query/document asymmetry.

So today, dropping `bge-small-en-v1.5` into `crates/core/models/` and rebuilding
produces a binary that:

- reads a CLS-pooled model with mean pooling — degraded vectors, no error;
- reports `model_id: "all-MiniLM-L6-v2-int8"` — so the consumer's invalidation
  check (`index-meta.ts:82`) passes, and a store built from MiniLM vectors keeps
  being queried with BGE vectors;
- never truncates, because the model's token limit lives in a comment.

That last one is a live defect, though a quieter one than it first looked.
`encode_batch` is called with no truncation configured, so a long input is
embedded over its full length. Measured against the released v0.3.1 binary, a
~4,800-token document does **not** fail — the ONNX export accepts it — it
simply returns a vector pooled over far more tokens than the model was trained
on, and a measurably different one (cosine 0.230 against a fixed query, versus
0.192 once truncated to the reference pipeline's 256). No error, no warning,
just a worse vector. The consumer's 1,000-char chunk cap is the only thing
currently bounding it, which means the library's output is correct by a
coincidence of its caller's configuration.

### Shape

```rust
/// Everything about a bi-encoder that changes the vectors it produces.
pub struct ModelSpec {
    /// Written to the store manifest; the only cross-process invalidation key.
    pub model_id: String,
    pub dim: usize,
    pub pooling: Pooling,
    pub normalize: bool,
    /// Enforced via tokenizer truncation, not documented in a comment.
    pub max_tokens: usize,
    /// Prepended before embedding a query. `None` for symmetric models.
    pub query_prefix: Option<String>,
    /// Prepended before embedding a document.
    pub document_prefix: Option<String>,
}

pub enum Pooling { Mean, Cls }
```

Loaded from a `model.json` sidecar beside `model.onnx` / `tokenizer.json`, with
the bundled model's spec compiled in. `from_dir` **errors when `model.json` is
absent** rather than assuming MiniLM's shape — guessing is the bug we are
fixing, and a `--model <dir>` pointed at Nomic must not silently mean-pool with
no prefixes.

`MiniLmEmbedder` becomes `OnnxEmbedder` (with a deprecated re-export for one
release). The name was always wrong: the struct never had anything
MiniLM-specific in it except the constants that are now data.

### `model_id` must change whenever the vectors change

The consumer invalidates on `model_id` alone — it compares
`meta.modelId !== modelId` and nothing else. So `model_id` cannot be a friendly
label that stays put while pooling or a prefix changes underneath it; every
field of `ModelSpec` feeds the vectors, and a change to any of them must
invalidate.

Proposal: `model_id = "<name>.<8 hex of spec hash>"`, e.g.
`bge-small-en-v1.5-int8.a3f19c2b`. Human-readable prefix for logs and the
`info` op, machine-safe suffix that makes a silent spec drift impossible.

Rejected: adding a separate `spec_hash` field to the manifest. It would fix the
Rust-side check while leaving the consumer — which reads only `model_id` —
exposed. The invalidation key has to be the thing consumers already compare.

### Asymmetric embedding

`e5` requires prefixes, `nomic` requires prefixes, `bge` benefits from a query
instruction. The trait is symmetric today, so none of that is expressible.

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;                       // document
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {                // query
        self.embed(text)
    }
    // embed_batch / embed_query_batch likewise
}
```

The default implementation keeps every symmetric model and `MockEmbedder`
correct with no changes. `Database::query` and `query_hybrid` call
`embed_query`; `add`, `update` and the bulk path call `embed`. That is the
whole change — three call sites and a defaulted method.

Note the asymmetry is a property of the *store*, not of a call: a store whose
documents were embedded with `search_document:` must be queried with
`search_query:` forever. That is exactly what folding the spec hash into
`model_id` guarantees.

---

## The chunk-window experiment this unlocks

`hoocode`'s chunker caps at 1,000 characters with the comment "MiniLM truncates
~256 tokens ≈ 1000 chars" (`chunker.ts:31`). That cap is a **model constraint
wearing the costume of a tuning constant**.

The consumer already measured the direction. Halving chunks to 30 lines /
500 chars: Recall@50 fell 79% → 69% and the boundary class went from 2-of-4
findable to 0-of-4, with the mechanism stated plainly — *a fixed top-k over
smaller chunks retrieves less content.* Recall@1 ticked up 53% → 55%, which is
sharpening, and it did not pay for the loss.

Symmetrically, at fixed k, **larger chunks retrieve more content**, and reach
is the binding constraint (R@50 85% vs R@10 72%). A 512-token model permits a
~2,000-char cap. That experiment is not currently runnable at all.

It is not a free win, and this note does not assume it is one. The opposing
force is the dilution the consumer documented for `boundary-daemon-info`: one
vector covering six methods, where the answer's own doc comment paraphrases the
query and still is not retrieved. Bigger chunks make that worse. Reach and
dilution pull in opposite directions and the eval is the only thing that can
say which wins — which is the point of running it rather than arguing it.

Bundling the chunker change with the model change is deliberate: both force a
full re-index, and the consumer has a chunker bug parked precisely for this
occasion (`hybrid-retrieval-design.md:846` — a blank-line range bug affecting
31% of chunks, measured at 0 better / 0 worse, "should ride along the next time
something else earns the rebuild"). Users should pay one rebuild, not three.

The cost of bundling is attribution: a single run cannot separate the model
from the window. The protocol below buys attribution back by measuring the
model at the old cap first.

---

## Experiment protocol

### The control has to be re-measured

The committed run record
(`packages/coding-agent/test/fixtures/search-eval-baseline.json`) is **not
usable as the control**: it is the contaminated corpus `ffeaad9` on embsearch
0.2.0 with `corpusExcluded` absent, i.e. taken before the gold fixture was
removed from the corpus it searches. The decontaminated numbers exist only as
prose in the design note. Arm A0 re-establishes the control on the same binary,
corpus and exclusions as every other arm.

### Arms

Everything not named is held fixed: corpus SHA, `CORPUS_EXCLUSIONS`, fusion k,
rerank weights, index kind, retriever config.

| arm | model | pooling | query prefix | chunk cap | purpose |
|---|---|---|---|---|---|
| A0 | MiniLM int8 | mean | — | 1000 | control, re-measured |
| A1 | bge-small-en-v1.5 int8 | CLS | none | 1000 | model effect, isolated |
| A2 | bge-small-en-v1.5 int8 | CLS | instruction | 1000 | does the prefix earn its place |
| A3 | winner of A1/A2 | — | — | 2000 | the window the model unlocks |
| A4 | winner of A1/A2 | — | — | 1500 | brackets A3 if A3 moves |
| N1 | nomic-embed-text-v1.5 | mean | `search_query:` | 2000 | price the ceiling, not a candidate |

A0–A2 share a chunker version, so `retrievalSourceHash` is constant across them
and the records are directly comparable. A3/A4 change it, which the provenance
makes visible rather than leaving to memory.

N1 runs once via `--model <dir>` at 768 dims. It is not a shipping candidate;
it tells us how much of the remaining gap is the model class rather than this
particular small model, which decides whether the next question is "a bigger
embedder" or "a cross-encoder".

### Endpoints, pre-registered

- **Primary:** MRR on the shipped default path (`auto +rr`), paired sign test
  over all 62 queries, p ≤ 0.05.
- **Secondary:** R@10 and MRR on the 24-query semantic subgroup (conceptual +
  cross-file + boundary). This is the subgroup the change targets, declared in
  advance so it is not a post-hoc slice.
- **Guardrail:** exact-symbol and error-fragment must not regress. Both are at
  100% R@10 with MRR 0.888 / 0.950 and are carried by BM25; a dense change that
  costs them is a net loss regardless of what it wins.
- **Reach:** R@50, which is what the chunk-window arms are actually about.

Honest statement of power: at n=4 (boundary) and n=6 (cross-file) nothing is
significant, ever. Those classes are reported descriptively. The 24-query
subgroup is the smallest unit any claim will be made about.

### What would falsify the premise

If A1/A2 do not beat A0 on the semantic subgroup, the embedder hypothesis in
`hybrid-retrieval-design.md:853` is wrong: the remaining failures are not the
model, and the next lever is the cross-encoder or the reranker, not a bigger
bi-encoder. That is a useful result and this note commits to reporting it as
one rather than reaching for a third model.

### Harness changes needed first

Two, both small, both in the consumer:

1. **`EmbSearchClient` cannot pass `--model`.** It builds
   `["serve", "--path", …]` plus metric and hybrid flags
   (`client.ts:107`). The Rust side already accepts `--model <dir>` on `serve`
   (it is in the flattened `StoreArgs`), so threading a `modelDir` option
   through `EmbSearchClientOptions` → `EmbsearchServiceOptions` makes every arm
   a flag change instead of a rebuild.
2. **Provenance records the binary, not the model.** `eval-harness.ts:86`
   records `binaryPath` and `binaryVersion` on the stated reasoning that "the
   embedding model is baked into the binary at build time, so this is the only
   thing that identifies which model produced a score." `--model <dir>` breaks
   that assumption, and two arms would be indistinguishable in the record. It
   must record `info.model_id`.

The service already supports a `storeDir` override "so a second index can exist
for the same repo without colliding with the primary one" — which is exactly
what running two model arms needs. That part is already built.

---

## Rollout

1. ~~**`ModelSpec` + `embed_query`.**~~ Shipped in v0.3.2. Note what this
   rollout got wrong: it was meant to leave `model_id` untouched so nothing
   invalidated, and the spec hash changed it anyway. Harmless in the event —
   `ensureTool` does not upgrade an installed binary — but it armed Bug A
   before Bug A was fixed, which is the wrong order.
2. ~~**`store-info` + consumer wipe-and-rebuild.**~~ Done, unreleased. Needs a
   release before any consumer can rely on it. `probeStore` degrades to the old
   behaviour against a binary too old to have the subcommand, so no explicit
   version floor is needed — unlike `MIN_LEXICAL_RETRIEVER_VERSION`, where an
   old daemon silently ignored an unknown field instead of failing.
3. **Run A0–A2.** Decide on evidence. See
   [`eval-runbook.md`](eval-runbook.md).
4. **If (3) wins: run A3/A4**, bump `CHUNKER_VERSION`, fold in the parked
   blank-line fix.
5. **Ship the model.** `model_id` changes, both ends invalidate, every user
   pays exactly one re-index. Consumer pins a version floor for the new
   `model_id`.
6. **Document Nomic as a `--model <dir>` option** with its real costs, in
   `crates/core/models/README.md`.

Step 1 is worth doing whether or not any model ever changes: it fixes a latent
truncation bug, removes a silent-corruption path, and is the difference between
"swap the weights" being a two-line change that quietly breaks things and a
supported operation.

## Explicitly not doing

- **Bundling Nomic, or any model above ~40 MB int8.** The consumer downloads
  this binary on demand.
- **Matryoshka dimension truncation.** It is a storage optimization for a model
  we are not bundling, and 384 dims is already below the 768 it would truncate
  to.
- **Changing `dim` or the store format.** All three small candidates are 384-d.
  A model that forces a `dim` change is a different, larger piece of work.
- **A model registry with runtime download.** `--model <dir>` plus a bundled
  default covers the cases we have. A downloader is the consumer's job, and it
  already has one.
- **Re-opening fusion, k, or rerank weights.** They are held fixed across every
  arm so the model is the only thing moving. They belong to the consumer's
  design note.
