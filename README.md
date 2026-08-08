# embeddingsearchtools

A minimal, modular embedding search engine in Rust. It generates embeddings and
serves low-latency similarity search behind a clean library API, a CLI, and a
long-lived stdio daemon designed to be driven from a TypeScript `spawn`.

## Design at a glance

Three decoupled layers, so the engine plugs into different workflows:

| Layer | What it does | Swap point |
|-------|--------------|-----------|
| **Embedder** (`Embedder` trait) | text → vector | `MockEmbedder` (default) or `MiniLmEmbedder` (`--features onnx`) |
| **Index** (`Index` trait) | top-k vector search | `FlatIndex` (exact) or `HnswIndex` (approximate), selected per store |
| **Lexical** (`LexicalIndex`) | BM25 keyword search | optional, enables hybrid retrieval |
| **Store** (`store` module) | atomic, mmap-friendly persistence | raw `f32` matrix + JSON manifest |

`Database` composes the three into the primary API: **index, query, update**.

### Embedding backend

- **Default build** uses `MockEmbedder` — a deterministic, dependency-free
  token-hashing embedder. It is not semantically meaningful, but it exercises the
  entire pipeline (indexing, querying, persistence, daemon, TS client) without any
  model download, and every test runs against it.
- **`--features onnx`** uses real **`all-MiniLM-L6-v2`** (384-d, mean-pooled,
  L2-normalized) via ONNX Runtime, with the int8-quantized weights **bundled into
  the binary** (`include_bytes!`). See [Bundling the model](#bundling-the-model).

The backend is chosen at build time; nothing else in the code changes.

### Similarity metrics

Configurable per store: `cosine` (default, vectors stored normalized), `dot`, or
`euclidean` (returned as negated distance so higher always means more similar).

The metric is fixed when a store is created; opening an existing store with a
conflicting `--metric` prints a warning to stderr and keeps the stored metric.

Under `cosine` and `dot`, hits scoring `<= 0` (orthogonal or opposed vectors)
are excluded from results instead of padding top-k with noise, so a query may
return fewer than `k` hits. This mirrors the filtering the TS client already
does client-side, as defense for other callers. `euclidean` scores are negated
distances — legitimately negative — and are never filtered.

### Index backend: exact vs approximate

The search backend is chosen per store and fixed at creation, exactly like the
metric (a conflicting `--index` on an existing store warns and is ignored):

- **`flat`** (default) — `FlatIndex`, exact brute-force. Every query scans every
  live vector. Simple, always correct, and fast to a few hundred thousand
  vectors.
- **`hnsw`** — `HnswIndex`, an approximate **Hierarchical Navigable Small World**
  graph. Queries walk a layered proximity graph in roughly `O(log n)` instead of
  `O(n)`, trading a little recall for a large speedup at scale.

Both backends share the same storage, so results are directly comparable: HNSW
returns the same score for a hit that `flat` would, and applies the same
non-positive-score filtering. Recall against exact search is high (>0.9 @k=10 in
the test harness); the diversity heuristic used when building the graph keeps
even orthogonal outliers reachable.

**The graph is never written to disk.** Only the vectors are persisted (identical
on-disk format for both backends); an HNSW store rebuilds its graph from those
vectors when opened. That keeps the format simple and mmap-friendly, but means a
one-shot CLI command against a large HNSW store re-pays graph construction each
time. The intended path for HNSW is the long-lived `serve` daemon, which builds
the graph **once** at startup and keeps it hot — the same reason the daemon
exists for model loading.

### Hybrid search (vector + BM25)

Dense vectors match on *meaning* but can miss exact terms — rare identifiers,
codes, names — that a smoothed embedding washes out. A store created with
`--hybrid` keeps an Okapi BM25 lexical index (`LexicalIndex`) alongside the
vectors; `query --hybrid` runs both retrievers and fuses their rankings with
**Reciprocal Rank Fusion** (combine by rank, not score, so the incomparable
cosine and BM25 scales need no normalization). Hybrid mode is fixed at creation
and preserved across reopens, like the metric and backend.

Only the texts are persisted (`texts.json`); the BM25 postings, like the HNSW
graph, are rebuilt on load. Fusion cost is dominated by the vector search — the
lexical side is a cheap postings walk — so a hybrid query costs about the same as
a vector query plus a small constant.

**Retrieving the legs separately.** `--hybrid` fuses inside the engine and
returns one blended ranking, which is convenient but lossy: the per-retriever
ranks and scores are gone and the RRF constant is not the caller's to choose.
A caller doing its own fusion — n-way with a third retriever, a different
constant, or wanting per-source diagnostics — can ask for the keyword leg on
its own with `query --lexical` (CLI) or `"retriever":"lexical"` (daemon).
Scores are then raw BM25 sums rather than RRF scores, and no embedding is
computed, so it is just a postings walk. Documents sharing no query term are
not returned, so a lexical query may yield fewer than `k` hits rather than
padding with noise.

### Efficient updates

Both backends support `add` / `update` / `upsert` / `remove` without a full
rebuild. Deletes are tombstoned for O(1) removal and stable rows; `compact`
reclaims them (and rebuilds the HNSW graph over the survivors). The lexical index
tracks the same mutations. Persistence writes each file atomically (temp +
rename) so a crash mid-save can't corrupt an existing store.

### Building the `onnx` feature behind a proxy

`ort` fetches a prebuilt ONNX Runtime from `cdn.pyke.io` during its build
script. On a network where that host is unreachable the build fails with
`Failed to GET https://cdn.pyke.io/...` before compiling anything.

Point it at a runtime you fetch yourself instead — Microsoft publishes the
same builds on GitHub Releases:

```bash
curl -fL -o ort.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf ort.tgz
export ORT_LIB_LOCATION="$PWD/onnxruntime-linux-x64-1.22.0"
cargo build --features onnx
```

Match the ONNX Runtime version to what the pinned `ort` expects (2.0.0-rc.10
wants 1.22.x); a mismatch fails at link time rather than silently.

This only covers the runtime. The `onnx` feature also needs the model
weights, which `scripts/fetch-model.sh` pulls from Hugging Face — see
[Bundling the model](#bundling-the-model). Without them the build still
succeeds (build.rs writes empty placeholders) and fails at
session-build time, so a clean build plus a runtime error pointing at the
model is the expected state when only the weights are missing.
### Cross-encoder reranking

Retrieval and ranking are different problems. A bi-encoder embeds query and
passage separately, so their vectors never see each other; a cross-encoder
runs the pair through one model together, letting every query token attend to
every passage token. That is more informative and far more expensive — it
cannot be precomputed or indexed — so it only pays over a shortlist something
cheaper already produced.

`{"op":"rerank","query":"...","passages":[{"id","text"}],"k":10}` scores the
pairs with `ms-marco-MiniLM-L-6-v2` and returns the best `k`, best first, as
`reranked: [{id, score}]`. Scores are raw logits: higher is more relevant, but
they are not probabilities and are comparable only within one call.

Passages are supplied inline rather than looked up by id. The caller has the
exact spans it intends to show, and those are what should be scored — the
stored chunk text may be a different span, and a non-hybrid store keeps no
texts at all.

**Released binaries do not carry the reranker.** It is a second ~23 MB model
on top of the embedder's, and bundling it took the extracted binary from 26 MB
to 73 MB. That price bought a reranker which, measured against a caller's own
deterministic lexical scorer over a 62-query code-retrieval set, was
significantly *worse* on Recall@1 in every configuration (p ≤ 0.05) and lost
on every query class but conceptual — a passage ranker trained on
natural-language queries is out of distribution on an identifier, an error
string or a filename. So the default release ships lean and `rerank` returns
an error naming the two ways to enable it:

- `scripts/fetch-model.sh --with-reranker` then `cargo build --features onnx`,
  which bundles the weights as before; or
- `embsearch serve --reranker-model <dir>`, pointing at a directory holding
  `model.onnx` + `tokenizer.json`, which needs no rebuild.

Either way it requires an `onnx` build; the default (mock) build returns an
error rather than ranking on something that is not a cross-encoder.

## Footprint

- Default (mock) release binary: **~1.1 MB**, tiny dependency tree.
- With `--features onnx` + bundled int8 MiniLM: **~35–45 MB total** (≈23 MB model
  + ≈10–15 MB ONNX Runtime + binary). *Note:* the original 10–15 MB target is only
  reachable with static-embedding models; MiniLM was chosen for accuracy, which
  moves the realistic budget to ~40 MB.
- Adding the cross-encoder reranker weights (`fetch-model.sh --with-reranker`)
  roughly doubles that again. Released binaries do not include them — see
  [Cross-encoder reranking](#cross-encoder-reranking).

## Performance

A dependency-free harness measures the scoring kernels and both index backends:

```bash
cargo run --release -p embsearch-core --example perf          # 10k vectors
cargo run --release -p embsearch-core --example perf -- 40000 # larger
```

**Scoring kernels.** The hot path (dot / squared-euclidean over `dim`-length
vectors) uses multi-accumulator loops that LLVM auto-vectorizes — no `unsafe`, no
`std::simd`, no external crate. Measured **~3x** throughput over the naive
single-accumulator reduction at `dim=384`.

**Flat vs HNSW.** `FlatIndex` is exact — recall is always 100%. `HnswIndex` is
approximate: it answers in `O(log n)` instead of scanning every vector, so its
speed edge widens with scale, and `ef_search` trades recall for latency. The
default `ef_search=200` **favors accuracy** (`HnswIndex::set_ef_search` tunes it
at runtime). On 10k clustered vectors (`dim=384`, cosine), against the exact scan:

| `ef_search` | recall@10 | speedup vs exact |
|------------:|----------:|-----------------:|
| 16          | ~46%      | ~13x             |
| 64          | ~74%      | ~8x              |
| 128         | ~87%      | ~4x              |
| **200 (default)** | **~94%** | **~2.5x**  |
| 256         | ~98%      | ~2x              |

Recall falls as the dataset grows at a *fixed* `ef` (a constant candidate list
covers a smaller fraction of a larger graph): the default reaches ~94% at 10k and
~82% at 40k, so raise `ef_search` for larger corpora. Because HNSW gets
disproportionately faster at scale (~50x at 40k, low `ef`), you can afford a much
larger `ef` there and still beat the exact scan.

The tradeoff is **build time**: the flat index just appends vectors, while HNSW
builds a graph (~10³ vectors/s here, distance-bound; the accuracy-first
`ef_construction=200` is part of that cost). It is paid once — the `serve` daemon
builds the graph at startup and keeps it hot, and searches then run
allocation-free — so HNSW is for the long-lived daemon, not one-shot CLI calls.

### Accuracy summary

| Backend | Recall | When |
|---------|--------|------|
| `flat` | **exact (100%)** | correctness matters, or up to a few 100k vectors |
| `hnsw` | **~94% default, tunable to ~98%+** | large corpora where query latency dominates |
| `--hybrid` | improves *relevance* (not recall vs vectors) | queries with exact terms embeddings blur |

The default is `flat` — exact — so you opt into approximation deliberately. When
you do, the harness above measures the actual recall for your data and settings.

## CLI

```bash
# Build (default mock backend)
cargo build --release          # -> target/release/embsearch
# Or with real MiniLM:
cargo build --release --features onnx

# Bulk-index a JSONL file of {"id","text"} records ("-" reads stdin)
embsearch index --path ./store --input docs.jsonl
# ...or build an approximate HNSW store for scale (backend fixed at creation)
embsearch index --path ./store --index hnsw --input docs.jsonl

# Query
embsearch query --path ./store "how do vector databases work" -k 5
embsearch query --path ./store "..." -k 5 --json

# Hybrid: build with a BM25 keyword index, then fuse vector + keyword results
embsearch index --path ./hstore --hybrid --input docs.jsonl
embsearch query --path ./hstore --hybrid "kubernetes ingress" -k 5
# ...or take the keyword leg alone (raw BM25 scores), to fuse it yourself
embsearch query --path ./hstore --lexical "kubernetes ingress" -k 5

# Single-record mutations
embsearch add    --path ./store --id doc42 --text "some text"
embsearch update --path ./store --id doc42 --text "new text"
embsearch remove --path ./store --id doc42

# Long-lived NDJSON daemon (low-latency path)
embsearch serve  --path ./store
```

Flags: `--metric cosine|dot|euclidean`, `--index flat|hnsw`, and `--hybrid` (all
fixed when a store is created; an existing store keeps its own — `--hybrid` also
selects fused ranking at query time), `--lexical` (query only: keyword-only BM25
ranking; errors on a non-hybrid store rather than silently downgrading), `--model <dir>` (override bundled weights
with an on-disk `model.onnx` + `tokenizer.json`, onnx build only).

## Daemon protocol (NDJSON)

`serve` loads the model + index **once** and answers one JSON object per line on
stdin, one response per line on stdout. This is the low-latency path — a per-call
`spawn` would re-pay model load every query.

Requests:

```json
{"op":"query","text":"...","k":5}
{"op":"query","vector":[/* dim floats */],"k":5}
{"op":"query","text":"...","k":5,"retriever":"dense"}
{"op":"query","text":"...","k":5,"retriever":"lexical"}
{"op":"query","text":"...","k":5,"retriever":"hybrid"}
{"op":"add","id":"x","text":"..."}
{"op":"update","id":"x","text":"..."}
{"op":"upsert","id":"x","text":"..."}
{"op":"remove","id":"x"}
{"op":"bulk","items":[{"id":"a","text":"..."},{"id":"b","text":"..."}]}
{"op":"save"}
{"op":"compact"}
{"op":"count"}
{"op":"info"}
{"op":"ping"}
```

`query` takes `text` **or** `vector`, not both — sending both is an error.
Responses always carry `ok`: `{"ok":true,"results":[{"id","score"}]}` or
`{"ok":false,"error":"..."}`. `bulk` embeds the whole batch in one inference
(the fast path for bulk indexing) and answers
`{"ok":true,"inserted_count":N,"updated_count":M}`; `info` answers
`{"ok":true,"model_id":"...","dim":384,"count":N,"index":"flat|hnsw","hybrid":bool,"rerank":bool}`
so clients can verify the backend (e.g. reject the non-semantic mock build, or
confirm the index type) before indexing. Two of those fields answer questions
the binary's version cannot: `hybrid` is fixed when a store is created, so a
client cannot infer it from the daemon it launched, and `rerank` reports
whether cross-encoder weights are actually loadable — released binaries carry
the `rerank` op without bundling the weights, so a version check alone would
advertise a reranker that fails on first call. `compact` reclaims rows
tombstoned by `remove`. Adding `"hybrid":true` to a `query` on a hybrid store
fuses vector and BM25 results (`text` only — a precomputed `vector` can't be tokenized).

## TypeScript usage

A zero-dependency client is in [`ts/client.ts`](ts/client.ts):

```ts
import { EmbSearchClient } from "./ts/client";

const client = new EmbSearchClient({
  binaryPath: "./target/release/embsearch",
  storePath: "./store",
  metric: "cosine",
});
await client.ready();

await client.add("doc1", "the quick brown fox");
await client.upsert("doc2", "machine learning embeddings");

const hits = await client.query("fast animal", 5); // [{ id, score }, ...]
await client.save();
await client.close();
```

## Rust library usage

```rust
use embsearch_core::{Database, Metric, MockEmbedder};

let mut db = Database::new(MockEmbedder::new(384), Metric::Cosine);
db.add("a", "the quick brown fox")?;
db.add("b", "a lazy sleeping dog")?;
let hits = db.query("quick fox", 1)?;   // -> [SearchResult { id: "a", score }]
db.save("./store")?;
```

## Bundling the model

The `onnx` build compiles the weights straight into the binary from
`crates/core/models/`. The committed placeholders are empty; fetch the real
weights and rebuild:

```bash
scripts/fetch-model.sh                 # downloads MiniLM int8 ONNX + tokenizer
cargo build --release --features onnx  # self-contained MiniLM binary
```

See [`crates/core/models/README.md`](crates/core/models/README.md) for details.

## Finishing setup end-to-end

The CI/release workflows and crates.io publishing need a login with GitHub's
`workflow` scope and Hugging Face access. The complete, self-contained runbook —
bundling the model, installing the workflows (from [`docs/workflows/`](docs/workflows/)),
publishing crates, and cutting a release — is in **[`docs/SETUP.md`](docs/SETUP.md)**.

## Development

```bash
cargo test                 # full suite against the mock backend
cargo build --release      # default binary
```

```bash
cargo test                 # full suite against the mock backend
cargo build --release      # default binary
```
