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

### Efficient updates

Both backends support `add` / `update` / `upsert` / `remove` without a full
rebuild. Deletes are tombstoned for O(1) removal and stable rows; `compact`
reclaims them (and rebuilds the HNSW graph over the survivors). Persistence
writes each file atomically (temp + rename) so a crash mid-save can't corrupt an
existing store.

## Footprint

- Default (mock) release binary: **~1.1 MB**, tiny dependency tree.
- With `--features onnx` + bundled int8 MiniLM: **~35–45 MB total** (≈23 MB model
  + ≈10–15 MB ONNX Runtime + binary). *Note:* the original 10–15 MB target is only
  reachable with static-embedding models; MiniLM was chosen for accuracy, which
  moves the realistic budget to ~40 MB.

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

# Single-record mutations
embsearch add    --path ./store --id doc42 --text "some text"
embsearch update --path ./store --id doc42 --text "new text"
embsearch remove --path ./store --id doc42

# Long-lived NDJSON daemon (low-latency path)
embsearch serve  --path ./store
```

Flags: `--metric cosine|dot|euclidean` and `--index flat|hnsw` (both used only
when creating a store; an existing store keeps its own), `--model <dir>`
(override bundled weights with an on-disk `model.onnx` + `tokenizer.json`, onnx
build only).

## Daemon protocol (NDJSON)

`serve` loads the model + index **once** and answers one JSON object per line on
stdin, one response per line on stdout. This is the low-latency path — a per-call
`spawn` would re-pay model load every query.

Requests:

```json
{"op":"query","text":"...","k":5}
{"op":"query","vector":[/* dim floats */],"k":5}
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
`{"ok":true,"model_id":"...","dim":384,"count":N,"index":"flat|hnsw"}` so clients
can verify the backend (e.g. reject the non-semantic mock build, or confirm the
index type) before indexing; `compact` reclaims rows tombstoned by `remove`.

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
