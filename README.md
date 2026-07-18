# embeddingsearchtools

A minimal, modular embedding search engine in Rust. It generates embeddings and
serves low-latency similarity search behind a clean library API, a CLI, and a
long-lived stdio daemon designed to be driven from a TypeScript `spawn`.

## Design at a glance

Three decoupled layers, so the engine plugs into different workflows:

| Layer | What it does | Swap point |
|-------|--------------|-----------|
| **Embedder** (`Embedder` trait) | text → vector | `MockEmbedder` (default) or `MiniLmEmbedder` (`--features onnx`) |
| **Index** (`Index` trait) | exact top-k vector search | `FlatIndex` today; HNSW can drop in later |
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

### Efficient updates

`FlatIndex` supports `add` / `update` / `upsert` / `remove` without rebuilding.
Deletes are tombstoned for O(1) removal and stable rows; `compact` reclaims them.
Persistence writes each file atomically (temp + rename) so a crash mid-save can't
corrupt an existing store.

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

Flags: `--metric cosine|dot|euclidean` (used when creating a store),
`--model <dir>` (override bundled weights with an on-disk `model.onnx` +
`tokenizer.json`, onnx build only).

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
{"op":"save"}
{"op":"count"}
{"op":"ping"}
```

Responses always carry `ok`: `{"ok":true,"results":[{"id","score"}]}` or
`{"ok":false,"error":"..."}`.

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
`crates/core/models/`. The committed placeholders are empty; drop in the real
files and rebuild. See [`crates/core/models/README.md`](crates/core/models/README.md)
for the exact files and where to get them. (This repo's build environment blocks
Hugging Face, so the weights are supplied out-of-band rather than fetched here.)

## Development

```bash
cargo test                 # full suite against the mock backend
cargo build --release      # default binary
```
