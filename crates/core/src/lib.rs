//! # embsearch-core
//!
//! A minimal, modular embedding search engine.
//!
//! Decoupled layers:
//! - [`Embedder`] — text → vector. [`MockEmbedder`] ships for tests/no-model use;
//!   `MiniLmEmbedder` (feature `onnx`) does real `all-MiniLM-L6-v2` inference.
//! - [`Index`] — top-k vector search behind a trait, with configurable
//!   [`Metric`]: [`FlatIndex`] (exact) or [`HnswIndex`] (approximate).
//! - [`LexicalIndex`] — optional BM25 keyword search for hybrid retrieval.
//! - [`store`] — atomic, mmap-friendly on-disk persistence.
//!
//! [`Database`] composes these into the primary API: index, query, update — plus
//! [`query_hybrid`](Database::query_hybrid) when built hybrid.
//!
//! ```
//! use embsearch_core::{Database, Metric, MockEmbedder};
//!
//! let mut db = Database::new(MockEmbedder::new(64), Metric::Cosine);
//! db.add("a", "the quick brown fox").unwrap();
//! db.add("b", "a lazy sleeping dog").unwrap();
//! let hits = db.query("quick fox", 1).unwrap();
//! assert_eq!(hits[0].id, "a");
//! ```

mod embed;
mod error;
mod hnsw;
mod index;
mod lexical;
mod simd;
pub mod store;

pub use embed::{l2_normalize, Embedder, MockEmbedder};
pub use error::{Error, Result};
pub use hnsw::HnswIndex;
pub use index::{AnyIndex, FlatIndex, Index, IndexKind, Metric, SearchResult};
pub use lexical::LexicalIndex;

/// Internal kernels exposed only so the `perf` benchmark example can compare the
/// vectorized scoring loops against their scalar baselines. Not part of the
/// stable API — do not depend on this module.
#[doc(hidden)]
pub mod internals {
    pub use crate::simd::{dot, dot_scalar, sq_euclidean, sq_euclidean_scalar};
}

#[cfg(feature = "onnx")]
pub use embed::MiniLmEmbedder;

use std::collections::HashMap;
use std::path::Path;

/// The lexical (BM25) side of a hybrid store: the inverted index plus the raw
/// texts it was built from (kept so the index can be persisted and rebuilt).
struct Hybrid {
    lexical: LexicalIndex,
    texts: HashMap<String, String>,
}

impl Hybrid {
    fn new() -> Self {
        Self {
            lexical: LexicalIndex::new(),
            texts: HashMap::new(),
        }
    }

    fn upsert(&mut self, id: &str, text: &str) {
        self.lexical.add(id, text); // add() replaces an existing id
        self.texts.insert(id.to_string(), text.to_string());
    }

    fn remove(&mut self, id: &str) {
        self.lexical.remove(id);
        self.texts.remove(id);
    }
}

/// Fuse a dense (vector) and a sparse (lexical) ranking with Reciprocal Rank
/// Fusion. RRF combines lists by *rank* rather than score, so it needs no
/// score normalization between two incomparable scales (cosine vs BM25) and is
/// robustly effective. A document's fused score is `Σ 1/(RRF_K + rank)` over the
/// lists it appears in (rank is 1-based); `RRF_K = 60` is the common default.
fn rrf_fuse(dense: &[SearchResult], lexical: &[(String, f32)], k: usize) -> Vec<SearchResult> {
    const RRF_K: f32 = 60.0;
    let mut fused: HashMap<String, f32> = HashMap::new();
    for (rank, hit) in dense.iter().enumerate() {
        *fused.entry(hit.id.clone()).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f32);
    }
    for (rank, (id, _)) in lexical.iter().enumerate() {
        *fused.entry(id.clone()).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f32);
    }
    let mut hits: Vec<SearchResult> = fused
        .into_iter()
        .map(|(id, score)| SearchResult { id, score })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    hits.truncate(k);
    hits
}

/// The primary API: an [`Embedder`] paired with a vector [`AnyIndex`].
///
/// Text goes in, vectors are produced and indexed, and queries are answered by
/// embedding the query text and searching. Vectors can also be supplied directly
/// via the `*_vector` methods when embeddings are computed elsewhere.
///
/// The index backend ([`IndexKind::Flat`] for exact search, [`IndexKind::Hnsw`]
/// for approximate) is chosen when the store is first created and preserved
/// across reopens. A store may additionally be **hybrid**: it keeps a BM25
/// lexical index alongside the vectors and answers [`query_hybrid`], which
/// fuses semantic and keyword matches. Hybrid mode is likewise fixed at creation
/// and preserved across reopens.
///
/// [`query_hybrid`]: Database::query_hybrid
pub struct Database<E: Embedder> {
    embedder: E,
    index: AnyIndex,
    /// Present iff this is a hybrid store.
    hybrid: Option<Hybrid>,
}

impl<E: Embedder> Database<E> {
    /// Create an empty database using `embedder` and `metric`, backed by the
    /// exact [`FlatIndex`].
    pub fn new(embedder: E, metric: Metric) -> Self {
        Self::new_with_index(embedder, metric, IndexKind::Flat)
    }

    /// Create an empty database with an explicit index backend.
    pub fn new_with_index(embedder: E, metric: Metric, kind: IndexKind) -> Self {
        Self::new_configured(embedder, metric, kind, false)
    }

    /// Create an empty **hybrid** database: `kind` vector index plus a BM25
    /// lexical index, queryable with [`query_hybrid`](Database::query_hybrid).
    pub fn new_hybrid(embedder: E, metric: Metric, kind: IndexKind) -> Self {
        Self::new_configured(embedder, metric, kind, true)
    }

    fn new_configured(embedder: E, metric: Metric, kind: IndexKind, hybrid: bool) -> Self {
        let dim = embedder.dim();
        Self {
            embedder,
            index: AnyIndex::new(kind, dim, metric),
            hybrid: hybrid.then(Hybrid::new),
        }
    }

    /// Open an existing store from `dir`, pairing it with `embedder`.
    ///
    /// Errors if the store's dimension or recorded model id disagree with the
    /// embedder — a guard against querying a store with the wrong model.
    pub fn open(embedder: E, dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let (index, model_id, is_hybrid) = store::load(dir)?;
        if index.dim() != embedder.dim() {
            return Err(Error::Config(format!(
                "store dim {} != embedder dim {}",
                index.dim(),
                embedder.dim()
            )));
        }
        if model_id != embedder.model_id() {
            return Err(Error::Config(format!(
                "store was built with model '{}' but embedder is '{}'; vectors from \
                 different models are incompatible — re-index the store with the \
                 current embedder, or open it with the model that built it",
                model_id,
                embedder.model_id()
            )));
        }
        // Rebuild the lexical index from the persisted texts if this is a hybrid
        // store (the BM25 postings are not written to disk, only the texts).
        let hybrid = if is_hybrid {
            let texts = store::load_texts(dir)?;
            let mut lexical = LexicalIndex::new();
            for (id, text) in &texts {
                lexical.add(id, text);
            }
            Some(Hybrid { lexical, texts })
        } else {
            None
        };
        Ok(Self {
            embedder,
            index,
            hybrid,
        })
    }

    /// Open the store at `dir` if it exists, otherwise create an empty exact
    /// ([`IndexKind::Flat`]) one.
    pub fn open_or_create(embedder: E, dir: impl AsRef<Path>, metric: Metric) -> Result<Self> {
        Self::open_or_create_with(embedder, dir, metric, IndexKind::Flat)
    }

    /// Open the store at `dir` if it exists, otherwise create an empty one with
    /// the given index backend. `kind` (like `metric`) only takes effect when a
    /// new store is created; an existing store keeps the backend it was built
    /// with.
    pub fn open_or_create_with(
        embedder: E,
        dir: impl AsRef<Path>,
        metric: Metric,
        kind: IndexKind,
    ) -> Result<Self> {
        if store::exists(&dir) {
            Self::open(embedder, dir)
        } else {
            Ok(Self::new_with_index(embedder, metric, kind))
        }
    }

    /// Open the store at `dir` if it exists, otherwise create an empty **hybrid**
    /// store with the given vector backend. As with `metric`/`kind`, hybrid mode
    /// only takes effect on creation; an existing store keeps whatever it was
    /// built as.
    pub fn open_or_create_hybrid(
        embedder: E,
        dir: impl AsRef<Path>,
        metric: Metric,
        kind: IndexKind,
    ) -> Result<Self> {
        if store::exists(&dir) {
            Self::open(embedder, dir)
        } else {
            Ok(Self::new_hybrid(embedder, metric, kind))
        }
    }

    /// Persist the index (and, for a hybrid store, the texts) to `dir`.
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        store::save(
            dir,
            &self.index,
            self.embedder.model_id(),
            self.hybrid.is_some(),
        )?;
        if let Some(h) = &self.hybrid {
            store::save_texts(dir, &h.texts)?;
        }
        Ok(())
    }

    /// Whether this is a hybrid (vector + BM25) store.
    pub fn is_hybrid(&self) -> bool {
        self.hybrid.is_some()
    }

    /// Embed `text` and add it under `id`. Errors if `id` already exists, or if
    /// `id` or `text` is empty.
    pub fn add(&mut self, id: &str, text: &str) -> Result<()> {
        validate_id(id)?;
        validate_text(text)?;
        let v = self.embedder.embed(text)?;
        self.index.add(id, v)?;
        if let Some(h) = &mut self.hybrid {
            h.upsert(id, text);
        }
        Ok(())
    }

    /// Add a precomputed vector under `id`. Errors if `id` is empty.
    pub fn add_vector(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        validate_id(id)?;
        self.index.add(id, vector)
    }

    /// Re-embed `text` and replace the vector for an existing `id`. Errors if
    /// `id` or `text` is empty.
    pub fn update(&mut self, id: &str, text: &str) -> Result<()> {
        validate_id(id)?;
        validate_text(text)?;
        let v = self.embedder.embed(text)?;
        self.index.update(id, v)?;
        if let Some(h) = &mut self.hybrid {
            h.upsert(id, text);
        }
        Ok(())
    }

    /// Insert or replace `id` with the embedding of `text`. Returns `true` if it
    /// was newly inserted. Errors if `id` or `text` is empty.
    pub fn upsert(&mut self, id: &str, text: &str) -> Result<bool> {
        validate_id(id)?;
        validate_text(text)?;
        let v = self.embedder.embed(text)?;
        let inserted = self.index.upsert(id, v)?;
        if let Some(h) = &mut self.hybrid {
            h.upsert(id, text);
        }
        Ok(inserted)
    }

    /// Remove `id`. Returns `true` if it existed.
    pub fn remove(&mut self, id: &str) -> Result<bool> {
        let removed = self.index.remove(id)?;
        if let Some(h) = &mut self.hybrid {
            h.remove(id);
        }
        Ok(removed)
    }

    /// Embed each `(id, text)` and add them. Uses the embedder's batch path.
    pub fn add_batch<I, S>(&mut self, items: I) -> Result<()>
    where
        I: IntoIterator<Item = (S, S)>,
        S: Into<String>,
    {
        let (ids, texts): (Vec<String>, Vec<String>) =
            items.into_iter().map(|(i, t)| (i.into(), t.into())).unzip();
        validate_batch(&ids, &texts)?;
        let vectors = self.embedder.embed_batch(&texts)?;
        for ((id, text), v) in ids.into_iter().zip(texts).zip(vectors) {
            self.index.add(&id, v)?;
            if let Some(h) = &mut self.hybrid {
                h.upsert(&id, &text);
            }
        }
        Ok(())
    }

    /// Embed each `(id, text)` and insert-or-replace them. Uses the embedder's
    /// batch path, so this is the fast route for bulk (re)indexing. Returns
    /// `(inserted, updated)` counts.
    pub fn upsert_batch<I, S>(&mut self, items: I) -> Result<(usize, usize)>
    where
        I: IntoIterator<Item = (S, S)>,
        S: Into<String>,
    {
        let (ids, texts): (Vec<String>, Vec<String>) =
            items.into_iter().map(|(i, t)| (i.into(), t.into())).unzip();
        validate_batch(&ids, &texts)?;
        let vectors = self.embedder.embed_batch(&texts)?;
        let mut inserted = 0usize;
        let mut updated = 0usize;
        for ((id, text), v) in ids.into_iter().zip(texts).zip(vectors) {
            if self.index.upsert(&id, v)? {
                inserted += 1;
            } else {
                updated += 1;
            }
            if let Some(h) = &mut self.hybrid {
                h.upsert(&id, &text);
            }
        }
        Ok((inserted, updated))
    }

    /// Top-`k` matches for the embedding of `query`, best first.
    pub fn query(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let v = self.embedder.embed(query)?;
        self.index.query(&v, k)
    }

    /// Top-`k` matches for a precomputed query vector.
    pub fn query_vector(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.index.query(query, k)
    }

    /// Hybrid top-`k`: fuse semantic (vector) and keyword (BM25) rankings with
    /// Reciprocal Rank Fusion. Catches both meaning-level matches and exact-term
    /// matches a dense embedding can miss. Errors if the store is not hybrid.
    ///
    /// The returned `score` is the RRF fusion score (not a cosine/BM25 score);
    /// only its ordering is meaningful. Each side contributes a candidate pool of
    /// `max(k, 4·k)` before fusion so a document ranked well by one retriever
    /// isn't lost by being just outside the other's top-`k`.
    pub fn query_hybrid(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let Some(h) = &self.hybrid else {
            return Err(Error::Config(
                "query_hybrid requires a hybrid store; create it with new_hybrid / \
                 open_or_create_hybrid (or the CLI --hybrid flag)"
                    .into(),
            ));
        };
        if k == 0 {
            return Ok(vec![]);
        }
        let pool = (k * 4).max(k);
        let v = self.embedder.embed(query)?;
        let dense = self.index.query(&v, pool)?;
        let lexical = h.lexical.search(query, pool);
        Ok(rrf_fuse(&dense, &lexical, k))
    }

    /// Number of live vectors.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the database holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Reclaim space from tombstoned (deleted) rows.
    pub fn compact(&mut self) {
        self.index.compact();
        if let Some(h) = &mut self.hybrid {
            h.lexical.compact();
        }
    }

    /// Borrow the embedder.
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Borrow the underlying index.
    pub fn index(&self) -> &AnyIndex {
        &self.index
    }

    /// Which index backend this database uses.
    pub fn index_kind(&self) -> IndexKind {
        self.index.kind()
    }
}

fn validate_id(id: &str) -> Result<()> {
    if id.is_empty() {
        return Err(Error::Config("id must not be empty".into()));
    }
    Ok(())
}

/// Empty (or whitespace-only) text embeds a zero vector that can never match
/// under cosine yet permanently occupies a row, so it is rejected up front.
/// Callers chunking files should skip empty chunks themselves.
fn validate_text(text: &str) -> Result<()> {
    if text.trim().is_empty() {
        return Err(Error::Config(
            "text must not be empty: it would embed a zero vector that can never match; \
             skip empty chunks before indexing"
                .into(),
        ));
    }
    Ok(())
}

/// Validate every id/text pair before any embedding runs, so a bad record
/// fails the batch up front instead of leaving it partially applied.
fn validate_batch(ids: &[String], texts: &[String]) -> Result<()> {
    for (id, text) in ids.iter().zip(texts) {
        validate_id(id)?;
        validate_text(text)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Database<MockEmbedder> {
        Database::new(MockEmbedder::new(128), Metric::Cosine)
    }

    #[test]
    fn add_query_roundtrip() {
        let mut db = db();
        db.add("fox", "the quick brown fox jumps").unwrap();
        db.add("dog", "a quick lazy sleepy dog").unwrap();
        db.add("car", "engine wheels highway speed").unwrap();

        let hits = db.query("quick brown fox", 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "fox");
        // Scores must be sorted descending.
        assert!(hits[0].score >= hits[1].score);
    }

    #[test]
    fn duplicate_add_errors() {
        let mut db = db();
        db.add("a", "hello world").unwrap();
        assert!(matches!(db.add("a", "again"), Err(Error::DuplicateId(_))));
    }

    #[test]
    fn update_changes_vector() {
        let mut db = db();
        db.add("a", "apple banana cherry").unwrap();
        let before = db.query("apple banana cherry", 1).unwrap()[0].score;
        db.update("a", "zebra yak walrus").unwrap();
        // The updated vector may no longer match the old text at all (hits with
        // score <= 0 are filtered), so an empty result also counts as "lower".
        let after = db
            .query("apple banana cherry", 1)
            .unwrap()
            .first()
            .map_or(f32::NEG_INFINITY, |h| h.score);
        assert!(
            after < before,
            "updating to unrelated text should lower score"
        );
    }

    #[test]
    fn update_unknown_errors() {
        let mut db = db();
        assert!(matches!(
            db.update("missing", "x"),
            Err(Error::UnknownId(_))
        ));
    }

    #[test]
    fn remove_then_absent() {
        let mut db = db();
        db.add("a", "one two three").unwrap();
        db.add("b", "four five six").unwrap();
        assert!(db.remove("a").unwrap());
        assert!(!db.remove("a").unwrap());
        assert_eq!(db.len(), 1);
        let hits = db.query("one two three", 5).unwrap();
        assert!(hits.iter().all(|h| h.id != "a"));
    }

    #[test]
    fn upsert_inserts_then_updates() {
        let mut db = db();
        assert!(db.upsert("a", "first text").unwrap());
        assert!(!db.upsert("a", "second text").unwrap());
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn dimension_mismatch_on_vector() {
        let mut db = db();
        let err = db.add_vector("a", vec![0.0; 5]).unwrap_err();
        assert!(matches!(
            err,
            Error::DimensionMismatch {
                expected: 128,
                got: 5
            }
        ));
    }

    #[test]
    fn k_zero_and_k_larger_than_len() {
        let mut db = db();
        db.add("a", "alpha shared").unwrap();
        db.add("b", "beta shared").unwrap();
        assert!(db.query("alpha shared", 0).unwrap().is_empty());
        assert_eq!(db.query("alpha shared", 100).unwrap().len(), 2);
    }

    #[test]
    fn compact_preserves_results() {
        let mut db = db();
        for i in 0..20 {
            db.add(&format!("id{i}"), &format!("token{i} shared word"))
                .unwrap();
        }
        for i in 0..10 {
            db.remove(&format!("id{i}")).unwrap();
        }
        let before = db.query("shared word", 5).unwrap();
        db.compact();
        let after = db.query("shared word", 5).unwrap();
        assert_eq!(db.len(), 10);
        assert_eq!(before, after);
    }

    #[test]
    fn persistence_roundtrip() {
        let dir = std::env::temp_dir().join(format!("embsearch-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut db = Database::new(MockEmbedder::new(64), Metric::Cosine);
        db.add("a", "persistent apple").unwrap();
        db.add("b", "persistent banana").unwrap();
        db.remove("a").unwrap();
        db.save(&dir).unwrap();

        let reopened = Database::open(MockEmbedder::new(64), &dir).unwrap();
        assert_eq!(reopened.len(), 1);
        let hits = reopened.query("persistent banana", 1).unwrap();
        assert_eq!(hits[0].id, "b");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metric_parsing() {
        assert_eq!("cosine".parse::<Metric>().unwrap(), Metric::Cosine);
        assert_eq!("l2".parse::<Metric>().unwrap(), Metric::Euclidean);
        assert!("bogus".parse::<Metric>().is_err());
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("embsearch-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn empty_id_rejected() {
        let mut db = db();
        assert!(matches!(db.add("", "some text"), Err(Error::Config(_))));
        assert!(matches!(db.update("", "some text"), Err(Error::Config(_))));
        assert!(matches!(db.upsert("", "some text"), Err(Error::Config(_))));
        assert!(matches!(
            db.add_vector("", vec![0.0; 128]),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            db.upsert_batch(vec![("ok", "fine"), ("", "bad")]),
            Err(Error::Config(_))
        ));
        // The batch failed validation before embedding, so nothing was applied.
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn empty_text_rejected() {
        let mut db = db();
        assert!(matches!(db.add("a", ""), Err(Error::Config(_))));
        assert!(matches!(db.add("a", "   \n\t"), Err(Error::Config(_))));
        assert!(matches!(db.upsert("a", ""), Err(Error::Config(_))));
        assert!(matches!(
            db.add_batch(vec![("a", "fine"), ("b", "")]),
            Err(Error::Config(_))
        ));
        assert_eq!(db.len(), 0);
        db.add("a", "real text").unwrap();
        assert!(matches!(db.update("a", ""), Err(Error::Config(_))));
    }

    #[test]
    fn model_id_mismatch_refused_on_open() {
        let dir = temp_dir("model-mismatch");

        let mut db = Database::new(MockEmbedder::new(64), Metric::Cosine);
        db.add("a", "some text").unwrap();
        // Persist the same index under a different model id, as if another
        // embedder had written the store.
        store::save(&dir, db.index(), "other-model-v9", false).unwrap();

        let err = Database::open(MockEmbedder::new(64), &dir).err().unwrap();
        let msg = err.to_string();
        assert!(matches!(err, Error::Config(_)), "got: {err:?}");
        assert!(msg.contains("other-model-v9"), "got: {msg}");
        assert!(msg.contains("mock-hash-v1"), "got: {msg}");
        assert!(msg.contains("re-index"), "got: {msg}");

        // open_or_create must refuse too, not fall back to creating.
        let err = Database::open_or_create(MockEmbedder::new(64), &dir, Metric::Cosine)
            .err()
            .unwrap();
        assert!(matches!(err, Error::Config(_)));

        // Tooling can still load the raw store without an embedder.
        let (index, model_id, _hybrid) = store::load(&dir).unwrap();
        assert_eq!(model_id, "other-model-v9");
        assert_eq!(index.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_manifest_reports_corrupt_store() {
        let dir = temp_dir("corrupt-manifest");

        let mut db = Database::new(MockEmbedder::new(64), Metric::Cosine);
        db.add("a", "some text").unwrap();
        db.save(&dir).unwrap();

        std::fs::write(dir.join("manifest.json"), "{not json").unwrap();
        let err = store::load(&dir).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got: {err:?}");
        assert!(err.to_string().contains("manifest.json"), "got: {err}");

        // Corrupt ids.json is reported the same way.
        db.save(&dir).unwrap();
        std::fs::write(dir.join("ids.json"), "42").unwrap();
        let err = store::load(&dir).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got: {err:?}");
        assert!(err.to_string().contains("ids.json"), "got: {err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_input_error_names_one_line() {
        let err = Error::InvalidInput {
            line: 2,
            msg: "key must be a string".into(),
        };
        assert_eq!(
            err.to_string(),
            "invalid input at line 2: key must be a string"
        );
    }

    #[test]
    fn nonpositive_scores_filtered_for_similarity_metrics() {
        for metric in [Metric::Cosine, Metric::Dot] {
            let mut db = Database::new(MockEmbedder::new(128), metric);
            db.add("a", "apple banana cherry").unwrap();
            // No shared tokens -> orthogonal under the mock embedder -> filtered.
            let hits = db.query("zebra yak walrus", 5).unwrap();
            assert!(hits.is_empty(), "{metric}: got {hits:?}");
        }
        // Euclidean scores are negated distances: legitimately negative, kept.
        let mut db = Database::new(MockEmbedder::new(128), Metric::Euclidean);
        db.add("a", "apple banana cherry").unwrap();
        let hits = db.query("zebra yak walrus", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score < 0.0);
    }

    #[test]
    fn query_hybrid_requires_hybrid_store() {
        let db = Database::new(MockEmbedder::new(64), Metric::Cosine);
        assert!(!db.is_hybrid());
        assert!(matches!(db.query_hybrid("x", 5), Err(Error::Config(_))));
    }

    #[test]
    fn hybrid_finds_keyword_and_semantic_matches() {
        let mut db = Database::new_hybrid(MockEmbedder::new(256), Metric::Cosine, IndexKind::Flat);
        assert!(db.is_hybrid());
        db.add("d1", "the quick brown fox jumps over the lazy dog")
            .unwrap();
        db.add("d2", "rust systems programming language performance")
            .unwrap();
        db.add("d3", "a fast auburn fox leaps above a sleepy hound")
            .unwrap();

        // A rare exact keyword: BM25 pins the one doc that contains it, and
        // fusion surfaces it at the top.
        let hits = db.query_hybrid("rust", 3).unwrap();
        assert_eq!(hits[0].id, "d2");

        // k = 0 short-circuits.
        assert!(db.query_hybrid("rust", 0).unwrap().is_empty());
    }

    #[test]
    fn hybrid_persistence_roundtrip() {
        let dir = temp_dir("hybrid-roundtrip");
        let mut db = Database::new_hybrid(MockEmbedder::new(64), Metric::Cosine, IndexKind::Hnsw);
        db.add("d1", "kubernetes helm chart deployment").unwrap();
        db.add("d2", "the quick brown fox").unwrap();
        db.remove("d2").unwrap();
        db.save(&dir).unwrap();

        // Reopen: hybrid flag, texts, and lexical index all rebuilt from disk.
        let reopened = Database::open(MockEmbedder::new(64), &dir).unwrap();
        assert!(reopened.is_hybrid());
        assert_eq!(reopened.len(), 1);
        let hits = reopened.query_hybrid("helm chart", 5).unwrap();
        assert_eq!(hits[0].id, "d1");
        // Removed doc never comes back.
        assert!(hits.iter().all(|h| h.id != "d2"));

        // open_or_create_hybrid on the existing store honors the stored config.
        let again = Database::open_or_create_hybrid(
            MockEmbedder::new(64),
            &dir,
            Metric::Cosine,
            IndexKind::Flat,
        )
        .unwrap();
        assert!(again.is_hybrid());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
