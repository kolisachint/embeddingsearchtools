//! # embsearch-core
//!
//! A minimal, modular embedding search engine.
//!
//! Three decoupled layers:
//! - [`Embedder`] — text → vector. [`MockEmbedder`] ships for tests/no-model use;
//!   `MiniLmEmbedder` (feature `onnx`) does real `all-MiniLM-L6-v2` inference.
//! - [`Index`] / [`FlatIndex`] — exact brute-force vector search behind a trait,
//!   with configurable [`Metric`].
//! - [`store`] — atomic, mmap-friendly on-disk persistence.
//!
//! [`Database`] composes the three into the primary API: index, query, update.
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
mod index;
pub mod store;

pub use embed::{l2_normalize, Embedder, MockEmbedder};
pub use error::{Error, Result};
pub use index::{FlatIndex, Index, Metric, SearchResult};

#[cfg(feature = "onnx")]
pub use embed::MiniLmEmbedder;

use std::path::Path;

/// The primary API: an [`Embedder`] paired with a [`FlatIndex`].
///
/// Text goes in, vectors are produced and indexed, and queries are answered by
/// embedding the query text and searching. Vectors can also be supplied directly
/// via the `*_vector` methods when embeddings are computed elsewhere.
pub struct Database<E: Embedder> {
    embedder: E,
    index: FlatIndex,
}

impl<E: Embedder> Database<E> {
    /// Create an empty database using `embedder` and `metric`.
    pub fn new(embedder: E, metric: Metric) -> Self {
        let dim = embedder.dim();
        Self {
            embedder,
            index: FlatIndex::new(dim, metric),
        }
    }

    /// Open an existing store from `dir`, pairing it with `embedder`.
    ///
    /// Errors if the store's dimension or recorded model id disagree with the
    /// embedder — a guard against querying a store with the wrong model.
    pub fn open(embedder: E, dir: impl AsRef<Path>) -> Result<Self> {
        let (index, model_id) = store::load(dir)?;
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
        Ok(Self { embedder, index })
    }

    /// Open the store at `dir` if it exists, otherwise create an empty one.
    pub fn open_or_create(embedder: E, dir: impl AsRef<Path>, metric: Metric) -> Result<Self> {
        if store::exists(&dir) {
            Self::open(embedder, dir)
        } else {
            Ok(Self::new(embedder, metric))
        }
    }

    /// Persist the index to `dir`.
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        store::save(dir, &self.index, self.embedder.model_id())
    }

    /// Embed `text` and add it under `id`. Errors if `id` already exists, or if
    /// `id` or `text` is empty.
    pub fn add(&mut self, id: &str, text: &str) -> Result<()> {
        validate_id(id)?;
        validate_text(text)?;
        let v = self.embedder.embed(text)?;
        self.index.add(id, v)
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
        self.index.update(id, v)
    }

    /// Insert or replace `id` with the embedding of `text`. Returns `true` if it
    /// was newly inserted. Errors if `id` or `text` is empty.
    pub fn upsert(&mut self, id: &str, text: &str) -> Result<bool> {
        validate_id(id)?;
        validate_text(text)?;
        let v = self.embedder.embed(text)?;
        self.index.upsert(id, v)
    }

    /// Remove `id`. Returns `true` if it existed.
    pub fn remove(&mut self, id: &str) -> Result<bool> {
        self.index.remove(id)
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
        for (id, v) in ids.into_iter().zip(vectors) {
            self.index.add(&id, v)?;
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
        for (id, v) in ids.into_iter().zip(vectors) {
            if self.index.upsert(&id, v)? {
                inserted += 1;
            } else {
                updated += 1;
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
    }

    /// Borrow the embedder.
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Borrow the underlying index.
    pub fn index(&self) -> &FlatIndex {
        &self.index
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
        store::save(&dir, db.index(), "other-model-v9").unwrap();

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
        let (index, model_id) = store::load(&dir).unwrap();
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
}
