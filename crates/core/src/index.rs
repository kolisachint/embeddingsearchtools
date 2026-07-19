//! Vector index.
//!
//! Two backends sit behind the [`Index`] trait:
//! - [`FlatIndex`] — exact brute-force search. Vectors live in a single
//!   contiguous `Vec<f32>` matrix and every query scans every live row. Exact,
//!   cheap to update/delete, and fast enough to a few hundred thousand vectors.
//! - [`HnswIndex`] — an approximate hierarchical navigable small-world graph
//!   (see [`crate::hnsw`]) that answers queries in sub-linear time, trading a
//!   little recall for a large speedup at scale.
//!
//! Both share [`RowStore`], which owns the vector matrix and the id/tombstone
//! bookkeeping and defines the on-disk representation; the two backends differ
//! only in how they *search* it. [`AnyIndex`] lets a store pick a backend at
//! runtime while presenting one type to the rest of the crate.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Similarity metric used for ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// Cosine similarity. Vectors are stored L2-normalized, so this reduces to a
    /// dot product. Higher is more similar.
    Cosine,
    /// Raw dot product. Higher is more similar.
    Dot,
    /// Negated Euclidean (L2) distance, so that — like the others — higher is
    /// more similar and results sort descending.
    Euclidean,
}

impl Metric {
    /// Whether this metric wants vectors normalized to unit length at insert.
    fn normalizes(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    /// Score two vectors. Larger = more similar for every metric.
    fn score(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Cosine | Metric::Dot => crate::simd::dot(a, b),
            Metric::Euclidean => -crate::simd::sq_euclidean(a, b).sqrt(),
        }
    }

    /// Whether hits scoring `<= 0` should be filtered out. True for the
    /// similarity metrics (orthogonal/opposed vectors are noise), false for
    /// Euclidean, whose scores are legitimately negative distances.
    pub(crate) fn filters_nonpositive(self) -> bool {
        matches!(self, Metric::Cosine | Metric::Dot)
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Metric::Cosine => "cosine",
            Metric::Dot => "dot",
            Metric::Euclidean => "euclidean",
        })
    }
}

impl std::str::FromStr for Metric {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cosine" | "cos" => Ok(Metric::Cosine),
            "dot" | "ip" => Ok(Metric::Dot),
            "euclidean" | "l2" => Ok(Metric::Euclidean),
            other => Err(Error::Config(format!("unknown metric: {other}"))),
        }
    }
}

/// Which index backend a store uses. Persisted in the manifest so a store
/// reopens with the backend it was built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexKind {
    /// Exact brute-force search ([`FlatIndex`]). The default.
    #[default]
    Flat,
    /// Approximate HNSW graph search ([`HnswIndex`]).
    Hnsw,
}

impl std::fmt::Display for IndexKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            IndexKind::Flat => "flat",
            IndexKind::Hnsw => "hnsw",
        })
    }
}

impl std::str::FromStr for IndexKind {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "flat" | "exact" | "brute" => Ok(IndexKind::Flat),
            "hnsw" | "approx" | "ann" => Ok(IndexKind::Hnsw),
            other => Err(Error::Config(format!("unknown index kind: {other}"))),
        }
    }
}

/// A single query hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

/// Behaviour common to all vector indexes.
pub trait Index {
    /// Add a new vector. Errors on duplicate id or dimension mismatch.
    fn add(&mut self, id: &str, vector: Vec<f32>) -> Result<()>;

    /// Replace the vector for an existing id.
    fn update(&mut self, id: &str, vector: Vec<f32>) -> Result<()>;

    /// Add if absent, otherwise update. Returns `true` if it was newly inserted.
    fn upsert(&mut self, id: &str, vector: Vec<f32>) -> Result<bool>;

    /// Remove a vector. Returns `true` if it existed.
    fn remove(&mut self, id: &str) -> Result<bool>;

    /// Top-`k` most similar vectors to `query`, best first.
    fn query(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Number of live vectors.
    fn len(&self) -> usize;

    /// Whether the index holds no live vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimensionality.
    fn dim(&self) -> usize;

    /// Configured similarity metric.
    fn metric(&self) -> Metric;
}

/// Shared vector storage: a contiguous row-major matrix plus id/tombstone
/// bookkeeping. Both [`FlatIndex`] and [`HnswIndex`] search over this; it also
/// defines the persisted layout (see [`crate::store`]).
///
/// Deletes are tombstoned (the row is marked dead, not physically removed) so
/// ids and row indices stay stable for cheap updates; [`RowStore::compact`]
/// reclaims the dead rows.
#[derive(Debug, Clone)]
pub(crate) struct RowStore {
    dim: usize,
    metric: Metric,
    /// Row-major, `rows * dim` long. Row `r` is `data[r*dim .. (r+1)*dim]`.
    data: Vec<f32>,
    /// Id for each row; `None` marks a tombstoned (deleted) row.
    ids: Vec<Option<String>>,
    /// Live id -> row lookup.
    lookup: std::collections::HashMap<String, usize>,
    /// Count of tombstoned rows awaiting compaction.
    dead: usize,
}

impl RowStore {
    pub(crate) fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            data: Vec::new(),
            ids: Vec::new(),
            lookup: std::collections::HashMap::new(),
            dead: 0,
        }
    }

    /// Prepare an incoming vector: validate its length and normalize if the
    /// metric requires it.
    pub(crate) fn prepare(&self, mut v: Vec<f32>) -> Result<Vec<f32>> {
        if v.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: v.len(),
            });
        }
        if self.metric.normalizes() {
            crate::embed::l2_normalize(&mut v);
        }
        Ok(v)
    }

    #[inline]
    pub(crate) fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.dim..(r + 1) * self.dim]
    }

    /// Similarity score between a prepared query and the vector at row `r`.
    #[inline]
    pub(crate) fn score_row(&self, query: &[f32], r: usize) -> f32 {
        self.metric.score(query, self.row(r))
    }

    #[inline]
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub(crate) fn metric(&self) -> Metric {
        self.metric
    }

    /// Number of live (non-tombstoned) rows.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.lookup.len()
    }

    /// Number of physical rows including tombstones.
    #[inline]
    pub(crate) fn raw_rows(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    pub(crate) fn contains(&self, id: &str) -> bool {
        self.lookup.contains_key(id)
    }

    #[inline]
    pub(crate) fn row_of(&self, id: &str) -> Option<usize> {
        self.lookup.get(id).copied()
    }

    /// The id at row `r`, or `None` if the row is tombstoned.
    #[inline]
    pub(crate) fn id_of(&self, r: usize) -> Option<&str> {
        self.ids.get(r).and_then(|o| o.as_deref())
    }

    /// Append a prepared vector under `id`, returning its new row. Errors on a
    /// duplicate id or dimension mismatch.
    pub(crate) fn insert(&mut self, id: &str, vector: Vec<f32>) -> Result<usize> {
        if self.lookup.contains_key(id) {
            return Err(Error::DuplicateId(id.to_string()));
        }
        let v = self.prepare(vector)?;
        let row = self.ids.len();
        self.data.extend_from_slice(&v);
        self.ids.push(Some(id.to_string()));
        self.lookup.insert(id.to_string(), row);
        Ok(row)
    }

    /// Overwrite the vector at `row` with a prepared `vector`.
    pub(crate) fn set(&mut self, row: usize, vector: Vec<f32>) -> Result<()> {
        let v = self.prepare(vector)?;
        self.data[row * self.dim..(row + 1) * self.dim].copy_from_slice(&v);
        Ok(())
    }

    /// Tombstone the row for `id`, returning it if the id existed.
    pub(crate) fn tombstone(&mut self, id: &str) -> Option<usize> {
        let row = self.lookup.remove(id)?;
        self.ids[row] = None;
        self.dead += 1;
        Some(row)
    }

    /// Reclaim tombstoned rows, rebuilding the matrix compactly. O(n) copy.
    /// Returns `true` if any rows were reclaimed (i.e. row indices changed).
    pub(crate) fn compact(&mut self) -> bool {
        if self.dead == 0 {
            return false;
        }
        let live = self.len();
        let mut new_data = Vec::with_capacity(live * self.dim);
        let mut new_ids = Vec::with_capacity(live);
        let mut new_lookup = std::collections::HashMap::with_capacity(live);
        for (r, id) in self.ids.iter().enumerate() {
            if let Some(id) = id {
                new_lookup.insert(id.clone(), new_ids.len());
                new_data.extend_from_slice(&self.data[r * self.dim..(r + 1) * self.dim]);
                new_ids.push(Some(id.clone()));
            }
        }
        self.data = new_data;
        self.ids = new_ids;
        self.lookup = new_lookup;
        self.dead = 0;
        true
    }

    pub(crate) fn parts(&self) -> (&[f32], &[Option<String>]) {
        (&self.data, &self.ids)
    }

    /// Reconstruct storage from persisted parts without re-normalizing (data on
    /// disk is already prepared).
    pub(crate) fn from_parts(
        dim: usize,
        metric: Metric,
        data: Vec<f32>,
        ids: Vec<Option<String>>,
    ) -> Result<Self> {
        if data.len() != ids.len() * dim {
            return Err(Error::corrupt(format!(
                "vector buffer {} not divisible by dim {} into {} rows",
                data.len(),
                dim,
                ids.len()
            )));
        }
        let mut lookup = std::collections::HashMap::new();
        let mut dead = 0;
        for (r, id) in ids.iter().enumerate() {
            match id {
                Some(id) => {
                    if lookup.insert(id.clone(), r).is_some() {
                        return Err(Error::corrupt(format!("duplicate id on load: {id}")));
                    }
                }
                None => dead += 1,
            }
        }
        Ok(Self {
            dim,
            metric,
            data,
            ids,
            lookup,
            dead,
        })
    }

    /// Exact brute-force top-`k` over the live rows. Shared by [`FlatIndex`] and
    /// used as the recall oracle in tests.
    pub(crate) fn brute_force(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(vec![]);
        }
        let mut q = query.to_vec();
        if self.metric.normalizes() {
            crate::embed::l2_normalize(&mut q);
        }
        let filter = self.metric.filters_nonpositive();

        // Bounded top-k via a small ascending-by-score vector. For the target
        // scale a partial-sort beats a full sort of all rows.
        let mut top: Vec<SearchResult> = Vec::with_capacity(k + 1);
        for r in 0..self.ids.len() {
            let Some(id) = self.id_of(r) else { continue };
            let score = self.score_row(&q, r);
            if filter && score <= 0.0 {
                continue;
            }
            if top.len() < k {
                top.push(SearchResult {
                    id: id.to_string(),
                    score,
                });
                if top.len() == k {
                    top.sort_by(|a, b| a.score.total_cmp(&b.score));
                }
            } else if score > top[0].score {
                // Replace current worst, keep `top` sorted ascending.
                top[0] = SearchResult {
                    id: id.to_string(),
                    score,
                };
                let mut i = 0;
                while i + 1 < top.len() && top[i].score > top[i + 1].score {
                    top.swap(i, i + 1);
                    i += 1;
                }
            }
        }
        // Fewer than k live rows: ensure sorted before reversing.
        if top.len() < k {
            top.sort_by(|a, b| a.score.total_cmp(&b.score));
        }
        top.reverse(); // best (highest score) first
        Ok(top)
    }
}

/// Exact brute-force index backed by a contiguous row-major matrix.
///
/// Under [`Metric::Cosine`] and [`Metric::Dot`], queries exclude rows scoring
/// `<= 0.0` (orthogonal or opposed vectors), so a query may return fewer than
/// `k` hits rather than padding with noise. [`Metric::Euclidean`] scores are
/// negated distances and are never filtered.
#[derive(Debug, Clone)]
pub struct FlatIndex {
    store: RowStore,
}

impl FlatIndex {
    /// Create an empty index.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            store: RowStore::new(dim, metric),
        }
    }

    /// Number of physical rows including tombstones.
    pub fn raw_rows(&self) -> usize {
        self.store.raw_rows()
    }

    /// Reclaim tombstoned rows, rebuilding the matrix compactly.
    pub fn compact(&mut self) {
        self.store.compact();
    }

    // --- accessors used by the persistence layer ---

    pub(crate) fn parts(&self) -> (&[f32], &[Option<String>]) {
        self.store.parts()
    }

    /// Reconstruct an index from persisted parts.
    pub(crate) fn from_parts(
        dim: usize,
        metric: Metric,
        data: Vec<f32>,
        ids: Vec<Option<String>>,
    ) -> Result<Self> {
        Ok(Self {
            store: RowStore::from_parts(dim, metric, data, ids)?,
        })
    }
}

impl Index for FlatIndex {
    fn add(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        self.store.insert(id, vector).map(|_| ())
    }

    fn update(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        let row = self
            .store
            .row_of(id)
            .ok_or_else(|| Error::UnknownId(id.to_string()))?;
        self.store.set(row, vector)
    }

    fn upsert(&mut self, id: &str, vector: Vec<f32>) -> Result<bool> {
        if self.store.contains(id) {
            self.update(id, vector)?;
            Ok(false)
        } else {
            self.add(id, vector)?;
            Ok(true)
        }
    }

    fn remove(&mut self, id: &str) -> Result<bool> {
        Ok(self.store.tombstone(id).is_some())
    }

    fn query(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.store.brute_force(query, k)
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn dim(&self) -> usize {
        self.store.dim()
    }

    fn metric(&self) -> Metric {
        self.store.metric()
    }
}

pub use crate::hnsw::HnswIndex;

/// A runtime-selected index backend. Presents one type to [`crate::Database`]
/// and the persistence layer while dispatching to the chosen implementation.
#[derive(Debug, Clone)]
pub enum AnyIndex {
    Flat(FlatIndex),
    Hnsw(HnswIndex),
}

impl AnyIndex {
    /// Create an empty index of the given `kind`.
    pub fn new(kind: IndexKind, dim: usize, metric: Metric) -> Self {
        match kind {
            IndexKind::Flat => AnyIndex::Flat(FlatIndex::new(dim, metric)),
            IndexKind::Hnsw => AnyIndex::Hnsw(HnswIndex::new(dim, metric)),
        }
    }

    /// Which backend this is.
    pub fn kind(&self) -> IndexKind {
        match self {
            AnyIndex::Flat(_) => IndexKind::Flat,
            AnyIndex::Hnsw(_) => IndexKind::Hnsw,
        }
    }

    /// Reclaim tombstoned rows.
    pub fn compact(&mut self) {
        match self {
            AnyIndex::Flat(i) => i.compact(),
            AnyIndex::Hnsw(i) => i.compact(),
        }
    }

    /// Number of physical rows including tombstones.
    pub fn raw_rows(&self) -> usize {
        match self {
            AnyIndex::Flat(i) => i.raw_rows(),
            AnyIndex::Hnsw(i) => i.raw_rows(),
        }
    }

    pub(crate) fn parts(&self) -> (&[f32], &[Option<String>]) {
        match self {
            AnyIndex::Flat(i) => i.parts(),
            AnyIndex::Hnsw(i) => i.parts(),
        }
    }

    /// Reconstruct an index of `kind` from persisted parts. HNSW rebuilds its
    /// navigation graph from the vectors (the graph itself is not persisted).
    pub(crate) fn from_parts(
        kind: IndexKind,
        dim: usize,
        metric: Metric,
        data: Vec<f32>,
        ids: Vec<Option<String>>,
    ) -> Result<Self> {
        Ok(match kind {
            IndexKind::Flat => AnyIndex::Flat(FlatIndex::from_parts(dim, metric, data, ids)?),
            IndexKind::Hnsw => AnyIndex::Hnsw(HnswIndex::from_parts(dim, metric, data, ids)?),
        })
    }
}

impl Index for AnyIndex {
    fn add(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        match self {
            AnyIndex::Flat(i) => i.add(id, vector),
            AnyIndex::Hnsw(i) => i.add(id, vector),
        }
    }

    fn update(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        match self {
            AnyIndex::Flat(i) => i.update(id, vector),
            AnyIndex::Hnsw(i) => i.update(id, vector),
        }
    }

    fn upsert(&mut self, id: &str, vector: Vec<f32>) -> Result<bool> {
        match self {
            AnyIndex::Flat(i) => i.upsert(id, vector),
            AnyIndex::Hnsw(i) => i.upsert(id, vector),
        }
    }

    fn remove(&mut self, id: &str) -> Result<bool> {
        match self {
            AnyIndex::Flat(i) => i.remove(id),
            AnyIndex::Hnsw(i) => i.remove(id),
        }
    }

    fn query(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        match self {
            AnyIndex::Flat(i) => i.query(query, k),
            AnyIndex::Hnsw(i) => i.query(query, k),
        }
    }

    fn len(&self) -> usize {
        match self {
            AnyIndex::Flat(i) => i.len(),
            AnyIndex::Hnsw(i) => i.len(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            AnyIndex::Flat(i) => i.dim(),
            AnyIndex::Hnsw(i) => i.dim(),
        }
    }

    fn metric(&self) -> Metric {
        match self {
            AnyIndex::Flat(i) => i.metric(),
            AnyIndex::Hnsw(i) => i.metric(),
        }
    }
}
