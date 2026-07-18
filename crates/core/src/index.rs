//! Vector index.
//!
//! [`FlatIndex`] is an exact brute-force index: vectors live in a single
//! contiguous `Vec<f32>` matrix, queries scan every live row. It is exact,
//! cheap to update/delete, and fast enough to a few hundred thousand vectors.
//! It sits behind the [`Index`] trait so an approximate backend (e.g. HNSW) can
//! be swapped in later without touching the public API.

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
            Metric::Cosine | Metric::Dot => dot(a, b),
            Metric::Euclidean => {
                let d2: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                -d2.sqrt()
            }
        }
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

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

/// Exact brute-force index backed by a contiguous row-major matrix.
///
/// Deletes are tombstoned (marked dead, not removed) so ids and rows stay stable
/// for cheap updates; [`FlatIndex::compact`] reclaims the dead rows.
#[derive(Debug, Clone)]
pub struct FlatIndex {
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

impl FlatIndex {
    /// Create an empty index.
    pub fn new(dim: usize, metric: Metric) -> Self {
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
    fn prepare(&self, mut v: Vec<f32>) -> Result<Vec<f32>> {
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
    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.dim..(r + 1) * self.dim]
    }

    /// Number of physical rows including tombstones.
    pub fn raw_rows(&self) -> usize {
        self.ids.len()
    }

    /// Reclaim tombstoned rows, rebuilding the matrix compactly. O(n) copy.
    pub fn compact(&mut self) {
        if self.dead == 0 {
            return;
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
    }

    // --- accessors used by the persistence layer ---

    pub(crate) fn parts(&self) -> (&[f32], &[Option<String>]) {
        (&self.data, &self.ids)
    }

    /// Reconstruct an index from persisted parts without re-normalizing (data on
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
}

impl Index for FlatIndex {
    fn add(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        if self.lookup.contains_key(id) {
            return Err(Error::DuplicateId(id.to_string()));
        }
        let v = self.prepare(vector)?;
        let row = self.ids.len();
        self.data.extend_from_slice(&v);
        self.ids.push(Some(id.to_string()));
        self.lookup.insert(id.to_string(), row);
        Ok(())
    }

    fn update(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        let row = *self
            .lookup
            .get(id)
            .ok_or_else(|| Error::UnknownId(id.to_string()))?;
        let v = self.prepare(vector)?;
        self.data[row * self.dim..(row + 1) * self.dim].copy_from_slice(&v);
        Ok(())
    }

    fn upsert(&mut self, id: &str, vector: Vec<f32>) -> Result<bool> {
        if self.lookup.contains_key(id) {
            self.update(id, vector)?;
            Ok(false)
        } else {
            self.add(id, vector)?;
            Ok(true)
        }
    }

    fn remove(&mut self, id: &str) -> Result<bool> {
        match self.lookup.remove(id) {
            Some(row) => {
                self.ids[row] = None;
                self.dead += 1;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn query(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
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

        // Bounded top-k via a small ascending-by-score vector. For the target
        // scale a partial-sort beats a full sort of all rows.
        let mut top: Vec<SearchResult> = Vec::with_capacity(k + 1);
        for (r, id) in self.ids.iter().enumerate() {
            let Some(id) = id else { continue };
            let score = self.metric.score(&q, self.row(r));
            if top.len() < k {
                top.push(SearchResult {
                    id: id.clone(),
                    score,
                });
                if top.len() == k {
                    top.sort_by(|a, b| a.score.total_cmp(&b.score));
                }
            } else if score > top[0].score {
                // Replace current worst, keep `top` sorted ascending.
                top[0] = SearchResult {
                    id: id.clone(),
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

    fn len(&self) -> usize {
        self.lookup.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Metric {
        self.metric
    }
}
