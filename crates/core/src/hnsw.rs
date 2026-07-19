//! Approximate nearest-neighbor search via a Hierarchical Navigable Small World
//! graph (HNSW, Malkov & Yashunin 2016).
//!
//! [`HnswIndex`] implements the same [`Index`](crate::Index) trait as
//! [`FlatIndex`](crate::FlatIndex) and shares its [`RowStore`] for vector
//! storage and persistence — the two differ only in how they *search*. Where the
//! flat index scans every vector, HNSW walks a layered proximity graph: a sparse
//! top layer for long hops, progressively denser layers for local refinement.
//! Query cost is roughly `O(log n)` instead of `O(n)`, at the price of a small,
//! tunable recall loss.
//!
//! ## Design choices for a small, dependency-free build
//!
//! - **The graph is never persisted.** [`RowStore`] already stores every vector;
//!   the graph is a pure acceleration structure rebuilt from those vectors on
//!   load ([`HnswIndex::from_parts`]) and after [`compact`](HnswIndex::compact).
//!   That keeps the on-disk format identical to the flat backend (vectors + ids
//!   + manifest, still mmap-friendly) and sidesteps graph-serialization bugs.
//! - **Deletes are tombstones.** A removed node stays in the graph as a routing
//!   waypoint (preserving connectivity) but is filtered out of results by id.
//!   `compact` rebuilds the graph over only the live rows.
//! - **Level assignment is seeded and deterministic**, so an index rebuilt from
//!   the same vectors in the same order yields the same graph.
//!
//! Result semantics match [`FlatIndex`] exactly: scores are the metric's scores
//! (higher = more similar) and, under cosine/dot, non-positive hits are filtered.

use crate::error::Result;
use crate::index::{Index, Metric, RowStore, SearchResult};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// --- tunables ----------------------------------------------------------------

/// Max neighbors per node on layers above 0.
const M: usize = 16;
/// Max neighbors per node on layer 0 (denser, as is conventional: `2 * M`).
const M0: usize = 32;
/// Candidate-list size while building the graph. Larger = better graph, slower
/// inserts.
const EF_CONSTRUCTION: usize = 128;
/// Candidate-list size while querying. Larger = better recall, slower queries.
/// The effective value is `max(EF_SEARCH, k)`. Queries are far cheaper than the
/// exact scan, so this is set generously to keep recall high.
const EF_SEARCH: usize = 128;
/// Hard cap on layer count, so a pathological RNG draw can't allocate absurdly.
const MAX_LEVEL: usize = 16;
/// Fixed RNG seed: builds are reproducible from the vectors alone.
const SEED: u64 = 0x243F_6A88_85A3_08D3;

// --- ordered float for the search heaps --------------------------------------

/// A total-ordered `f32` wrapper so distances can live in a [`BinaryHeap`].
/// All distances here are finite, so `total_cmp` is a total order.
#[derive(Debug, Clone, Copy)]
struct Ordf(f32);

impl PartialEq for Ordf {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}
impl Eq for Ordf {}
impl PartialOrd for Ordf {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ordf {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// --- visited set (epoch-stamped, allocation-free per search) -----------------

/// A visited-set that avoids per-search allocation. Instead of clearing a
/// `HashSet` each call, every node carries the epoch it was last stamped with;
/// bumping the current epoch logically empties the set in O(1). This is the
/// standard HNSW trick and matters because graph construction runs a search per
/// insert on every layer — millions of tiny sets otherwise.
struct Visited {
    stamp: Vec<u32>,
    epoch: u32,
}

impl Visited {
    fn new() -> Self {
        Self {
            stamp: Vec::new(),
            epoch: 0,
        }
    }

    /// Prepare for a search over `n` nodes: grow if needed and start a fresh
    /// epoch. Handles `u32` wraparound by zeroing (astronomically rare).
    fn begin(&mut self, n: usize) {
        if self.stamp.len() < n {
            self.stamp.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
    }

    /// Mark `node` visited, returning `true` if it was newly inserted.
    #[inline]
    fn insert(&mut self, node: u32) -> bool {
        let slot = &mut self.stamp[node as usize];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }
}

thread_local! {
    /// Per-thread scratch visited-set reused across searches.
    static VISITED: RefCell<Visited> = RefCell::new(Visited::new());
}

// --- deterministic RNG (splitmix64) ------------------------------------------

/// Tiny deterministic PRNG (splitmix64) — no external dependency, and seeded so
/// graph construction is reproducible.
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// --- the index ---------------------------------------------------------------

/// Approximate nearest-neighbor index over a layered proximity graph.
///
/// See the [module docs](self) for the design. Construction parameters (`M`,
/// `ef_construction`, `ef_search`) are compile-time constants tuned for good
/// recall at the target scale.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    store: RowStore,
    /// Adjacency, indexed by row: `links[row][layer]` is that node's neighbor
    /// rows at `layer`. A node exists on layers `0..links[row].len()`; an empty
    /// outer vec means the row is not a graph node (tombstoned or detached).
    links: Vec<Vec<Vec<u32>>>,
    /// Entry point (top-layer node) for searches, or `None` when empty.
    entry: Option<u32>,
    /// Highest layer currently populated.
    max_layer: usize,
    rng: Rng,
    m_l: f64,
    /// Query-time candidate-list size. Tunable per index; see [`set_ef_search`].
    ///
    /// [`set_ef_search`]: HnswIndex::set_ef_search
    ef_search: usize,
}

impl HnswIndex {
    /// Create an empty HNSW index.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            store: RowStore::new(dim, metric),
            links: Vec::new(),
            entry: None,
            max_layer: 0,
            rng: Rng::new(SEED),
            m_l: 1.0 / (M as f64).ln(),
            ef_search: EF_SEARCH,
        }
    }

    /// Set the query-time candidate-list size (`ef`). Higher values raise recall
    /// toward exact search at the cost of query latency; the effective value is
    /// always at least `k`. This only affects [`query`](Index::query) and can be
    /// changed at any time. The default is 128.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef.max(1);
    }

    /// The current query-time candidate-list size.
    pub fn ef_search(&self) -> usize {
        self.ef_search
    }

    /// Number of physical rows including tombstones.
    pub fn raw_rows(&self) -> usize {
        self.store.raw_rows()
    }

    /// Reclaim tombstoned rows and rebuild the graph over the survivors.
    pub fn compact(&mut self) {
        if self.store.compact() {
            self.rebuild_graph();
        }
    }

    pub(crate) fn parts(&self) -> (&[f32], &[Option<String>]) {
        self.store.parts()
    }

    /// Reconstruct from persisted parts, rebuilding the navigation graph from
    /// the stored vectors (the graph itself is never written to disk).
    pub(crate) fn from_parts(
        dim: usize,
        metric: Metric,
        data: Vec<f32>,
        ids: Vec<Option<String>>,
    ) -> Result<Self> {
        let store = RowStore::from_parts(dim, metric, data, ids)?;
        let mut idx = Self {
            store,
            links: Vec::new(),
            entry: None,
            max_layer: 0,
            rng: Rng::new(SEED),
            m_l: 1.0 / (M as f64).ln(),
            ef_search: EF_SEARCH,
        };
        idx.rebuild_graph();
        Ok(idx)
    }

    // --- graph maintenance ---------------------------------------------------

    /// Distance from a prepared query to the vector at `node`. Smaller = closer,
    /// so it is the negated similarity score (which the metric defines so that
    /// higher = more similar).
    #[inline]
    fn dist(&self, query: &[f32], node: u32) -> f32 {
        -self.store.score_row(query, node as usize)
    }

    /// Distance between the vectors at two nodes.
    #[inline]
    fn node_dist(&self, a: u32, b: u32) -> f32 {
        -self.store.score_row(self.store.row(a as usize), b as usize)
    }

    /// Select up to `m` neighbors from `found` using the HNSW diversity
    /// heuristic (Malkov & Yashunin, Algorithm 4).
    ///
    /// A candidate `e` is kept only if it is closer to the base than to any
    /// already-selected neighbor — so the chosen set spreads across directions
    /// instead of clustering, which is what keeps *outliers* reachable: a plain
    /// "closest M" rule severs the long links that connect a distant node back
    /// into the graph. Any shortfall is then topped up with the closest
    /// discarded candidates (`keepPrunedConnections`) so node degree stays high.
    fn select_neighbors(&self, found: &[(f32, u32)], m: usize) -> Vec<u32> {
        let mut cands = found.to_vec();
        cands.sort_by(|a, b| a.0.total_cmp(&b.0)); // nearest to base first
        let mut kept: Vec<u32> = Vec::with_capacity(m);
        let mut discarded: Vec<u32> = Vec::new();
        for &(dist_to_base, e) in &cands {
            if kept.len() >= m {
                break;
            }
            let diverse = kept
                .iter()
                .all(|&sel| self.node_dist(e, sel) >= dist_to_base);
            if diverse {
                kept.push(e);
            } else {
                discarded.push(e);
            }
        }
        for e in discarded {
            if kept.len() >= m {
                break;
            }
            kept.push(e);
        }
        kept
    }

    /// Draw an insertion level from the exponential level distribution.
    fn random_level(&mut self) -> usize {
        let u = 1.0 - self.rng.next_f64(); // (0, 1], so ln() is finite and <= 0
        let level = (-(u.ln()) * self.m_l).floor();
        (level.max(0.0) as usize).min(MAX_LEVEL)
    }

    /// Greedy descent within a single `layer`: hop to the closest neighbor until
    /// no neighbor improves on the current node.
    fn greedy(&self, query: &[f32], entry: u32, layer: usize) -> u32 {
        let mut cur = entry;
        let mut cur_d = self.dist(query, cur);
        loop {
            let mut improved = false;
            if let Some(nbrs) = self.links[cur as usize].get(layer) {
                for &nb in nbrs {
                    let d = self.dist(query, nb);
                    if d < cur_d {
                        cur_d = d;
                        cur = nb;
                        improved = true;
                    }
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    /// Best-first search of a single `layer`, returning up to `ef` nodes closest
    /// to `query` as `(distance, node)` pairs (unordered).
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
    ) -> Vec<(f32, u32)> {
        VISITED.with(|v| {
            let mut visited = v.borrow_mut();
            visited.begin(self.links.len());
            // Candidates: min-heap by distance (explore closest first).
            let mut candidates: BinaryHeap<Reverse<(Ordf, u32)>> = BinaryHeap::new();
            // Results: max-heap by distance (the farthest kept result is at the top).
            let mut results: BinaryHeap<(Ordf, u32)> = BinaryHeap::new();

            for &e in entry_points {
                if visited.insert(e) {
                    let d = self.dist(query, e);
                    candidates.push(Reverse((Ordf(d), e)));
                    results.push((Ordf(d), e));
                }
            }

            while let Some(Reverse((Ordf(cd), c))) = candidates.pop() {
                let farthest = match results.peek() {
                    Some(&(Ordf(d), _)) => d,
                    None => break,
                };
                // Closest remaining candidate is worse than our worst keeper: done.
                if cd > farthest {
                    break;
                }
                if let Some(nbrs) = self.links[c as usize].get(layer) {
                    for &nb in nbrs {
                        if visited.insert(nb) {
                            let d = self.dist(query, nb);
                            let farthest = results
                                .peek()
                                .map(|&(Ordf(d), _)| d)
                                .unwrap_or(f32::INFINITY);
                            if results.len() < ef || d < farthest {
                                candidates.push(Reverse((Ordf(d), nb)));
                                results.push((Ordf(d), nb));
                                if results.len() > ef {
                                    results.pop();
                                }
                            }
                        }
                    }
                }
            }

            results.into_iter().map(|(Ordf(d), n)| (d, n)).collect()
        })
    }

    /// Insert the vector already stored at `row` into the graph. Assumes
    /// `links[row]` exists (as an empty slot).
    fn insert_node(&mut self, row: usize) {
        let level = self.random_level();
        self.links[row].clear();
        self.links[row].resize_with(level + 1, Vec::new);

        let q = self.store.row(row).to_vec();
        let mut ep = match self.entry {
            Some(e) => e,
            None => {
                self.entry = Some(row as u32);
                self.max_layer = level;
                return;
            }
        };
        let cur_max = self.max_layer;

        // Descend from the top down to just above the node's level with ef = 1.
        if cur_max > level {
            for lc in ((level + 1)..=cur_max).rev() {
                ep = self.greedy(&q, ep, lc);
            }
        }

        // From the node's level down to 0, connect it to its ef_construction
        // nearest neighbors on each layer.
        let start = level.min(cur_max);
        let mut entry_points = vec![ep];
        for lc in (0..=start).rev() {
            let found = self.search_layer(&q, &entry_points, EF_CONSTRUCTION, lc);
            let cap = if lc == 0 { M0 } else { M };
            let selected = self.select_neighbors(&found, cap);

            for &nb in &selected {
                self.links[row][lc].push(nb);
                self.links[nb as usize][lc].push(row as u32);
            }
            for &nb in &selected {
                self.prune(nb, lc);
            }

            entry_points = found.iter().map(|&(_, n)| n).collect();
            if entry_points.is_empty() {
                entry_points = vec![ep];
            }
        }

        if level > cur_max {
            self.entry = Some(row as u32);
            self.max_layer = level;
        }
    }

    /// Trim node `node`'s layer-`layer` neighbor list back to the cap using the
    /// same diversity heuristic as insertion, so pruning preserves the long
    /// links that keep the graph connected rather than just dropping the
    /// farthest neighbors.
    fn prune(&mut self, node: u32, layer: usize) {
        let cap = if layer == 0 { M0 } else { M };
        if self.links[node as usize][layer].len() <= cap {
            return;
        }
        let cands: Vec<(f32, u32)> = self.links[node as usize][layer]
            .iter()
            .map(|&x| (self.node_dist(node, x), x))
            .collect();
        self.links[node as usize][layer] = self.select_neighbors(&cands, cap);
    }

    /// Unlink `row` from the graph entirely (used before re-inserting on update).
    fn detach(&mut self, row: usize) {
        let layers: Vec<Vec<u32>> = self.links[row].clone();
        for (lc, nbrs) in layers.iter().enumerate() {
            for &nb in nbrs {
                if let Some(list) = self.links[nb as usize].get_mut(lc) {
                    list.retain(|&x| x != row as u32);
                }
            }
        }
        self.links[row].clear();
        if self.entry == Some(row as u32) {
            self.recompute_entry();
        }
    }

    /// Pick the highest-level remaining node as the entry point.
    fn recompute_entry(&mut self) {
        let mut best: Option<(usize, usize)> = None; // (level, row)
        for (r, layers) in self.links.iter().enumerate() {
            if layers.is_empty() {
                continue;
            }
            let level = layers.len() - 1;
            if best.is_none_or(|(bl, _)| level > bl) {
                best = Some((level, r));
            }
        }
        match best {
            Some((level, row)) => {
                self.entry = Some(row as u32);
                self.max_layer = level;
            }
            None => {
                self.entry = None;
                self.max_layer = 0;
            }
        }
    }

    /// Drop the whole graph and rebuild it over the store's live rows.
    fn rebuild_graph(&mut self) {
        let n = self.store.raw_rows();
        self.links = vec![Vec::new(); n];
        self.entry = None;
        self.max_layer = 0;
        self.rng = Rng::new(SEED);
        for r in 0..n {
            if self.store.id_of(r).is_some() {
                self.insert_node(r);
            }
        }
    }
}

impl Index for HnswIndex {
    fn add(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        let row = self.store.insert(id, vector)?;
        // Keep `links` aligned 1:1 with the store's rows.
        debug_assert_eq!(row, self.links.len());
        self.links.push(Vec::new());
        self.insert_node(row);
        Ok(())
    }

    fn update(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        let row = self
            .store
            .row_of(id)
            .ok_or_else(|| crate::Error::UnknownId(id.to_string()))?;
        self.store.set(row, vector)?;
        // The stored vector moved, so the node's graph position is stale:
        // detach and re-insert it fresh.
        self.detach(row);
        self.insert_node(row);
        Ok(())
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
        // Tombstone in the store; the node stays as a routing waypoint and is
        // filtered from results by id. `compact` rebuilds without it.
        Ok(self.store.tombstone(id).is_some())
    }

    fn query(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if k == 0 {
            return Ok(vec![]);
        }
        // Validate dimension and normalize exactly as the flat path does.
        let q = self.store.prepare(query.to_vec())?;
        if self.store.len() == 0 {
            return Ok(vec![]);
        }
        let Some(entry) = self.entry else {
            return Ok(vec![]);
        };

        let mut ep = entry;
        for lc in (1..=self.max_layer).rev() {
            ep = self.greedy(&q, ep, lc);
        }

        let ef = self.ef_search.max(k);
        let found = self.search_layer(&q, &[ep], ef, 0);

        let filter = self.store.metric().filters_nonpositive();
        let mut hits: Vec<SearchResult> = Vec::with_capacity(found.len());
        for (d, node) in found {
            let Some(id) = self.store.id_of(node as usize) else {
                continue; // tombstoned waypoint
            };
            let score = -d;
            if filter && score <= 0.0 {
                continue;
            }
            hits.push(SearchResult {
                id: id.to_string(),
                score,
            });
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(k);
        Ok(hits)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatIndex;
    use std::collections::HashSet;

    /// Deterministic vector generator for tests.
    fn gen(seed: u64, dim: usize) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..dim)
            .map(|_| (rng.next_f64() as f32) * 2.0 - 1.0)
            .collect()
    }

    /// Recall@k of HNSW against the exact flat index over the same data.
    fn recall(metric: Metric, n: usize, dim: usize, queries: usize, k: usize) -> f64 {
        let mut hnsw = HnswIndex::new(dim, metric);
        let mut flat = FlatIndex::new(dim, metric);
        for i in 0..n {
            let v = gen(i as u64 + 1, dim);
            hnsw.add(&format!("id{i}"), v.clone()).unwrap();
            flat.add(&format!("id{i}"), v).unwrap();
        }

        let mut total = 0.0;
        for q in 0..queries {
            let qv = gen(1_000_000 + q as u64, dim);
            let approx: HashSet<String> = hnsw
                .query(&qv, k)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            let exact = flat.query(&qv, k).unwrap();
            let denom = exact.len().max(1);
            let hit = exact.iter().filter(|e| approx.contains(&e.id)).count();
            total += hit as f64 / denom as f64;
        }
        total / queries as f64
    }

    #[test]
    fn recall_is_high_euclidean() {
        // Euclidean never filters, so top-k is always full — a clean recall test.
        let r = recall(Metric::Euclidean, 800, 32, 60, 10);
        assert!(r >= 0.90, "euclidean recall too low: {r}");
    }

    #[test]
    fn recall_is_high_cosine() {
        let r = recall(Metric::Cosine, 800, 32, 60, 10);
        assert!(r >= 0.90, "cosine recall too low: {r}");
    }

    #[test]
    fn top1_is_exact_on_structured_data() {
        // With a clearly-nearest vector, the approximate index must still find it.
        let dim = 16;
        let mut idx = HnswIndex::new(dim, Metric::Cosine);
        for i in 0..200 {
            idx.add(&format!("id{i}"), gen(i as u64 + 1, dim)).unwrap();
        }
        // Query equal to a stored vector: that id must rank first.
        let target = gen(42 + 1, dim);
        let hits = idx.query(&target, 1).unwrap();
        assert_eq!(hits[0].id, "id42");
    }

    #[test]
    fn update_moves_the_vector() {
        let dim = 16;
        let mut idx = HnswIndex::new(dim, Metric::Euclidean);
        for i in 0..100 {
            idx.add(&format!("id{i}"), gen(i as u64 + 1, dim)).unwrap();
        }
        let target = gen(7 + 1, dim);
        assert_eq!(idx.query(&target, 1).unwrap()[0].id, "id7");
        // Move id7 far away; it should no longer be the nearest to `target`.
        idx.update("id7", gen(999_999, dim)).unwrap();
        let hits = idx.query(&target, 5).unwrap();
        assert_ne!(hits[0].id, "id7");
        // And it is findable at its new location.
        let near_new = idx.query(&gen(999_999, dim), 1).unwrap();
        assert_eq!(near_new[0].id, "id7");
        assert_eq!(idx.len(), 100);
    }

    #[test]
    fn remove_then_absent_and_compact_preserves() {
        let dim = 16;
        let mut idx = HnswIndex::new(dim, Metric::Euclidean);
        for i in 0..120 {
            idx.add(&format!("id{i}"), gen(i as u64 + 1, dim)).unwrap();
        }
        for i in 0..40 {
            assert!(idx.remove(&format!("id{i}")).unwrap());
        }
        assert!(!idx.remove("id0").unwrap());
        assert_eq!(idx.len(), 80);

        // Removed ids never surface.
        for q in 0..20 {
            let hits = idx.query(&gen(2_000_000 + q, dim), 10).unwrap();
            assert!(hits.iter().all(|h| {
                let n: usize = h.id.trim_start_matches("id").parse().unwrap();
                n >= 40
            }));
        }

        idx.compact();
        assert_eq!(idx.len(), 80);
        assert_eq!(idx.raw_rows(), 80); // tombstones reclaimed
        let hits = idx.query(&gen(50 + 1, dim), 1).unwrap();
        assert_eq!(hits[0].id, "id50");
    }

    #[test]
    fn rebuild_from_parts_matches_live_graph() {
        let dim = 24;
        let mut idx = HnswIndex::new(dim, Metric::Cosine);
        for i in 0..300 {
            idx.add(&format!("id{i}"), gen(i as u64 + 1, dim)).unwrap();
        }
        idx.remove("id5").unwrap();

        let (data, ids) = idx.parts();
        let rebuilt =
            HnswIndex::from_parts(dim, Metric::Cosine, data.to_vec(), ids.to_vec()).unwrap();
        assert_eq!(rebuilt.len(), idx.len());

        // Rebuild is deterministic, so results match the live index exactly.
        for q in 0..25 {
            let qv = gen(3_000_000 + q, dim);
            assert_eq!(idx.query(&qv, 10).unwrap(), rebuilt.query(&qv, 10).unwrap());
        }
    }

    #[test]
    fn outlier_is_reachable() {
        // Regression: a near-orthogonal outlier (the shape the mock embedder
        // produces for out-of-vocabulary text) must stay reachable. A naive
        // "keep closest M" neighbor rule severs its inbound links; the diversity
        // heuristic must preserve them so an exact-duplicate query finds it.
        let dim = 32;
        let mut idx = HnswIndex::new(dim, Metric::Cosine);
        // A tight cluster: vectors sharing a dominant first coordinate.
        for i in 0..400 {
            let mut v = gen(i as u64 + 1, dim);
            v[0] += 8.0; // pull them all into one region of the sphere
            idx.add(&format!("id{i}"), v).unwrap();
        }
        // One outlier pointing in an orthogonal direction.
        let mut outlier = vec![0.0f32; dim];
        outlier[dim - 1] = 1.0;
        idx.add("outlier", outlier.clone()).unwrap();

        // Querying the outlier's exact vector must return it first.
        let hits = idx.query(&outlier, 1).unwrap();
        assert_eq!(hits[0].id, "outlier", "outlier was unreachable");

        // Survives a rebuild (the persistence path) too.
        let (data, ids) = idx.parts();
        let rebuilt =
            HnswIndex::from_parts(dim, Metric::Cosine, data.to_vec(), ids.to_vec()).unwrap();
        assert_eq!(rebuilt.query(&outlier, 1).unwrap()[0].id, "outlier");
    }

    #[test]
    fn edge_cases() {
        let dim = 8;
        let mut idx = HnswIndex::new(dim, Metric::Cosine);
        // Empty index.
        assert!(idx.query(&gen(1, dim), 5).unwrap().is_empty());
        idx.add("a", gen(1, dim)).unwrap();
        // k = 0.
        assert!(idx.query(&gen(1, dim), 0).unwrap().is_empty());
        // Dimension mismatch is rejected like the flat path.
        assert!(matches!(
            idx.query(&vec![0.0; dim + 1], 1),
            Err(crate::Error::DimensionMismatch { .. })
        ));
        // Duplicate add is rejected.
        assert!(matches!(
            idx.add("a", gen(2, dim)),
            Err(crate::Error::DuplicateId(_))
        ));
    }
}
