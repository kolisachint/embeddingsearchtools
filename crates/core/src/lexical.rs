//! Sparse lexical retrieval (BM25) for hybrid search.
//!
//! Dense vector search matches on *meaning* but can miss exact terms — rare
//! tokens, identifiers, codes, names — whose signal a smoothed embedding washes
//! out. A classic Okapi BM25 inverted index covers that gap, and
//! [`crate::Database::query_hybrid`] fuses the two rankings.
//!
//! This index is deliberately small: whitespace/punctuation tokenization,
//! lowercasing, no stemming or stopword list (BM25's IDF term already discounts
//! common words). Deletes are tombstoned and filtered at query time, matching
//! the vector side; the whole thing rebuilds from the stored texts on load.

use std::collections::HashMap;

/// Okapi BM25 term-frequency saturation. 1.2 is the standard default.
const K1: f32 = 1.2;
/// Okapi BM25 length-normalization strength. 0.75 is the standard default.
const B: f32 = 0.75;

/// One posting: a document ordinal and the term's frequency in it.
#[derive(Debug, Clone)]
struct Posting {
    ord: u32,
    tf: u32,
}

/// An in-memory BM25 inverted index over document texts.
///
/// Documents are addressed by string id (shared with the vector index).
/// Internally each gets a monotonic ordinal; deletes tombstone the ordinal and
/// are filtered from results and statistics, so scores stay correct without
/// rewriting the postings. [`compact`](LexicalIndex::compact) or a reload drops
/// the tombstones.
#[derive(Debug, Clone, Default)]
pub struct LexicalIndex {
    postings: HashMap<String, Vec<Posting>>,
    /// Token count per ordinal (0 for a tombstoned doc).
    len: Vec<u32>,
    /// Id per ordinal (`None` once removed).
    ids: Vec<Option<String>>,
    lookup: HashMap<String, u32>,
    live: usize,
    total_len: u64,
}

/// Split `text` into lowercased alphanumeric tokens.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

impl LexicalIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live documents.
    pub fn len(&self) -> usize {
        self.live
    }

    /// Whether the index holds no live documents.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Index `text` under `id`. If `id` already exists it is replaced.
    pub fn add(&mut self, id: &str, text: &str) {
        if self.lookup.contains_key(id) {
            self.remove(id);
        }
        let tokens = tokenize(text);
        let ord = self.ids.len() as u32;

        // Term frequencies within this document.
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }
        for (term, count) in tf {
            self.postings
                .entry(term.to_string())
                .or_default()
                .push(Posting { ord, tf: count });
        }

        self.len.push(tokens.len() as u32);
        self.ids.push(Some(id.to_string()));
        self.lookup.insert(id.to_string(), ord);
        self.total_len += tokens.len() as u64;
        self.live += 1;
    }

    /// Remove `id`. Returns `true` if it existed. The ordinal is tombstoned;
    /// its postings are ignored until [`compact`](LexicalIndex::compact).
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(ord) = self.lookup.remove(id) else {
            return false;
        };
        self.ids[ord as usize] = None;
        self.total_len -= self.len[ord as usize] as u64;
        self.len[ord as usize] = 0;
        self.live -= 1;
        true
    }

    /// Drop tombstoned documents and renumber, shrinking the postings.
    pub fn compact(&mut self) {
        if self.ids.len() == self.live {
            return;
        }
        let mut rebuilt = LexicalIndex::new();
        // Re-add survivors is not possible without their text; instead remap
        // ordinals in place. Build old->new and filter postings.
        let mut remap = vec![u32::MAX; self.ids.len()];
        let mut new_len = Vec::with_capacity(self.live);
        let mut new_ids = Vec::with_capacity(self.live);
        for (old, id) in self.ids.iter().enumerate() {
            if let Some(id) = id {
                let new = new_ids.len() as u32;
                remap[old] = new;
                rebuilt.lookup.insert(id.clone(), new);
                new_len.push(self.len[old]);
                new_ids.push(Some(id.clone()));
            }
        }
        for (term, plist) in &self.postings {
            let kept: Vec<Posting> = plist
                .iter()
                .filter(|p| remap[p.ord as usize] != u32::MAX)
                .map(|p| Posting {
                    ord: remap[p.ord as usize],
                    tf: p.tf,
                })
                .collect();
            if !kept.is_empty() {
                rebuilt.postings.insert(term.clone(), kept);
            }
        }
        rebuilt.len = new_len;
        rebuilt.ids = new_ids;
        rebuilt.live = self.live;
        rebuilt.total_len = self.total_len;
        *self = rebuilt;
    }

    /// BM25-rank documents for `query`, returning up to `k` `(id, score)` pairs
    /// best first. Scores are BM25 sums (higher = more relevant); documents
    /// sharing no query term are not returned.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        if k == 0 || self.live == 0 {
            return vec![];
        }
        let avgdl = self.total_len as f32 / self.live as f32;
        let terms = tokenize(query);
        // De-duplicate query terms; a repeated query term shouldn't double-count.
        let mut seen = std::collections::HashSet::new();

        let mut scores: HashMap<u32, f32> = HashMap::new();
        for term in &terms {
            if !seen.insert(term.clone()) {
                continue;
            }
            let Some(plist) = self.postings.get(term) else {
                continue;
            };
            // Live document frequency for this term.
            let n_t = plist
                .iter()
                .filter(|p| self.ids[p.ord as usize].is_some())
                .count();
            if n_t == 0 {
                continue;
            }
            let idf = (1.0 + (self.live as f32 - n_t as f32 + 0.5) / (n_t as f32 + 0.5)).ln();
            for p in plist {
                if self.ids[p.ord as usize].is_none() {
                    continue;
                }
                let dl = self.len[p.ord as usize] as f32;
                let tf = p.tf as f32;
                let denom = tf + K1 * (1.0 - B + B * dl / avgdl);
                scores.entry(p.ord).or_insert(0.0);
                *scores.get_mut(&p.ord).unwrap() += idf * (tf * (K1 + 1.0)) / denom;
            }
        }

        let mut hits: Vec<(String, f32)> = scores
            .into_iter()
            .filter_map(|(ord, s)| self.ids[ord as usize].clone().map(|id| (id, s)))
            .collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hits.truncate(k);
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx() -> LexicalIndex {
        let mut ix = LexicalIndex::new();
        ix.add("d1", "the quick brown fox jumps over the lazy dog");
        ix.add("d2", "a fast auburn fox leaps above a sleepy hound");
        ix.add("d3", "rust systems programming language performance");
        ix.add("d4", "the dog barks and the fox runs quick");
        ix
    }

    #[test]
    fn ranks_exact_terms() {
        let ix = idx();
        // "fox" appears in d1, d2, d4 — d3 must not appear.
        let hits = ix.search("fox", 10);
        let ids: Vec<&str> = hits.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"d1") && ids.contains(&"d2") && ids.contains(&"d4"));
        assert!(!ids.contains(&"d3"));
    }

    #[test]
    fn rare_term_outranks_common() {
        let ix = idx();
        // "rust" is unique to d3; a rare term has high IDF, so d3 ranks first.
        let hits = ix.search("rust programming quick", 4);
        assert_eq!(hits[0].0, "d3");
    }

    #[test]
    fn no_shared_terms_returns_empty() {
        let ix = idx();
        assert!(ix.search("kubernetes helm chart", 5).is_empty());
    }

    #[test]
    fn remove_then_absent_and_stats_correct() {
        let mut ix = idx();
        assert!(ix.remove("d3"));
        assert!(!ix.remove("d3"));
        assert_eq!(ix.len(), 3);
        // d3's unique term now matches nothing.
        assert!(ix.search("rust", 5).is_empty());
        // Other queries still work and never surface d3.
        let hits = ix.search("fox dog", 10);
        assert!(hits.iter().all(|(id, _)| id != "d3"));
    }

    #[test]
    fn update_replaces_text() {
        let mut ix = idx();
        ix.add("d3", "now about databases and indexing"); // replaces
        assert_eq!(ix.len(), 4);
        assert!(ix.search("rust", 5).is_empty());
        assert_eq!(ix.search("databases", 5)[0].0, "d3");
    }

    #[test]
    fn compact_preserves_ranking() {
        let mut ix = idx();
        let before = ix.search("fox dog quick", 10);
        ix.remove("d2");
        ix.compact();
        assert_eq!(ix.len(), 3);
        // Ranking among survivors is unchanged by compaction.
        let after = ix.search("fox dog quick", 10);
        let before_live: Vec<&String> = before
            .iter()
            .map(|(id, _)| id)
            .filter(|id| *id != "d2")
            .collect();
        let after_ids: Vec<&String> = after.iter().map(|(id, _)| id).collect();
        assert_eq!(before_live, after_ids);
    }
}
