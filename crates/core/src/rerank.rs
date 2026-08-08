//! Cross-encoder reranking.
//!
//! A bi-encoder embeds query and passage independently, so their vectors never
//! see each other and the score is a similarity between two summaries. A
//! cross-encoder runs the pair *through one model together*, letting every
//! query token attend to every passage token. That is strictly more
//! informative and strictly more expensive: it cannot be precomputed or
//! indexed, so it only makes sense over a shortlist a cheaper retriever has
//! already produced.
//!
//! That is exactly the shape of the problem it is here for. In hoocode's
//! 62-query retrieval eval the gold span reaches the fused top-50 about 85% of
//! the time but the top-10 only about 72% — roughly an eighth of the set is
//! retrieved and then buried. No amount of extra recall fixes that; it is a
//! ranking problem over candidates already in hand.
//!
//! Scores are raw logits: higher is more relevant, but they are not
//! probabilities and are only comparable within one call's candidate set.

use crate::error::Result;

/// Scores `(query, passage)` pairs jointly.
pub trait CrossEncoder: Send + Sync {
    /// Identifies the model behind a score, for provenance.
    fn model_id(&self) -> &str;

    /// Relevance logits, one per passage, in the order given.
    fn score(&self, query: &str, passages: &[String]) -> Result<Vec<f32>>;
}

/// A passage paired with the id its score belongs to.
#[derive(Debug, Clone)]
pub struct RerankCandidate {
    pub id: String,
    pub text: String,
}

/// A scored candidate.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RerankResult {
    pub id: String,
    pub score: f32,
}

/// Score `candidates` against `query` and return the top `k`, best first.
///
/// Ties break by id so the ordering is deterministic — the same inputs must
/// produce the same context in an agent harness, or debugging and evals pick
/// up noise that has nothing to do with retrieval.
pub fn rerank_candidates(
    encoder: &dyn CrossEncoder,
    query: &str,
    candidates: &[RerankCandidate],
    k: usize,
) -> Result<Vec<RerankResult>> {
    if candidates.is_empty() || k == 0 {
        return Ok(vec![]);
    }
    let passages: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
    let scores = encoder.score(query, &passages)?;

    let mut scored: Vec<RerankResult> = candidates
        .iter()
        .zip(scores)
        .map(|(c, score)| RerankResult {
            id: c.id.clone(),
            score,
        })
        .collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    scored.truncate(k);
    Ok(scored)
}

#[cfg(feature = "onnx")]
pub use onnx::MiniLmCrossEncoder;

#[cfg(feature = "onnx")]
mod onnx {
    use super::{CrossEncoder, Result};
    use crate::error::Error;
    use ndarray::Array2;
    use ort::session::Session;
    use ort::value::Value;
    use std::path::Path;
    use std::sync::Mutex;
    use tokenizers::Tokenizer;

    /// Sequence length cap for a (query, passage) pair.
    ///
    /// The pair is truncated to fit, which is why passages should be candidate
    /// windows rather than whole files: past this the tail is simply unseen.
    const MAX_PAIR_TOKENS: usize = 512;

    /// `ms-marco-MiniLM-L-6-v2` relevance scoring via ONNX Runtime.
    ///
    /// Mirrors [`MiniLmEmbedder`](crate::MiniLmEmbedder): weights bundled at
    /// compile time, `Session` behind a `Mutex` because ONNX Runtime's `run`
    /// takes `&mut self` while the trait needs `&self`.
    pub struct MiniLmCrossEncoder {
        session: Mutex<Session>,
        tokenizer: Tokenizer,
    }

    impl MiniLmCrossEncoder {
        /// Build from the reranker weights compiled into the binary.
        pub fn from_bundled() -> Result<Self> {
            Self::from_bytes(bundled::MODEL_ONNX, bundled::TOKENIZER_JSON)
        }

        /// Build from a directory holding `model.onnx` + `tokenizer.json`.
        pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
            let dir = dir.as_ref();
            let model = std::fs::read(dir.join("model.onnx"))?;
            let tok = std::fs::read(dir.join("tokenizer.json"))?;
            Self::from_bytes(&model, &tok)
        }

        fn from_bytes(model: &[u8], tokenizer_json: &[u8]) -> Result<Self> {
            if model.is_empty() {
                return Err(Error::embed(
                    "no reranker weights bundled (released binaries ship without them); \
                     start the daemon with --reranker-model <dir>, or rebuild after \
                     scripts/fetch-model.sh --with-reranker",
                ));
            }
            let session = Session::builder()
                .map_err(Error::embed)?
                .commit_from_memory(model)
                .map_err(Error::embed)?;
            let mut tokenizer = Tokenizer::from_bytes(tokenizer_json).map_err(Error::embed)?;
            // A pair that overflows the model's positions would otherwise fail
            // at run time with an opaque shape error.
            let truncation = tokenizers::TruncationParams {
                max_length: MAX_PAIR_TOKENS,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                ..Default::default()
            };
            tokenizer
                .with_truncation(Some(truncation))
                .map_err(Error::embed)?;
            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
            })
        }
    }

    impl CrossEncoder for MiniLmCrossEncoder {
        fn model_id(&self) -> &str {
            "ms-marco-MiniLM-L-6-v2"
        }

        fn score(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
            if passages.is_empty() {
                return Ok(vec![]);
            }
            // Encoded as a *pair*, so token_type_ids separate query from
            // passage — that segment boundary is what makes this a cross
            // encoder rather than two independent encodings.
            let pairs: Vec<(String, String)> = passages
                .iter()
                .map(|p| (query.to_string(), p.clone()))
                .collect();
            let encodings = self
                .tokenizer
                .encode_batch(pairs, true)
                .map_err(Error::embed)?;

            let batch = encodings.len();
            let seq = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
            if seq == 0 {
                return Ok(vec![0.0; batch]);
            }

            let mut ids = Array2::<i64>::zeros((batch, seq));
            let mut mask = Array2::<i64>::zeros((batch, seq));
            let mut types = Array2::<i64>::zeros((batch, seq));
            for (r, enc) in encodings.iter().enumerate() {
                let type_ids = enc.get_type_ids();
                for (c, (&id, &m)) in enc
                    .get_ids()
                    .iter()
                    .zip(enc.get_attention_mask())
                    .enumerate()
                {
                    ids[[r, c]] = id as i64;
                    mask[[r, c]] = m as i64;
                    types[[r, c]] = *type_ids.get(c).unwrap_or(&0) as i64;
                }
            }

            let inputs = ort::inputs![
                "input_ids" => Value::from_array(ids).map_err(Error::embed)?,
                "attention_mask" => Value::from_array(mask).map_err(Error::embed)?,
                "token_type_ids" => Value::from_array(types).map_err(Error::embed)?,
            ];

            let mut session = self
                .session
                .lock()
                .map_err(|_| Error::embed("reranker session mutex poisoned"))?;
            let outputs = session.run(inputs).map_err(Error::embed)?;
            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(Error::embed)?;

            // Sequence-classification heads vary: `ms-marco-MiniLM-L-6-v2` is
            // trained with a single regression logit ([batch, 1] or [batch]),
            // but binary-classifier rerankers emit [batch, 2] where the
            // relevant-class logit is index 1. Handle both rather than assume,
            // since guessing wrong silently inverts the ranking.
            let labels = match shape.len() {
                0 | 1 => 1,
                _ => *shape.last().unwrap() as usize,
            };
            if labels == 0 || data.len() < batch * labels {
                return Err(Error::embed(format!(
                    "reranker returned an unusable output shape {shape:?} for {batch} passages"
                )));
            }
            let relevant = if labels >= 2 { 1 } else { 0 };
            Ok((0..batch).map(|b| data[b * labels + relevant]).collect())
        }
    }

    /// Reranker weights embedded at compile time.
    ///
    /// Empty placeholders until `scripts/fetch-model.sh` runs, so the crate
    /// still compiles without them and fails with a clear runtime message —
    /// the same contract the embedder's bundled weights use.
    mod bundled {
        pub const MODEL_ONNX: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/reranker/model.onnx"
        ));
        pub const TOKENIZER_JSON: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/reranker/tokenizer.json"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic stand-in: scores by how many query words the passage
    /// contains, so ordering is predictable without a model.
    struct WordOverlapEncoder;

    impl CrossEncoder for WordOverlapEncoder {
        fn model_id(&self) -> &str {
            "word-overlap-test"
        }
        fn score(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
            let terms: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
            Ok(passages
                .iter()
                .map(|p| {
                    let lower = p.to_lowercase();
                    terms.iter().filter(|t| lower.contains(*t)).count() as f32
                })
                .collect())
        }
    }

    fn candidate(id: &str, text: &str) -> RerankCandidate {
        RerankCandidate {
            id: id.into(),
            text: text.into(),
        }
    }

    #[test]
    fn orders_by_score_and_truncates_to_k() {
        let candidates = vec![
            candidate("a", "nothing relevant here"),
            candidate("b", "quick brown fox"),
            candidate("c", "quick fox"),
        ];
        let out = rerank_candidates(&WordOverlapEncoder, "quick fox", &candidates, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, "b");
        assert_eq!(out[1].id, "c");
    }

    #[test]
    fn breaks_ties_by_id_so_ordering_is_deterministic() {
        let candidates = vec![
            candidate("z", "quick fox"),
            candidate("a", "quick fox"),
            candidate("m", "quick fox"),
        ];
        let ids: Vec<String> = rerank_candidates(&WordOverlapEncoder, "quick fox", &candidates, 3)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, vec!["a", "m", "z"]);
    }

    #[test]
    fn empty_input_and_zero_k_are_not_errors() {
        assert!(rerank_candidates(&WordOverlapEncoder, "q", &[], 5)
            .unwrap()
            .is_empty());
        let candidates = vec![candidate("a", "quick fox")];
        assert!(rerank_candidates(&WordOverlapEncoder, "q", &candidates, 0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn k_larger_than_the_candidate_set_returns_everything() {
        let candidates = vec![candidate("a", "quick"), candidate("b", "fox")];
        let out = rerank_candidates(&WordOverlapEncoder, "quick fox", &candidates, 99).unwrap();
        assert_eq!(out.len(), 2);
    }
}
