//! Embedding backends.
//!
//! The [`Embedder`] trait decouples the search engine from any particular model,
//! which is what makes the library modular: the flat index, persistence, and CLI
//! all operate on `Vec<f32>` and never see the model.
//!
//! Two implementations ship in-tree:
//! - [`MockEmbedder`] — a deterministic, dependency-free embedder used for tests
//!   and for exercising the full pipeline without model weights.
//! - `MiniLmEmbedder` — real `all-MiniLM-L6-v2` inference via ONNX Runtime,
//!   compiled only under the `onnx` feature.

use crate::error::Result;

/// Anything that turns text into a fixed-length vector.
///
/// Implementations must be deterministic for a given input and must always
/// return vectors of length [`Embedder::dim`].
pub trait Embedder: Send + Sync {
    /// Dimensionality of the produced vectors.
    fn dim(&self) -> usize;

    /// A short identifier for the backing model (persisted in the manifest so a
    /// store can be checked against the embedder that created it).
    fn model_id(&self) -> &str;

    /// Embed a single string.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch. The default routes through [`Embedder::embed`]; backends
    /// with real batching (e.g. ONNX) should override this.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// Lets a boxed, runtime-selected embedder be used anywhere an `Embedder` is
/// expected (e.g. the CLI choosing mock vs MiniLM at startup).
impl Embedder for Box<dyn Embedder> {
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn model_id(&self) -> &str {
        (**self).model_id()
    }
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed(text)
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_batch(texts)
    }
}

/// A deterministic, hashing-based embedder with no external dependencies.
///
/// It is **not** semantically meaningful — it exists so the indexing, querying,
/// persistence, and daemon paths can be built and tested without downloading a
/// model. Same text in, same vector out; different text, near-orthogonal vector.
#[derive(Debug, Clone)]
pub struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    /// Create a mock embedder producing `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "embedding dim must be positive");
        Self { dim }
    }
}

impl Default for MockEmbedder {
    /// 384-d to mirror `all-MiniLM-L6-v2`, so stores are drop-in swappable.
    fn default() -> Self {
        Self::new(384)
    }
}

impl Embedder for MockEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        "mock-hash-v1"
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Bag-of-tokens hashing: each token contributes a signed spike to a few
        // dimensions. This makes shared tokens pull vectors together, which is
        // enough structure for tests to assert "closer than unrelated text".
        let mut v = vec![0f32; self.dim];
        for token in tokenize(text) {
            let h = fnv1a(token.as_bytes());
            // Spread each token across 4 dimensions with alternating sign.
            for i in 0..4 {
                let hh = h.wrapping_mul(0x100000001b3).wrapping_add(i as u64);
                let idx = (hh % self.dim as u64) as usize;
                let sign = if (hh >> 33) & 1 == 0 { 1.0 } else { -1.0 };
                v[idx] += sign;
            }
        }
        l2_normalize(&mut v);
        Ok(v)
    }
}

/// Lowercase whitespace/punctuation tokenizer good enough for the mock backend.
fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
}

/// 64-bit FNV-1a hash.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Normalize a vector to unit L2 length in place. Zero vectors are left as-is.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(feature = "onnx")]
pub use minilm::MiniLmEmbedder;

#[cfg(feature = "onnx")]
mod minilm {
    use super::{l2_normalize, Embedder};
    use crate::error::{Error, Result};
    use ndarray::{Array2, Axis};
    use ort::session::Session;
    use ort::value::Value;
    use std::path::Path;
    use std::sync::Mutex;
    use tokenizers::Tokenizer;

    /// `all-MiniLM-L6-v2` embeddings via ONNX Runtime.
    ///
    /// Produces 384-d, L2-normalized, mean-pooled embeddings matching the
    /// `sentence-transformers` reference pipeline.
    ///
    /// The `Session` is behind a `Mutex` because ONNX Runtime's `run` takes
    /// `&mut self`, while [`Embedder`] (and `Send + Sync` sharing) needs `&self`.
    pub struct MiniLmEmbedder {
        session: Mutex<Session>,
        tokenizer: Tokenizer,
        dim: usize,
    }

    impl MiniLmEmbedder {
        /// Build from the bundled weights compiled into the binary.
        ///
        /// See `crates/core/models/` — `build.rs`/`include_bytes!` supply these.
        pub fn from_bundled() -> Result<Self> {
            Self::from_bytes(bundled::MODEL_ONNX, bundled::TOKENIZER_JSON)
        }

        /// Build from an on-disk model directory containing `model.onnx` and
        /// `tokenizer.json` (the `--model <path>` override path).
        pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
            let dir = dir.as_ref();
            let model = std::fs::read(dir.join("model.onnx"))?;
            let tok = std::fs::read(dir.join("tokenizer.json"))?;
            Self::from_bytes(&model, &tok)
        }

        fn from_bytes(model: &[u8], tokenizer_json: &[u8]) -> Result<Self> {
            let session = Session::builder()
                .map_err(Error::embed)?
                .commit_from_memory(model)
                .map_err(Error::embed)?;
            let tokenizer = Tokenizer::from_bytes(tokenizer_json).map_err(Error::embed)?;
            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
                dim: 384,
            })
        }
    }

    impl Embedder for MiniLmEmbedder {
        fn dim(&self) -> usize {
            self.dim
        }

        fn model_id(&self) -> &str {
            "all-MiniLM-L6-v2-int8"
        }

        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.embed_batch(&[text.to_string()])?.remove(0))
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            let encodings = self
                .tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(Error::embed)?;
            let batch = encodings.len();
            let seq = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

            let mut ids = Array2::<i64>::zeros((batch, seq));
            let mut mask = Array2::<i64>::zeros((batch, seq));
            // token_type_ids are all zeros for single-sentence MiniLM input.
            let types = Array2::<i64>::zeros((batch, seq));
            for (r, enc) in encodings.iter().enumerate() {
                for (c, (&id, &m)) in enc
                    .get_ids()
                    .iter()
                    .zip(enc.get_attention_mask())
                    .enumerate()
                {
                    ids[[r, c]] = id as i64;
                    mask[[r, c]] = m as i64;
                }
            }

            let inputs = ort::inputs![
                "input_ids" => Value::from_array(ids.clone()).map_err(Error::embed)?,
                "attention_mask" => Value::from_array(mask.clone()).map_err(Error::embed)?,
                "token_type_ids" => Value::from_array(types).map_err(Error::embed)?,
            ];

            let mut session = self
                .session
                .lock()
                .map_err(|_| Error::embed("embedding session mutex poisoned"))?;
            let outputs = session.run(inputs).map_err(Error::embed)?;
            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(Error::embed)?;
            // Expect [batch, seq, hidden].
            let hidden = *shape.last().unwrap() as usize;
            let token_embeds = Array2::from_shape_vec((batch * seq, hidden), data.to_vec())
                .map_err(Error::embed)?;

            // Mean-pool over tokens using the attention mask, then normalize.
            let mut out = Vec::with_capacity(batch);
            for b in 0..batch {
                let mut acc = vec![0f32; hidden];
                let mut count = 0f32;
                for s in 0..seq {
                    if mask[[b, s]] == 0 {
                        continue;
                    }
                    let row = token_embeds.index_axis(Axis(0), b * seq + s);
                    for (a, &x) in acc.iter_mut().zip(row.iter()) {
                        *a += x;
                    }
                    count += 1.0;
                }
                if count > 0.0 {
                    for a in acc.iter_mut() {
                        *a /= count;
                    }
                }
                l2_normalize(&mut acc);
                out.push(acc);
            }
            Ok(out)
        }
    }

    /// Model bytes embedded at compile time via `include_bytes!`.
    ///
    /// The paths are resolved against the crate root (`CARGO_MANIFEST_DIR`) so
    /// they don't depend on this file's location. Until real weights are dropped
    /// into `crates/core/models/`, the committed placeholders are empty and
    /// [`super::MiniLmEmbedder::from_bundled`] fails at session-build time with a
    /// clear runtime error rather than breaking compilation.
    mod bundled {
        pub const MODEL_ONNX: &[u8] =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/model.onnx"));
        pub const TOKENIZER_JSON: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/tokenizer.json"
        ));
    }
}
