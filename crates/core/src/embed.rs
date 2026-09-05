//! Embedding backends.
//!
//! The [`Embedder`] trait decouples the search engine from any particular model,
//! which is what makes the library modular: the flat index, persistence, and CLI
//! all operate on `Vec<f32>` and never see the model.
//!
//! Two implementations ship in-tree:
//! - [`MockEmbedder`] — a deterministic, dependency-free embedder used for tests
//!   and for exercising the full pipeline without model weights.
//! - `OnnxEmbedder` — real bi-encoder inference via ONNX Runtime, compiled only
//!   under the `onnx` feature. Which model it runs is data, not code: a
//!   [`ModelSpec`] (`model.json`) supplies pooling, token limit, prefixes,
//!   dimensionality and identity.

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

    /// Embed a single string **as a document** — the side of an asymmetric
    /// model that is stored in the index.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of documents. The default routes through
    /// [`Embedder::embed`]; backends with real batching (e.g. ONNX) should
    /// override this.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Embed a single string **as a query**.
    ///
    /// Asymmetric models (`e5`, `nomic`, and `bge` with its query instruction)
    /// need the query and the document embedded differently; symmetric ones do
    /// not. The default makes every symmetric backend correct without changes,
    /// so only backends that actually distinguish the two override this.
    ///
    /// The asymmetry is a property of the **store**, not of a call: documents
    /// embedded with one convention must be queried with the matching one for
    /// the life of the index. [`ModelSpec::model_id`] folds the prefixes into
    /// the identity so a store built under one convention cannot be opened
    /// under another.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }

    /// Embed a batch of queries. See [`Embedder::embed_query`].
    fn embed_query_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_query(t)).collect()
    }
}

/// Lets a boxed, runtime-selected embedder be used anywhere an `Embedder` is
/// expected (e.g. the CLI choosing mock vs ONNX at startup).
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
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_query(text)
    }
    fn embed_query_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_query_batch(texts)
    }
}

/// How a transformer's token vectors are collapsed into one sentence vector.
///
/// Not a tuning knob — it is fixed by how the model was trained, and getting it
/// wrong degrades the vectors silently. `all-MiniLM-L6-v2` and `e5` mean-pool;
/// `bge-*` reads the `[CLS]` token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    /// Mean over unmasked token vectors.
    Mean,
    /// The first (`[CLS]`) token vector.
    Cls,
}

fn default_true() -> bool {
    true
}

/// Everything about a bi-encoder that changes the vectors it produces.
///
/// Read from a `model.json` sidecar sitting beside `model.onnx` and
/// `tokenizer.json`. This exists because the previous code hardcoded pooling,
/// identity and token limit for one specific model, so pointing `--model` at a
/// *different* model produced wrong vectors under the right model's name — with
/// no error anywhere, and with the store's identity check waved through.
///
/// Every field here feeds the vectors, which is why every field feeds
/// [`ModelSpec::model_id`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelSpec {
    /// Human-readable model name, e.g. `bge-small-en-v1.5-int8`. Forms the
    /// readable half of the model id.
    pub name: String,
    /// Output dimensionality. Checked against the model's actual output.
    pub dim: usize,
    pub pooling: Pooling,
    /// L2-normalize the pooled vector. True for every model shipped here;
    /// present because it is a property of the model, not an assumption.
    #[serde(default = "default_true")]
    pub normalize: bool,
    /// Tokenizer truncation length, from the model's own reference pipeline
    /// (`sentence-transformers`' `max_seq_length`).
    ///
    /// Enforced rather than documented. Measured on the released v0.3.1
    /// binary, an untruncated ~4,800-token input does not fail — the ONNX
    /// export accepts it — it just produces a vector the model was never
    /// trained to produce, and a measurably different one (cosine 0.230 vs
    /// 0.192 against a fixed query). Silently different is the harder failure
    /// to notice, and the only thing bounding input length before was a
    /// character cap in one caller.
    pub max_tokens: usize,
    /// Prepended when embedding a query. `None` for symmetric models.
    #[serde(default)]
    pub query_prefix: Option<String>,
    /// Prepended when embedding a document. `None` for symmetric models.
    #[serde(default)]
    pub document_prefix: Option<String>,
}

impl ModelSpec {
    /// Parse a `model.json`.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let spec: Self = serde_json::from_slice(bytes).map_err(|e| {
            crate::error::Error::embed(format!("model.json is not a valid ModelSpec: {e}"))
        })?;
        spec.validate()?;
        Ok(spec)
    }

    fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(crate::error::Error::embed(
                "model.json: name must not be empty",
            ));
        }
        if self.dim == 0 {
            return Err(crate::error::Error::embed(
                "model.json: dim must be positive",
            ));
        }
        if self.max_tokens == 0 {
            return Err(crate::error::Error::embed(
                "model.json: max_tokens must be positive",
            ));
        }
        Ok(())
    }

    /// Store identity: `<name>.<8 hex of spec hash>`.
    ///
    /// The suffix is not decoration. Consumers invalidate their caches by
    /// comparing this string and nothing else, so a friendly name that survived
    /// a pooling or prefix change would let a store built one way be queried
    /// another way. Hashing every field makes that impossible: change anything
    /// that moves the vectors and the id moves with it.
    pub fn model_id(&self) -> String {
        let canonical = format!(
            "{}|{}|{:?}|{}|{}|{}|{}",
            self.name,
            self.dim,
            self.pooling,
            self.normalize,
            self.max_tokens,
            self.query_prefix.as_deref().unwrap_or(""),
            self.document_prefix.as_deref().unwrap_or(""),
        );
        format!("{}.{:08x}", self.name, fnv1a(canonical.as_bytes()) as u32)
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
    let norm = crate::simd::dot(v, v).sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbedder;

#[cfg(feature = "onnx")]
mod onnx {
    use super::{l2_normalize, Embedder, ModelSpec, Pooling};
    use crate::error::{Error, Result};
    use ndarray::{Array2, Axis};
    use ort::session::Session;
    use ort::value::Value;
    use std::path::Path;
    use std::sync::Mutex;
    use tokenizers::{Tokenizer, TruncationParams};

    /// Bi-encoder embeddings via ONNX Runtime, driven entirely by a
    /// [`ModelSpec`].
    ///
    /// Nothing here is specific to any one model: pooling, token limit,
    /// prefixes, dimensionality and identity all come from the spec, so a
    /// different model is a different `model.json` rather than a different
    /// code path. That is deliberate — the previous version hardcoded MiniLM's
    /// mean pooling and MiniLM's identity, which meant pointing `--model` at a
    /// CLS-pooled model produced quietly wrong vectors under a name claiming
    /// otherwise.
    ///
    /// The `Session` is behind a `Mutex` because ONNX Runtime's `run` takes
    /// `&mut self`, while [`Embedder`] (and `Send + Sync` sharing) needs `&self`.
    pub struct OnnxEmbedder {
        session: Mutex<Session>,
        tokenizer: Tokenizer,
        spec: ModelSpec,
        /// Cached because [`Embedder::model_id`] returns a borrow.
        model_id: String,
    }

    impl OnnxEmbedder {
        /// Build from the weights compiled into the binary.
        ///
        /// See `crates/core/models/` — `build.rs`/`include_bytes!` supply these.
        pub fn from_bundled() -> Result<Self> {
            let spec = ModelSpec::from_json(bundled::MODEL_JSON)?;
            Self::from_bytes(bundled::MODEL_ONNX, bundled::TOKENIZER_JSON, spec)
        }

        /// Build from an on-disk model directory holding `model.onnx`,
        /// `tokenizer.json` and `model.json` (the `--model <dir>` path).
        ///
        /// A missing `model.json` is an error rather than a fallback to some
        /// default shape. Guessing is precisely the bug this type exists to
        /// remove: a wrong guess is not visible in any output, only in worse
        /// search results.
        pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
            let dir = dir.as_ref();
            let spec_path = dir.join("model.json");
            let spec_bytes = std::fs::read(&spec_path).map_err(|e| {
                Error::embed(format!(
                    "{}: {e} — a model directory must describe its model \
                     (pooling, max_tokens, prefixes); see crates/core/models/model.json",
                    spec_path.display()
                ))
            })?;
            let spec = ModelSpec::from_json(&spec_bytes)?;
            let model = std::fs::read(dir.join("model.onnx"))?;
            let tok = std::fs::read(dir.join("tokenizer.json"))?;
            Self::from_bytes(&model, &tok, spec)
        }

        fn from_bytes(model: &[u8], tokenizer_json: &[u8], spec: ModelSpec) -> Result<Self> {
            let session = Session::builder()
                .map_err(Error::embed)?
                .commit_from_memory(model)
                .map_err(Error::embed)?;
            let mut tokenizer = Tokenizer::from_bytes(tokenizer_json).map_err(Error::embed)?;
            // Enforce the model's own sequence limit here rather than trusting
            // every caller to cap its input. Overrunning it is not an error the
            // ONNX export raises — it quietly returns a vector pooled over more
            // tokens than the model was trained on — and a downstream chunker's
            // character cap is not a property this library gets to assume.
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: spec.max_tokens,
                    ..Default::default()
                }))
                .map_err(Error::embed)?;
            let model_id = spec.model_id();
            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
                spec,
                model_id,
            })
        }

        /// The spec this embedder was built from.
        pub fn spec(&self) -> &ModelSpec {
            &self.spec
        }

        /// Embed `texts` with `prefix` prepended to each.
        ///
        /// The query/document split is only ever this prefix — the weights and
        /// pooling are identical for both sides.
        fn embed_with(&self, texts: &[String], prefix: Option<&str>) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            let prepared: Vec<String> = match prefix {
                Some(p) => texts.iter().map(|t| format!("{p}{t}")).collect(),
                None => texts.to_vec(),
            };
            let encodings = self
                .tokenizer
                .encode_batch(prepared, true)
                .map_err(Error::embed)?;
            let batch = encodings.len();
            let seq = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

            let mut ids = Array2::<i64>::zeros((batch, seq));
            let mut mask = Array2::<i64>::zeros((batch, seq));
            // token_type_ids are all zeros for single-sentence BERT input.
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
            if hidden != self.spec.dim {
                return Err(Error::embed(format!(
                    "model '{}' produced {hidden}-d token vectors but model.json declares dim {}",
                    self.spec.name, self.spec.dim
                )));
            }
            let token_embeds = Array2::from_shape_vec((batch * seq, hidden), data.to_vec())
                .map_err(Error::embed)?;

            let mut out = Vec::with_capacity(batch);
            for b in 0..batch {
                let mut acc = vec![0f32; hidden];
                match self.spec.pooling {
                    // Mean over unmasked tokens.
                    Pooling::Mean => {
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
                    }
                    // The [CLS] vector, which is always position 0 and always
                    // unmasked — right-padding never displaces it.
                    Pooling::Cls => {
                        let row = token_embeds.index_axis(Axis(0), b * seq);
                        for (a, &x) in acc.iter_mut().zip(row.iter()) {
                            *a = x;
                        }
                    }
                }
                if self.spec.normalize {
                    l2_normalize(&mut acc);
                }
                out.push(acc);
            }
            Ok(out)
        }
    }

    impl Embedder for OnnxEmbedder {
        fn dim(&self) -> usize {
            self.spec.dim
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }

        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.embed_batch(&[text.to_string()])?.remove(0))
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.embed_with(texts, self.spec.document_prefix.as_deref())
        }

        fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.embed_query_batch(&[text.to_string()])?.remove(0))
        }

        fn embed_query_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.embed_with(texts, self.spec.query_prefix.as_deref())
        }
    }

    /// Model bytes embedded at compile time via `include_bytes!`.
    ///
    /// The paths are resolved against the crate root (`CARGO_MANIFEST_DIR`) so
    /// they don't depend on this file's location. Until real weights are dropped
    /// into `crates/core/models/`, the committed placeholders are empty and
    /// [`super::OnnxEmbedder::from_bundled`] fails at session-build time with a
    /// clear runtime error rather than breaking compilation. `model.json` is
    /// small and always committed, so it is never a placeholder.
    mod bundled {
        pub const MODEL_ONNX: &[u8] =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/model.onnx"));
        pub const TOKENIZER_JSON: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/tokenizer.json"
        ));
        pub const MODEL_JSON: &[u8] =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/model.json"));
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    fn base() -> ModelSpec {
        ModelSpec {
            name: "demo-v1".into(),
            dim: 384,
            pooling: Pooling::Mean,
            normalize: true,
            max_tokens: 256,
            query_prefix: None,
            document_prefix: None,
        }
    }

    #[test]
    fn model_id_keeps_the_name_readable() {
        let id = base().model_id();
        assert!(id.starts_with("demo-v1."), "unexpected id: {id}");
        assert_eq!(id.len(), "demo-v1.".len() + 8);
    }

    #[test]
    fn model_id_is_stable_for_an_unchanged_spec() {
        assert_eq!(base().model_id(), base().model_id());
    }

    /// The invariant the whole design rests on: consumers invalidate their
    /// caches by comparing `model_id` and nothing else, so *every* field that
    /// moves the vectors must move the id. A field added here without being
    /// added to `model_id`'s canonical string would let a store built one way
    /// be queried another way, silently.
    #[test]
    fn every_vector_affecting_field_changes_the_model_id() {
        let baseline = base().model_id();

        let mut pooling = base();
        pooling.pooling = Pooling::Cls;
        assert_ne!(baseline, pooling.model_id(), "pooling must change the id");

        let mut normalize = base();
        normalize.normalize = false;
        assert_ne!(
            baseline,
            normalize.model_id(),
            "normalize must change the id"
        );

        let mut tokens = base();
        tokens.max_tokens = 512;
        assert_ne!(baseline, tokens.model_id(), "max_tokens must change the id");

        let mut query = base();
        query.query_prefix = Some("query: ".into());
        assert_ne!(
            baseline,
            query.model_id(),
            "query_prefix must change the id"
        );

        let mut document = base();
        document.document_prefix = Some("passage: ".into());
        assert_ne!(
            baseline,
            document.model_id(),
            "document_prefix must change the id"
        );

        let mut dim = base();
        dim.dim = 768;
        assert_ne!(baseline, dim.model_id(), "dim must change the id");

        let mut name = base();
        name.name = "demo-v2".into();
        assert_ne!(baseline, name.model_id(), "name must change the id");
    }

    /// Swapping which side carries a prefix is a different model, even though
    /// the same strings appear in the spec.
    #[test]
    fn prefix_sides_are_not_interchangeable() {
        let mut a = base();
        a.query_prefix = Some("x".into());
        let mut b = base();
        b.document_prefix = Some("x".into());
        assert_ne!(a.model_id(), b.model_id());
    }

    #[test]
    fn parses_a_minimal_spec_and_defaults_normalize_on() {
        let spec =
            ModelSpec::from_json(br#"{"name":"m","dim":384,"pooling":"cls","max_tokens":512}"#)
                .unwrap();
        assert_eq!(spec.pooling, Pooling::Cls);
        assert!(spec.normalize);
        assert!(spec.query_prefix.is_none());
    }

    #[test]
    fn rejects_a_spec_that_cannot_describe_a_model() {
        for bad in [
            br#"{"name":"","dim":384,"pooling":"mean","max_tokens":256}"#.as_slice(),
            br#"{"name":"m","dim":0,"pooling":"mean","max_tokens":256}"#.as_slice(),
            br#"{"name":"m","dim":384,"pooling":"mean","max_tokens":0}"#.as_slice(),
            // An unknown pooling is refused rather than silently defaulting —
            // a wrong pooling is invisible except as worse results.
            br#"{"name":"m","dim":384,"pooling":"maxpool","max_tokens":256}"#.as_slice(),
            br#"{"name":"m","dim":384,"max_tokens":256}"#.as_slice(),
        ] {
            assert!(
                ModelSpec::from_json(bad).is_err(),
                "should have been rejected: {}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    /// The bundled spec must parse, or `from_bundled` fails at runtime in a
    /// build where nothing else would catch it.
    #[test]
    fn bundled_model_json_is_valid() {
        let bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/model.json"));
        let spec = ModelSpec::from_json(bytes).expect("bundled models/model.json must be valid");
        assert_eq!(spec.dim, 384);
    }
}
