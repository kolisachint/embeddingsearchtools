//! Long-lived stdio daemon.
//!
//! This is the low-latency path for callers like a TypeScript `spawn`: the
//! process loads the model and index **once**, then answers newline-delimited
//! JSON (NDJSON) requests on stdin, one response per line on stdout. Keeping the
//! process alive amortizes model-load cost across every query.
//!
//! Protocol — one JSON object per line.
//!
//! Requests (`op` selects the operation):
//! - `{"op":"query","text":"...","k":5}`        → search (dense)
//! - `{"op":"query","text":"...","k":5,"retriever":"lexical"}` → BM25 only
//! - `{"op":"query","vector":[...],"k":5}`       → search a precomputed vector
//!   (`text` and `vector` are mutually exclusive; sending both is an error)
//! - `{"op":"add","id":"x","text":"..."}`        → insert
//! - `{"op":"update","id":"x","text":"..."}`     → replace
//! - `{"op":"upsert","id":"x","text":"..."}`     → insert-or-replace
//! - `{"op":"remove","id":"x"}`                  → delete
//! - `{"op":"bulk","items":[{"id","text"},..]}`  → batched upsert (one
//!   batched embedding inference; the fast path for bulk indexing)
//! - `{"op":"save"}`                             → persist to the store dir
//! - `{"op":"compact"}`                          → reclaim tombstoned rows
//! - `{"op":"count"}`                            → live vector count
//! - `{"op":"info"}`                             → model id, dim, count,
//!   and `rerank`: whether this binary can actually serve the `rerank` op
//! - `{"op":"rerank","query":"...","passages":[...],"k":10}` → cross-encoder
//!   reranking of caller-supplied passages
//! - `{"op":"ping"}`                             → readiness probe
//!
//! Responses always carry `ok`:
//! - `{"ok":true, ...}` with op-specific fields (`results`, `inserted`, ...)
//! - `{"ok":false,"error":"message"}`

use embsearch_core::rerank::{rerank_candidates, CrossEncoder, RerankCandidate, RerankResult};
use embsearch_core::{Database, Embedder, SearchResult};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Which retriever answers a `query`.
///
/// `Lexical` exists so a caller can fuse the retrievers itself: `Hybrid`
/// applies this crate's RRF with its own constant and returns one blended
/// list, discarding the per-retriever ranks and scores a caller needs for
/// n-way fusion, a different constant, or diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Retriever {
    /// Vector search over the embedding index.
    Dense,
    /// BM25 over the lexical index. Requires a hybrid store.
    Lexical,
    /// Both, fused by this crate's RRF. Requires a hybrid store.
    Hybrid,
}

/// One candidate to score, as sent by the caller.
#[derive(Debug, Deserialize)]
struct RerankPassage {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum Request {
    Query {
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        vector: Option<Vec<f32>>,
        #[serde(default = "default_k")]
        k: usize,
        /// Which retriever answers: `dense` (default), `lexical`, or `hybrid`.
        /// `lexical` and `hybrid` need a hybrid store and `text`.
        #[serde(default)]
        retriever: Option<Retriever>,
        /// Deprecated alias for `"retriever":"hybrid"`, kept so existing
        /// callers keep working. Setting both is only an error when they
        /// disagree.
        #[serde(default)]
        hybrid: bool,
    },
    /// Score `(query, passage)` pairs with the bundled cross-encoder and
    /// return the best `k`. Passages are supplied inline rather than looked up
    /// by id: the caller has the exact spans it intends to show, which is what
    /// should be scored, and this works against any store — including a
    /// non-hybrid one that keeps no texts at all.
    Rerank {
        query: String,
        #[serde(default)]
        passages: Vec<RerankPassage>,
        #[serde(default = "default_k")]
        k: usize,
    },
    Add {
        id: String,
        text: String,
    },
    Update {
        id: String,
        text: String,
    },
    Upsert {
        id: String,
        text: String,
    },
    Remove {
        id: String,
    },
    Bulk {
        items: Vec<BulkItem>,
    },
    Save,
    Compact,
    Count,
    Info,
    Ping,
}

#[derive(Debug, Deserialize)]
struct BulkItem {
    id: String,
    text: String,
}

fn default_k() -> usize {
    10
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Response {
    Ok(OkResponse),
    Err { ok: bool, error: String },
}

#[derive(Debug, Serialize)]
struct OkResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    results: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inserted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    removed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inserted_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    updated_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hybrid: Option<bool>,
    /// Whether `rerank` will actually work on this binary.
    ///
    /// Reported on `info` because the daemon's version no longer implies it:
    /// released binaries carry the `rerank` op but not the weights, so a
    /// caller that gated on version alone would offer a reranker that errors
    /// on first use. Only ever `Some` on `info`.
    #[serde(skip_serializing_if = "Option::is_none")]
    rerank: Option<bool>,
    /// Reranked ids with their cross-encoder logits. Distinct from `results`
    /// because the scores are not comparable to retrieval scores.
    #[serde(skip_serializing_if = "Option::is_none")]
    reranked: Option<Vec<RerankResult>>,
}

impl OkResponse {
    fn empty() -> Self {
        Self {
            ok: true,
            results: None,
            inserted: None,
            removed: None,
            count: None,
            inserted_count: None,
            updated_count: None,
            model_id: None,
            dim: None,
            index: None,
            hybrid: None,
            rerank: None,
            reranked: None,
        }
    }
}

impl Response {
    fn error(msg: impl std::fmt::Display) -> Self {
        Response::Err {
            ok: false,
            error: msg.to_string(),
        }
    }
}

/// Run the NDJSON request loop until stdin closes.
///
/// `store_dir` is where `save` writes. Reads from `input`, writes to `output`
/// (parameterized so the loop is unit-testable without real pipes).
/// Where to load cross-encoder weights from, when they are not bundled.
///
/// Set once from `--reranker-model` before the daemon loop starts; read on
/// first `rerank`. A `OnceLock` rather than a parameter because the encoder
/// itself is process-wide and lazily built, and threading a path through every
/// request would say the model can change per request, which it cannot.
static RERANKER_MODEL_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Record the `--reranker-model` directory. Call before [`run`]; later calls
/// are ignored, since the encoder is built once and kept hot.
pub fn set_reranker_model_dir(dir: Option<PathBuf>) {
    let _ = RERANKER_MODEL_DIR.set(dir);
}

/// The process-wide cross-encoder, built on first use.
///
/// Loading the model costs a second or so, so it is built once and kept hot
/// for the daemon's lifetime — the same reason `serve` exists at all.
///
/// Released binaries bundle the embedder but **not** a reranker: it is a
/// second ~23 MB model, and bundling it tripled the download for a reranker
/// that measured worse than a caller's own deterministic scorer on every query
/// class but one. So the usual path here is `--reranker-model <dir>`; a build
/// that ran `scripts/fetch-model.sh --with-reranker` still works with no flag.
/// A build with neither says so, rather than scoring with something that is
/// not a cross-encoder.
#[cfg(feature = "onnx")]
fn cross_encoder() -> std::result::Result<&'static dyn CrossEncoder, String> {
    use embsearch_core::rerank::MiniLmCrossEncoder;
    static ENCODER: OnceLock<std::result::Result<MiniLmCrossEncoder, String>> = OnceLock::new();
    let built = ENCODER.get_or_init(|| {
        match RERANKER_MODEL_DIR.get().and_then(|d| d.as_ref()) {
            Some(dir) => MiniLmCrossEncoder::from_dir(dir)
                .map_err(|e| format!("--reranker-model {}: {e}", dir.display())),
            // Falls back to bundled weights, which are an empty placeholder
            // unless this binary was built with --with-reranker.
            None => MiniLmCrossEncoder::from_bundled().map_err(|e| e.to_string()),
        }
    });
    match built {
        Ok(encoder) => Ok(encoder),
        Err(e) => Err(e.clone()),
    }
}

#[cfg(not(feature = "onnx"))]
fn cross_encoder() -> std::result::Result<&'static dyn CrossEncoder, String> {
    Err("rerank requires an onnx build; this binary has no cross-encoder".into())
}

pub fn run<E, R, W>(
    mut db: Database<E>,
    store_dir: Option<PathBuf>,
    input: R,
    mut output: W,
) -> std::io::Result<()>
where
    E: Embedder,
    R: BufRead,
    W: Write,
{
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(req) => handle(&mut db, store_dir.as_deref(), req),
            Err(e) => Response::error(format!("invalid request: {e}")),
        };
        let encoded = serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"ok":false,"error":"failed to encode response"}"#.into());
        output.write_all(encoded.as_bytes())?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

fn handle<E: Embedder>(
    db: &mut Database<E>,
    store_dir: Option<&std::path::Path>,
    req: Request,
) -> Response {
    match req {
        Request::Ping => Response::Ok(OkResponse::empty()),
        Request::Rerank { query, passages, k } => {
            let candidates: Vec<RerankCandidate> = passages
                .into_iter()
                .map(|p| RerankCandidate {
                    id: p.id,
                    text: p.text,
                })
                .collect();
            // Nothing to score is a valid question with an empty answer, and
            // answering it must not depend on whether this binary has weights.
            if candidates.is_empty() {
                let mut r = OkResponse::empty();
                r.reranked = Some(vec![]);
                return Response::Ok(r);
            }
            match cross_encoder() {
                Err(e) => Response::error(e),
                Ok(encoder) => match rerank_candidates(encoder, &query, &candidates, k) {
                    Ok(reranked) => {
                        let mut r = OkResponse::empty();
                        r.reranked = Some(reranked);
                        Response::Ok(r)
                    }
                    Err(e) => Response::error(e),
                },
            }
        }
        Request::Count => {
            let mut r = OkResponse::empty();
            r.count = Some(db.len());
            Response::Ok(r)
        }
        Request::Query {
            text,
            vector,
            k,
            retriever,
            hybrid,
        } => {
            // `hybrid: true` is the old spelling of `retriever: "hybrid"`.
            // Accept either, and reject only a genuine contradiction rather
            // than silently picking one.
            let retriever = match (retriever, hybrid) {
                (Some(r), true) if r != Retriever::Hybrid => {
                    return Response::error("`hybrid: true` contradicts the given `retriever`");
                }
                (Some(r), _) => r,
                (None, true) => Retriever::Hybrid,
                (None, false) => Retriever::Dense,
            };
            if retriever != Retriever::Dense && vector.is_some() {
                return Response::error("lexical and hybrid queries require `text`, not `vector`");
            }
            let result = match (text, vector) {
                (Some(_), Some(_)) => {
                    return Response::error("query accepts text or vector, not both");
                }
                (Some(t), None) => match retriever {
                    Retriever::Dense => db.query(&t, k),
                    Retriever::Lexical => db.query_lexical(&t, k),
                    Retriever::Hybrid => db.query_hybrid(&t, k),
                },
                (None, Some(v)) => db.query_vector(&v, k),
                (None, None) => {
                    return Response::error("query requires `text` or `vector`");
                }
            };
            match result {
                Ok(hits) => {
                    let mut r = OkResponse::empty();
                    r.results = Some(hits);
                    Response::Ok(r)
                }
                Err(e) => Response::error(e),
            }
        }
        Request::Add { id, text } => match db.add(&id, &text) {
            Ok(()) => Response::Ok(OkResponse::empty()),
            Err(e) => Response::error(e),
        },
        Request::Update { id, text } => match db.update(&id, &text) {
            Ok(()) => Response::Ok(OkResponse::empty()),
            Err(e) => Response::error(e),
        },
        Request::Upsert { id, text } => match db.upsert(&id, &text) {
            Ok(inserted) => {
                let mut r = OkResponse::empty();
                r.inserted = Some(inserted);
                Response::Ok(r)
            }
            Err(e) => Response::error(e),
        },
        Request::Remove { id } => match db.remove(&id) {
            Ok(removed) => {
                let mut r = OkResponse::empty();
                r.removed = Some(removed);
                Response::Ok(r)
            }
            Err(e) => Response::error(e),
        },
        Request::Bulk { items } => {
            let pairs = items.into_iter().map(|i| (i.id, i.text));
            match db.upsert_batch(pairs) {
                Ok((inserted, updated)) => {
                    let mut r = OkResponse::empty();
                    r.inserted_count = Some(inserted);
                    r.updated_count = Some(updated);
                    Response::Ok(r)
                }
                Err(e) => Response::error(e),
            }
        }
        Request::Compact => {
            db.compact();
            Response::Ok(OkResponse::empty())
        }
        Request::Info => {
            let mut r = OkResponse::empty();
            r.model_id = Some(db.embedder().model_id().to_string());
            r.dim = Some(db.embedder().dim());
            r.count = Some(db.len());
            r.index = Some(db.index_kind().to_string());
            r.hybrid = Some(db.is_hybrid());
            // Builds the encoder if it is not built yet — the one place where
            // paying that cost up front is right, since the answer is the
            // question being asked.
            r.rerank = Some(cross_encoder().is_ok());
            Response::Ok(r)
        }
        Request::Save => match store_dir {
            Some(dir) => match db.save(dir) {
                Ok(()) => Response::Ok(OkResponse::empty()),
                Err(e) => Response::error(e),
            },
            None => Response::error("no store directory configured; start `serve` with --path"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embsearch_core::{IndexKind, Metric, MockEmbedder};

    fn drive(requests: &[&str]) -> Vec<serde_json::Value> {
        drive_db(
            Database::new(MockEmbedder::new(32), Metric::Cosine),
            requests,
        )
    }

    /// Same, against a hybrid (vector + BM25) store.
    fn drive_hybrid(requests: &[&str]) -> Vec<serde_json::Value> {
        drive_db(
            Database::new_hybrid(MockEmbedder::new(32), Metric::Cosine, IndexKind::Flat),
            requests,
        )
    }

    fn drive_db(db: Database<MockEmbedder>, requests: &[&str]) -> Vec<serde_json::Value> {
        let input = requests.join("\n");
        let mut out: Vec<u8> = Vec::new();
        run(db, None, std::io::Cursor::new(input), &mut out).unwrap();
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn add_then_query_over_protocol() {
        let out = drive(&[
            r#"{"op":"add","id":"a","text":"quick brown fox"}"#,
            r#"{"op":"add","id":"b","text":"lazy dog"}"#,
            r#"{"op":"query","text":"quick fox","k":1}"#,
            r#"{"op":"count"}"#,
        ]);
        assert_eq!(out[0]["ok"], true);
        assert_eq!(out[2]["results"][0]["id"], "a");
        assert_eq!(out[3]["count"], 2);
    }

    #[test]
    fn bulk_upserts_and_reports_counts() {
        let out = drive(&[
            r#"{"op":"add","id":"a","text":"old text"}"#,
            r#"{"op":"bulk","items":[{"id":"a","text":"quick brown fox"},{"id":"b","text":"lazy dog"}]}"#,
            r#"{"op":"count"}"#,
            r#"{"op":"query","text":"quick fox","k":1}"#,
        ]);
        assert_eq!(out[1]["ok"], true);
        assert_eq!(out[1]["inserted_count"], 1); // b was new
        assert_eq!(out[1]["updated_count"], 1); // a was replaced
        assert_eq!(out[2]["count"], 2);
        assert_eq!(out[3]["results"][0]["id"], "a");
    }

    #[test]
    fn bulk_empty_items_is_ok() {
        let out = drive(&[r#"{"op":"bulk","items":[]}"#]);
        assert_eq!(out[0]["ok"], true);
        assert_eq!(out[0]["inserted_count"], 0);
        assert_eq!(out[0]["updated_count"], 0);
    }

    #[test]
    fn info_reports_model_and_dim() {
        let out = drive(&[
            r#"{"op":"add","id":"a","text":"hello"}"#,
            r#"{"op":"info"}"#,
        ]);
        assert_eq!(out[1]["ok"], true);
        assert_eq!(out[1]["model_id"], "mock-hash-v1");
        assert_eq!(out[1]["dim"], 32);
        assert_eq!(out[1]["count"], 1);
    }

    #[test]
    fn info_reports_whether_rerank_is_actually_available() {
        // A mock build has no cross-encoder, so `info` must say so. Callers
        // gate on this rather than on the daemon version: released onnx
        // binaries carry the `rerank` op but not the weights, and a caller
        // that assumed version implied capability would offer a reranker that
        // errors on first use.
        let out = drive(&[r#"{"op":"info"}"#]);
        assert_eq!(out[0]["ok"], true);
        assert_eq!(out[0]["rerank"], false);
    }

    #[test]
    fn compact_after_remove_keeps_results() {
        let out = drive(&[
            r#"{"op":"add","id":"a","text":"quick brown fox"}"#,
            r#"{"op":"add","id":"b","text":"lazy dog"}"#,
            r#"{"op":"remove","id":"b"}"#,
            r#"{"op":"compact"}"#,
            r#"{"op":"query","text":"quick fox","k":2}"#,
            r#"{"op":"count"}"#,
        ]);
        assert_eq!(out[3]["ok"], true);
        assert_eq!(out[4]["results"][0]["id"], "a");
        assert_eq!(out[4]["results"].as_array().unwrap().len(), 1);
        assert_eq!(out[5]["count"], 1);
    }

    #[test]
    fn errors_are_reported_not_fatal() {
        let out = drive(&[
            r#"{"op":"query"}"#,              // missing text/vector
            r#"{"op":"remove","id":"nope"}"#, // absent id -> removed:false
            r#"not json at all"#,             // parse error
            r#"{"op":"ping"}"#,               // loop still alive
        ]);
        assert_eq!(out[0]["ok"], false);
        assert_eq!(out[1]["removed"], false);
        assert_eq!(out[2]["ok"], false);
        assert_eq!(out[3]["ok"], true);
    }

    #[test]
    fn query_with_both_text_and_vector_is_an_error() {
        let out = drive(&[
            r#"{"op":"add","id":"a","text":"quick brown fox"}"#,
            r#"{"op":"query","text":"quick fox","vector":[0.1],"k":1}"#,
            r#"{"op":"ping"}"#, // loop still alive
        ]);
        assert_eq!(out[1]["ok"], false);
        assert_eq!(out[1]["error"], "query accepts text or vector, not both");
        assert_eq!(out[2]["ok"], true);
    }

    #[test]
    fn empty_id_and_empty_text_are_rejected() {
        let out = drive(&[
            r#"{"op":"add","id":"","text":"some text"}"#,
            r#"{"op":"add","id":"a","text":""}"#,
            r#"{"op":"upsert","id":"","text":"some text"}"#,
            r#"{"op":"bulk","items":[{"id":"ok","text":"fine"},{"id":"","text":"bad"}]}"#,
            r#"{"op":"count"}"#,
        ]);
        assert_eq!(out[0]["ok"], false);
        assert!(out[0]["error"]
            .as_str()
            .unwrap()
            .contains("id must not be empty"));
        assert_eq!(out[1]["ok"], false);
        assert!(out[1]["error"]
            .as_str()
            .unwrap()
            .contains("text must not be empty"));
        assert_eq!(out[2]["ok"], false);
        assert_eq!(out[3]["ok"], false);
        // Validation failed the whole batch before anything was applied.
        assert_eq!(out[4]["count"], 0);
    }

    #[test]
    fn lexical_retriever_returns_bm25_only() {
        let out = drive_hybrid(&[
            r#"{"op":"add","id":"a","text":"the quick brown fox"}"#,
            r#"{"op":"add","id":"b","text":"rust systems programming"}"#,
            r#"{"op":"add","id":"c","text":"postgres replication lag"}"#,
            r#"{"op":"query","text":"quick fox","k":5,"retriever":"lexical"}"#,
        ]);
        let results = out[3]["results"].as_array().unwrap();
        // Only the document sharing query terms comes back: no top-k padding
        // with unrelated documents, which is what makes this a usable leg.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["id"], "a");
        // Raw BM25, not the RRF score `hybrid` would return (~1/61).
        assert!(results[0]["score"].as_f64().unwrap() > 1.0);
    }

    #[test]
    fn lexical_and_dense_legs_are_separately_retrievable() {
        // The reason the op exists: a caller fusing the retrievers itself needs
        // each list on its own, which `hybrid` cannot give it.
        let out = drive_hybrid(&[
            r#"{"op":"add","id":"a","text":"the quick brown fox"}"#,
            r#"{"op":"add","id":"b","text":"rust systems programming"}"#,
            r#"{"op":"query","text":"quick fox","k":5,"retriever":"lexical"}"#,
            r#"{"op":"query","text":"quick fox","k":5,"retriever":"dense"}"#,
            r#"{"op":"query","text":"quick fox","k":5,"retriever":"hybrid"}"#,
        ]);
        for (i, response) in out.iter().enumerate().skip(2) {
            assert_eq!(response["ok"], true, "request {i} failed: {response:?}");
        }
        let lexical_top = out[2]["results"][0]["score"].as_f64().unwrap();
        let fused_top = out[4]["results"][0]["score"].as_f64().unwrap();
        assert!(lexical_top > fused_top, "BM25 sums dwarf RRF scores");
    }

    #[test]
    fn retriever_defaults_to_dense_and_hybrid_flag_still_works() {
        let out = drive_hybrid(&[
            r#"{"op":"add","id":"a","text":"the quick brown fox"}"#,
            r#"{"op":"query","text":"quick fox","k":1}"#,
            r#"{"op":"query","text":"quick fox","k":1,"hybrid":true}"#,
        ]);
        assert_eq!(out[1]["ok"], true);
        assert_eq!(out[2]["ok"], true);
        // The deprecated flag must still select fusion, i.e. an RRF score.
        assert!(out[2]["results"][0]["score"].as_f64().unwrap() < 0.1);
    }

    #[test]
    fn contradicting_hybrid_flag_and_retriever_is_rejected() {
        let out = drive_hybrid(&[
            r#"{"op":"add","id":"a","text":"quick fox"}"#,
            r#"{"op":"query","text":"quick fox","k":1,"hybrid":true,"retriever":"lexical"}"#,
        ]);
        assert_eq!(out[1]["ok"], false);
        assert!(out[1]["error"].as_str().unwrap().contains("contradicts"));
    }

    #[test]
    fn lexical_retriever_needs_a_hybrid_store() {
        let out = drive(&[
            r#"{"op":"add","id":"a","text":"quick fox"}"#,
            r#"{"op":"query","text":"quick fox","k":1,"retriever":"lexical"}"#,
        ]);
        assert_eq!(out[1]["ok"], false);
        assert!(out[1]["error"].as_str().unwrap().contains("hybrid store"));
    }

    #[test]
    fn lexical_retriever_rejects_a_vector_query() {
        let out =
            drive_hybrid(&[r#"{"op":"query","vector":[0.1,0.2],"k":1,"retriever":"lexical"}"#]);
        assert_eq!(out[0]["ok"], false);
        assert!(out[0]["error"].as_str().unwrap().contains("`text`"));
    }

    #[test]
    fn unknown_retriever_is_rejected() {
        let out = drive_hybrid(&[r#"{"op":"query","text":"x","k":1,"retriever":"bogus"}"#]);
        assert_eq!(out[0]["ok"], false);
    }

    #[test]
    fn rerank_on_a_mock_build_refuses_rather_than_scoring() {
        // The default build has no cross-encoder. Answering anyway — with
        // cosine, or with the input order — would look like a working reranker
        // while ranking on something else entirely.
        let out = drive(&[
            r#"{"op":"rerank","query":"quick fox","passages":[{"id":"a","text":"quick brown fox"}],"k":5}"#,
        ]);
        assert_eq!(out[0]["ok"], false);
        let err = out[0]["error"].as_str().unwrap();
        // Two builds refuse for two reasons — no onnx feature at all, or the
        // feature without bundled weights — and each names its own remedy.
        assert!(
            err.contains("onnx") || err.contains("--reranker-model"),
            "error should say how to get a reranker: {err}"
        );
    }

    #[test]
    fn rerank_validates_its_request_shape() {
        // Missing `query` is a malformed request, not an empty one.
        let out = drive(&[r#"{"op":"rerank","passages":[],"k":5}"#]);
        assert_eq!(out[0]["ok"], false);
        assert!(out[0]["error"]
            .as_str()
            .unwrap()
            .contains("invalid request"));
    }

    #[test]
    fn rerank_accepts_an_empty_passage_list() {
        // Nothing to score is a valid question with an empty answer, and must
        // not depend on whether a reranker exists — so this passes on a mock
        // build, an onnx build with weights, and an onnx build without.
        let out = drive(&[r#"{"op":"rerank","query":"q","passages":[],"k":5}"#]);
        assert_eq!(out[0]["ok"], true);
        assert_eq!(out[0]["reranked"], serde_json::json!([]));
    }
}
