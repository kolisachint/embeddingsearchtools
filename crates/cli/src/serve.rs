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
//! - `{"op":"query","text":"...","k":5}`        → search
//! - `{"op":"query","vector":[...],"k":5}`       → search a precomputed vector
//! - `{"op":"add","id":"x","text":"..."}`        → insert
//! - `{"op":"update","id":"x","text":"..."}`     → replace
//! - `{"op":"upsert","id":"x","text":"..."}`     → insert-or-replace
//! - `{"op":"remove","id":"x"}`                  → delete
//! - `{"op":"save"}`                             → persist to the store dir
//! - `{"op":"count"}`                            → live vector count
//! - `{"op":"ping"}`                             → readiness probe
//!
//! Responses always carry `ok`:
//! - `{"ok":true, ...}` with op-specific fields (`results`, `inserted`, ...)
//! - `{"ok":false,"error":"message"}`

use embsearch_core::{Database, Embedder, SearchResult};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::PathBuf;

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
    Save,
    Count,
    Ping,
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
}

impl OkResponse {
    fn empty() -> Self {
        Self {
            ok: true,
            results: None,
            inserted: None,
            removed: None,
            count: None,
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
        Request::Count => {
            let mut r = OkResponse::empty();
            r.count = Some(db.len());
            Response::Ok(r)
        }
        Request::Query { text, vector, k } => {
            let result = match (text, vector) {
                (Some(t), _) => db.query(&t, k),
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
    use embsearch_core::{Metric, MockEmbedder};

    fn drive(requests: &[&str]) -> Vec<serde_json::Value> {
        let db = Database::new(MockEmbedder::new(32), Metric::Cosine);
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
}
