//! `embsearch` — CLI + stdio daemon for the embedding search engine.
//!
//! Subcommands:
//! - `index`  — bulk-build a store from a JSONL file of `{"id","text"}`.
//! - `add` / `update` / `remove` — single-record mutations.
//! - `query`  — search a store and print hits.
//! - `serve`  — run the long-lived NDJSON daemon (see `serve.rs`).
//!
//! The embedding backend is chosen at build time: the default build uses the
//! deterministic `MockEmbedder`; building with `--features onnx` uses real
//! `all-MiniLM-L6-v2` inference.

mod serve;

use clap::{Parser, Subcommand};
use embsearch_core::{Database, Embedder, Index, Metric};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "embsearch", version, about = "Minimal embedding search engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Bulk-index a JSONL file of {"id","text"} records into a store.
    Index {
        #[command(flatten)]
        store: StoreArgs,
        /// Path to a JSONL file; each line is `{"id":"...","text":"..."}`.
        /// Use `-` to read from stdin.
        #[arg(short, long)]
        input: String,
    },
    /// Add one record.
    Add {
        #[command(flatten)]
        store: StoreArgs,
        #[arg(long)]
        id: String,
        #[arg(long)]
        text: String,
    },
    /// Update the text of an existing record.
    Update {
        #[command(flatten)]
        store: StoreArgs,
        #[arg(long)]
        id: String,
        #[arg(long)]
        text: String,
    },
    /// Remove a record by id.
    Remove {
        #[command(flatten)]
        store: StoreArgs,
        #[arg(long)]
        id: String,
    },
    /// Query a store and print the top matches.
    Query {
        #[command(flatten)]
        store: StoreArgs,
        /// Query text.
        text: String,
        /// Number of results.
        #[arg(short, long, default_value_t = 5)]
        k: usize,
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Run the long-lived NDJSON daemon on stdin/stdout.
    Serve {
        #[command(flatten)]
        store: StoreArgs,
    },
}

/// Shared store-location + config flags.
#[derive(clap::Args)]
struct StoreArgs {
    /// Store directory. Created on first write.
    #[arg(short, long, default_value = "./embsearch-store")]
    path: PathBuf,
    /// Similarity metric: cosine|dot|euclidean. Used when creating a new store
    /// (default: cosine); an existing store keeps its stored metric, and a
    /// conflicting flag only produces a warning.
    #[arg(short, long)]
    metric: Option<String>,
    /// Override the model directory (with `--features onnx`): a dir holding
    /// `model.onnx` + `tokenizer.json`. Ignored by the default mock build.
    #[arg(long)]
    model: Option<PathBuf>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> embsearch_core::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Index { store, input } => cmd_index(store, input),
        Command::Add { store, id, text } => {
            let mut db = open_db(&store)?;
            db.add(&id, &text)?;
            db.save(&store.path)?;
            println!("added '{id}' ({} vectors)", db.len());
            Ok(())
        }
        Command::Update { store, id, text } => {
            let mut db = open_db(&store)?;
            db.update(&id, &text)?;
            db.save(&store.path)?;
            println!("updated '{id}'");
            Ok(())
        }
        Command::Remove { store, id } => {
            let mut db = open_db(&store)?;
            let removed = db.remove(&id)?;
            db.compact();
            db.save(&store.path)?;
            if removed {
                println!("removed '{id}' ({} vectors)", db.len());
            } else {
                println!("'{id}' not found");
            }
            Ok(())
        }
        Command::Query {
            store,
            text,
            k,
            json,
        } => {
            let db = open_db(&store)?;
            let hits = db.query(&text, k)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else if hits.is_empty() {
                println!("(no results)");
            } else {
                for (rank, h) in hits.iter().enumerate() {
                    println!("{:>2}. {:<24} {:.4}", rank + 1, h.id, h.score);
                }
            }
            Ok(())
        }
        Command::Serve { store } => {
            let db = open_db(&store)?;
            eprintln!(
                "embsearch daemon ready: {} vectors, model '{}', dim {}",
                db.len(),
                db.embedder().model_id(),
                db.embedder().dim()
            );
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            serve::run(db, Some(store.path.clone()), stdin.lock(), stdout.lock())?;
            Ok(())
        }
    }
}

fn cmd_index(store: StoreArgs, input: String) -> embsearch_core::Result<()> {
    let mut db = open_db(&store)?;
    let reader: Box<dyn BufRead> = if input == "-" {
        Box::new(std::io::BufReader::new(std::io::stdin()))
    } else {
        Box::new(std::io::BufReader::new(std::fs::File::open(&input)?))
    };

    #[derive(serde::Deserialize)]
    struct Record {
        id: String,
        text: String,
    }

    let mut n = 0usize;
    let stderr = std::io::stderr();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: Record =
            serde_json::from_str(&line).map_err(|e| embsearch_core::Error::InvalidInput {
                line: lineno + 1,
                msg: e.to_string(),
            })?;
        // Upsert so re-indexing an existing store updates in place.
        db.upsert(&rec.id, &rec.text)?;
        n += 1;
        if n.is_multiple_of(1000) {
            let _ = writeln!(stderr.lock(), "  indexed {n}...");
        }
    }
    db.save(&store.path)?;
    println!(
        "indexed {n} records -> {} vectors at {}",
        db.len(),
        store.path.display()
    );
    Ok(())
}

/// Open (or create) the store, wiring in the compile-time-selected embedder.
///
/// `--metric` only takes effect when a new store is created. When it conflicts
/// with an existing store's manifest, the stored metric stays authoritative and
/// a warning is printed to stderr.
fn open_db(store: &StoreArgs) -> embsearch_core::Result<Database<Box<dyn Embedder>>> {
    let requested: Option<Metric> = store.metric.as_deref().map(str::parse).transpose()?;
    let embedder = build_embedder(store)?;
    let db = Database::open_or_create(embedder, &store.path, requested.unwrap_or(Metric::Cosine))?;
    if let Some(requested) = requested {
        let stored = db.index().metric();
        if stored != requested {
            eprintln!(
                "warning: --metric {requested} ignored: existing store at {} uses metric \
                 '{stored}' (the stored metric is authoritative; re-index to change it)",
                store.path.display()
            );
        }
    }
    Ok(db)
}

#[cfg(not(feature = "onnx"))]
fn build_embedder(_store: &StoreArgs) -> embsearch_core::Result<Box<dyn Embedder>> {
    // Default build: deterministic mock embedder, 384-d to mirror MiniLM.
    Ok(Box::new(embsearch_core::MockEmbedder::default()))
}

#[cfg(feature = "onnx")]
fn build_embedder(store: &StoreArgs) -> embsearch_core::Result<Box<dyn Embedder>> {
    use embsearch_core::MiniLmEmbedder;
    let embedder = match &store.model {
        Some(dir) => MiniLmEmbedder::from_dir(dir)?,
        None => MiniLmEmbedder::from_bundled()?,
    };
    Ok(Box::new(embedder))
}
