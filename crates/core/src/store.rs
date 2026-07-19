//! On-disk persistence for an [`AnyIndex`].
//!
//! Layout (a directory):
//! - `manifest.json` — dim, metric, index kind, model id, row/live counts,
//!   format version.
//! - `ids.json`      — array aligned to matrix rows; `null` marks a tombstone.
//! - `vectors.bin`   — raw little-endian `f32`, row-major, `rows * dim` values.
//!
//! Only the vectors are persisted — both backends share this layout. The exact
//! [`FlatIndex`](crate::FlatIndex) needs nothing more; the approximate
//! [`HnswIndex`](crate::HnswIndex) rebuilds its navigation graph from these
//! vectors on load, so the on-disk format is identical regardless of backend.
//!
//! `vectors.bin` is a flat `f32` buffer specifically so it can be memory-mapped
//! by future backends; the current loader reads it with buffered IO. Writes go
//! to temp files and are atomically renamed so a crash mid-save can't corrupt an
//! existing store.

use crate::error::{Error, Result};
use crate::index::{AnyIndex, Index, IndexKind, Metric};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const FORMAT_VERSION: u32 = 1;
const MANIFEST: &str = "manifest.json";
const IDS: &str = "ids.json";
const VECTORS: &str = "vectors.bin";
const TEXTS: &str = "texts.json";

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    format_version: u32,
    dim: usize,
    metric: Metric,
    /// Index backend. Defaults to `flat` so stores written before this field
    /// existed still load as exact indexes.
    #[serde(default)]
    index: IndexKind,
    /// Whether a BM25 lexical index (and `texts.json`) accompanies the vectors.
    /// Defaults to `false` for backward compatibility.
    #[serde(default)]
    hybrid: bool,
    model_id: String,
    /// Physical rows in `vectors.bin` (live + tombstoned).
    rows: usize,
    /// Live (non-tombstoned) rows.
    live: usize,
}

/// Write `index` to `dir`, creating it if needed. Atomic per-file. `hybrid`
/// records whether a companion `texts.json` (written separately via
/// [`save_texts`]) is expected on load.
pub fn save(dir: impl AsRef<Path>, index: &AnyIndex, model_id: &str, hybrid: bool) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let (data, ids) = index.parts();

    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        dim: index.dim(),
        metric: index.metric(),
        index: index.kind(),
        hybrid,
        model_id: model_id.to_string(),
        rows: ids.len(),
        live: index.len(),
    };

    write_atomic(&dir.join(MANIFEST), |w| {
        serde_json::to_writer_pretty(w, &manifest).map_err(Error::from)
    })?;
    write_atomic(&dir.join(IDS), |w| {
        serde_json::to_writer(w, &ids).map_err(Error::from)
    })?;
    write_atomic(&dir.join(VECTORS), |w| {
        let mut bw = BufWriter::new(w);
        for value in data {
            bw.write_all(&value.to_le_bytes())?;
        }
        bw.flush()?;
        Ok(())
    })?;
    Ok(())
}

/// Load an index from `dir`. Returns the index and the model id recorded when it
/// was saved. [`crate::Database::open`] verifies that id against its embedder;
/// callers loading a store directly (e.g. tooling without an embedder) should do
/// the same check themselves before mixing vectors from different models.
///
/// An HNSW-backed store rebuilds its navigation graph from the persisted vectors
/// here, so loading is `O(n log n)` for that backend versus `O(n)` for flat.
///
/// The third tuple element is the `hybrid` flag: when true a companion
/// `texts.json` exists and should be read with [`load_texts`] to rebuild the
/// lexical index.
pub fn load(dir: impl AsRef<Path>) -> Result<(AnyIndex, String, bool)> {
    let dir = dir.as_ref();

    let manifest: Manifest = {
        let f = std::fs::File::open(dir.join(MANIFEST))?;
        serde_json::from_reader(BufReader::new(f))
            .map_err(|e| Error::corrupt(format!("invalid {MANIFEST}: {e}")))?
    };
    if manifest.format_version != FORMAT_VERSION {
        return Err(Error::corrupt(format!(
            "unsupported store format version {} (expected {FORMAT_VERSION})",
            manifest.format_version
        )));
    }

    let ids: Vec<Option<String>> = {
        let f = std::fs::File::open(dir.join(IDS))?;
        serde_json::from_reader(BufReader::new(f))
            .map_err(|e| Error::corrupt(format!("invalid {IDS}: {e}")))?
    };
    if ids.len() != manifest.rows {
        return Err(Error::corrupt(format!(
            "manifest rows {} != ids length {}",
            manifest.rows,
            ids.len()
        )));
    }

    let data = read_f32_le(&dir.join(VECTORS))?;
    let expected = manifest.rows * manifest.dim;
    if data.len() != expected {
        return Err(Error::corrupt(format!(
            "vectors.bin holds {} floats, expected {} ({} rows x {} dim)",
            data.len(),
            expected,
            manifest.rows,
            manifest.dim
        )));
    }

    let index = AnyIndex::from_parts(manifest.index, manifest.dim, manifest.metric, data, ids)?;
    Ok((index, manifest.model_id, manifest.hybrid))
}

/// Persist the id → text map for a hybrid store's lexical index. Atomic.
pub fn save_texts(dir: impl AsRef<Path>, texts: &HashMap<String, String>) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    write_atomic(&dir.join(TEXTS), |w| {
        serde_json::to_writer(BufWriter::new(w), texts).map_err(Error::from)
    })
}

/// Load the id → text map written by [`save_texts`].
pub fn load_texts(dir: impl AsRef<Path>) -> Result<HashMap<String, String>> {
    let path = dir.as_ref().join(TEXTS);
    let f = std::fs::File::open(&path)?;
    serde_json::from_reader(BufReader::new(f))
        .map_err(|e| Error::corrupt(format!("invalid {TEXTS}: {e}")))
}

/// True if `dir` looks like an existing store (has a manifest).
pub fn exists(dir: impl AsRef<Path>) -> bool {
    dir.as_ref().join(MANIFEST).is_file()
}

fn read_f32_le(path: &Path) -> Result<Vec<f32>> {
    let f = std::fs::File::open(path)?;
    let mut reader = BufReader::new(f);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    if bytes.len() % 4 != 0 {
        return Err(Error::corrupt(format!(
            "{} length {} is not a multiple of 4",
            path.display(),
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Write via a sibling temp file, then rename into place.
fn write_atomic<F>(path: &Path, write: F) -> Result<()>
where
    F: FnOnce(&mut std::fs::File) -> Result<()>,
{
    let tmp: PathBuf = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        write(&mut f)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}
