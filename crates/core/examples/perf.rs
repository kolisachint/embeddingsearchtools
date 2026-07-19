//! Performance evaluation harness — no external dependencies.
//!
//! Run with an optimized build (numbers from a debug build are meaningless):
//!
//! ```bash
//! cargo run --release -p embsearch-core --example perf
//! # optional: cargo run --release -p embsearch-core --example perf -- 20000
//! ```
//!
//! It reports three things:
//! 1. **Scoring kernels** — vectorized vs scalar dot / squared-euclidean
//!    (throughput in GFLOP/s and the speedup).
//! 2. **Index build** — wall-clock to index N vectors, flat vs HNSW.
//! 3. **Query** — latency percentiles and throughput, flat vs HNSW, plus HNSW
//!    recall@k measured against exact (flat) search as ground truth.

use embsearch_core::internals::{dot, dot_scalar, sq_euclidean, sq_euclidean_scalar};
use embsearch_core::{FlatIndex, HnswIndex, Index, Metric};
use std::collections::HashSet;
use std::time::Instant;

const DIM: usize = 384; // matches all-MiniLM-L6-v2 / the default mock embedder

/// splitmix64 — deterministic, dependency-free.
struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

/// Generate `n` vectors drawn from `clusters` random centers with additive
/// noise, then L2-normalized. Real sentence embeddings are strongly clustered by
/// topic; uniform-random vectors are the curse-of-dimensionality worst case for
/// any ANN index and are not representative, so the index benchmark uses this.
fn clustered(rng: &mut Rng, n: usize, dim: usize, clusters: usize) -> Vec<Vec<f32>> {
    let centers: Vec<Vec<f32>> = (0..clusters).map(|_| rng.vec(dim)).collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % clusters];
            let mut v: Vec<f32> = c.iter().map(|&x| x + 0.35 * rng.f32()).collect();
            embsearch_core::l2_normalize(&mut v);
            v
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    println!("embeddingsearchtools — performance evaluation");
    println!("dim={DIM}, dataset={n} vectors, metric=cosine\n");

    bench_kernels();
    bench_index(n);
}

// --- 1. scoring kernels ------------------------------------------------------

fn bench_kernels() {
    println!("## 1. Scoring kernels (dim={DIM})");
    let mut rng = Rng(1);
    // A block of vectors so the working set doesn't sit entirely in registers.
    let count = 4096;
    let data: Vec<Vec<f32>> = (0..count).map(|_| rng.vec(DIM)).collect();
    let iters = 20_000usize;

    // 2*DIM flops per dot (mul + add); same for squared-euclidean (sub, mul, add ~ 3).
    let measure = |flops_per: f64, f: &dyn Fn(&[f32], &[f32]) -> f32| {
        // Warm up + defeat dead-code elimination by accumulating the result.
        let mut acc = 0.0f32;
        let start = Instant::now();
        for i in 0..iters {
            let a = &data[i % count];
            let b = &data[(i * 7 + 1) % count];
            acc += f(a, b);
        }
        let secs = start.elapsed().as_secs_f64();
        let gflops = (iters as f64 * flops_per) / secs / 1e9;
        std::hint::black_box(acc);
        (gflops, secs)
    };

    let (dsc, _) = measure(2.0 * DIM as f64, &dot_scalar);
    let (dv, _) = measure(2.0 * DIM as f64, &dot);
    let (esc, _) = measure(3.0 * DIM as f64, &sq_euclidean_scalar);
    let (ev, _) = measure(3.0 * DIM as f64, &sq_euclidean);

    println!(
        "  {:<22} {:>10} {:>10} {:>9}",
        "kernel", "scalar", "vectorized", "speedup"
    );
    println!(
        "  {:<22} {:>8.1}G {:>9.1}G {:>8.2}x",
        "dot",
        dsc,
        dv,
        dv / dsc
    );
    println!(
        "  {:<22} {:>8.1}G {:>9.1}G {:>8.2}x",
        "sq_euclidean",
        esc,
        ev,
        ev / esc
    );
    println!("  (GFLOP/s, higher is better)\n");
}

// --- 2 & 3. index build + query ---------------------------------------------

fn bench_index(n: usize) {
    let mut rng = Rng(42);
    let clusters = (n / 100).clamp(16, 512);
    let vectors = clustered(&mut rng, n, DIM, clusters);
    // Queries share the cluster structure (a query lands near some topic).
    let queries = clustered(&mut rng, 200, DIM, clusters);
    let k = 10;
    println!("(data: {clusters} topic clusters + noise, L2-normalized)\n");

    // --- build ---
    println!("## 2. Index build ({n} vectors)");
    let t = Instant::now();
    let mut flat = FlatIndex::new(DIM, Metric::Cosine);
    for (i, v) in vectors.iter().enumerate() {
        flat.add(&i.to_string(), v.clone()).unwrap();
    }
    let flat_build = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut hnsw = HnswIndex::new(DIM, Metric::Cosine);
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(&i.to_string(), v.clone()).unwrap();
    }
    let hnsw_build = t.elapsed().as_secs_f64();

    println!("  {:<8} {:>10} {:>14}", "backend", "build (s)", "vecs/s");
    println!(
        "  {:<8} {:>10.3} {:>14.0}",
        "flat",
        flat_build,
        n as f64 / flat_build
    );
    println!(
        "  {:<8} {:>10.3} {:>14.0}",
        "hnsw",
        hnsw_build,
        n as f64 / hnsw_build
    );
    println!();

    // Exact top-k for every query, as recall ground truth.
    let truth: Vec<HashSet<String>> = queries
        .iter()
        .map(|q| {
            flat.query(q, k)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();

    // --- query latency + recall ---
    println!("## 3. Query (k={k}, {} queries)", queries.len());
    let (flat_lat, _) = query_latencies(&flat, &queries, k);
    let flat_mean = mean(&flat_lat);
    println!(
        "  {:<16} p50={:>7.3}ms p95={:>7.3}ms mean={:>7.3}ms  {:>8.0} q/s",
        "flat (exact)",
        percentile(&sorted(&flat_lat), 50.0),
        percentile(&sorted(&flat_lat), 95.0),
        flat_mean,
        1000.0 / flat_mean,
    );

    // HNSW recall/latency tradeoff as the query-time ef sweeps.
    println!("\n  HNSW ef_search sweep (recall vs latency vs the exact scan):");
    println!(
        "  {:<10} {:>9} {:>9} {:>9} {:>10}",
        "ef_search", "recall", "p50(ms)", "p95(ms)", "speedup"
    );
    let default_ef = HnswIndex::new(DIM, Metric::Cosine).ef_search();
    for &ef in &[16usize, 32, 64, 128, 200, 256] {
        hnsw.set_ef_search(ef);
        let (lat, hits) = query_latencies(&hnsw, &queries, k);
        let recall = mean_recall(&hits, &truth);
        let s = sorted(&lat);
        let marker = if ef == default_ef { " (default)" } else { "" };
        println!(
            "  {:<10} {:>8.1}% {:>9.3} {:>9.3} {:>9.1}x{}",
            ef,
            recall * 100.0,
            percentile(&s, 50.0),
            percentile(&s, 95.0),
            flat_mean / mean(&lat),
            marker,
        );
    }
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn mean_recall(approx: &[Vec<String>], truth: &[HashSet<String>]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0;
    for (a, t) in approx.iter().zip(truth) {
        if t.is_empty() {
            continue;
        }
        let hit = a.iter().filter(|id| t.contains(*id)).count();
        sum += hit as f64 / t.len() as f64;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Time each query (milliseconds) and capture the returned ids.
fn query_latencies<I: Index>(
    idx: &I,
    queries: &[Vec<f32>],
    k: usize,
) -> (Vec<f64>, Vec<Vec<String>>) {
    // Warm up.
    for q in queries.iter().take(20) {
        std::hint::black_box(idx.query(q, k).unwrap());
    }
    let mut lat = Vec::with_capacity(queries.len());
    let mut hits = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        let res = idx.query(q, k).unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
        hits.push(res.into_iter().map(|h| h.id).collect());
    }
    (lat, hits)
}
