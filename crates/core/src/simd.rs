//! Vector kernels for the scoring hot path.
//!
//! Every query — flat or HNSW — bottoms out in a dot product (cosine/dot) or a
//! squared-distance (euclidean) over `dim`-length vectors, so these two loops
//! dominate query cost. The implementations here are written to *auto-vectorize*
//! rather than to use `std::simd` (nightly-only) or an external SIMD crate: the
//! project keeps a tiny, stable-toolchain dependency tree, so we lean on LLVM.
//!
//! The trick is **multiple independent accumulators**. A naive
//! `iter().zip().map().sum()` folds into a single accumulator, so every multiply
//! waits on the previous add (a serial dependency chain that pins throughput to
//! floating-point add latency). Splitting the reduction across `LANES` lanes
//! gives the compiler independent chains it can both vectorize and pipeline,
//! then we fold the lanes once at the end. See [`crate::internals`] for the
//! scalar references these are benchmarked against.

/// Number of independent accumulator lanes. Eight covers a 256-bit vector of
/// `f32`, and degrades gracefully to narrower SIMD or plain ILP.
const LANES: usize = 8;

/// Dot product `Σ aᵢ·bᵢ`. Panics (in debug) if the slices differ in length.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        // `x` and `y` have length exactly LANES, so the compiler drops the
        // bounds checks and vectorizes this fixed-width loop.
        for l in 0..LANES {
            acc[l] += x[l] * y[l];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// Squared Euclidean distance `Σ (aᵢ-bᵢ)²`. The caller takes the square root
/// once, outside the hot loop, if it needs the true distance.
#[inline]
pub fn sq_euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for l in 0..LANES {
            let d = x[l] - y[l];
            acc[l] += d * d;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Straightforward single-accumulator dot product. Kept as the correctness and
/// performance baseline for [`dot`]; see [`crate::internals`].
#[inline]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Single-accumulator squared Euclidean distance. Baseline for [`sq_euclidean`].
#[inline]
pub fn sq_euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(seed: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                ((seed.wrapping_mul(2654435761).wrapping_add(i as u32) % 1000) as f32) / 500.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn matches_scalar_reference() {
        // Across lengths that exercise the tail path (not a multiple of LANES).
        for n in [0usize, 1, 7, 8, 9, 16, 31, 64, 127, 384, 385] {
            let a = v(1, n);
            let b = v(2, n);
            let (d, ds) = (dot(&a, &b), dot_scalar(&a, &b));
            assert!(
                (d - ds).abs() <= 1e-3 * (1.0 + ds.abs()),
                "dot n={n}: {d} vs {ds}"
            );
            let (e, es) = (sq_euclidean(&a, &b), sq_euclidean_scalar(&a, &b));
            assert!(
                (e - es).abs() <= 1e-3 * (1.0 + es.abs()),
                "sq_euclidean n={n}: {e} vs {es}"
            );
        }
    }

    #[test]
    fn dot_of_basis_vectors() {
        let mut a = vec![0.0f32; 384];
        let mut b = vec![0.0f32; 384];
        a[3] = 1.0;
        b[3] = 2.0;
        assert_eq!(dot(&a, &b), 2.0);
        b[3] = 0.0;
        b[7] = 5.0;
        assert_eq!(dot(&a, &b), 0.0);
    }
}
