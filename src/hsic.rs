//! Port of `lingam/hsic.py` restricted to `bw_method="mdbs"` (median-distance
//! bandwidth), used by `get_error_independence_p_values`.
//!
//! The gamma-approximation test is inherently O(n^2); this implementation keeps
//! that complexity but strips the Python version's intermediate work:
//!
//! * the doubly-centered gram matrices `Kc`, `Lc` are never materialised — every
//!   reduction is rewritten in terms of the raw gram, its row sums and a handful
//!   of scalars. `test_stat` needs only one side centered
//!   (`<HKH, HLH>_F = <K, HLH>_F`), and the RBF diagonal is exactly 1 so the
//!   mean under H0 is closed-form;
//! * only the strict upper triangle of the symmetric grams is built and scanned;
//! * for large `n` the build and the two reduction passes run on rayon (routed
//!   through the user thread pool, see [`crate::pool`]).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::pool;
use crate::util::{gamma_sf, median};

/// At or above this sample count the O(n^2) passes are parallelised. Below it the
/// serial path (already a few times faster than the Python version) wins once
/// rayon's fork/join overhead is accounted for.
const PAR_THRESHOLD: usize = 1024;

/// Minimum rows per rayon job, so task count stays bounded (~n/256) instead of
/// scaling with the pool width — the per-row work here is tiny.
const MIN_ROWS_PER_JOB: usize = 256;

/// `get_kernel_width` for a single-column input: sqrt(0.5 * median of the
/// positive squared pairwise distances over the first <=100 points).
fn kernel_width(x: &[f64]) -> f64 {
    let n = x.len().min(100);
    let xm = &x[..n];
    let mut dists = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (xm[i] - xm[j]).powi(2);
            if d > 0.0 {
                dists.push(d);
            }
        }
    }
    (0.5 * median(&dists)).sqrt()
}

/// Start offset of each row inside a packed strict-upper-triangle buffer; row `i`
/// stores columns `i+1..n`, i.e. `n - 1 - i` entries.
fn row_offsets(n: usize) -> Vec<usize> {
    let mut offs = vec![0usize; n];
    for i in 1..n {
        offs[i] = offs[i - 1] + (n - i);
    }
    offs
}

/// Split a packed strict-upper-triangle buffer into one mutable slice per row.
fn row_segments_mut(buf: &mut [f64], n: usize) -> Vec<&mut [f64]> {
    let mut segs = Vec::with_capacity(n.saturating_sub(1));
    let mut rest = buf;
    for i in 0..n.saturating_sub(1) {
        let (head, tail) = rest.split_at_mut(n - 1 - i);
        segs.push(head);
        rest = tail;
    }
    segs
}

/// Fill `out` with the strict upper triangle (row-major, `i < j`) of the RBF gram
/// `exp(-(x_i - x_j)^2 / (2 w^2))`.
fn build_upper(x: &[f64], width: f64, out: &mut [f64], parallel: bool) {
    let n = x.len();
    let inv = -1.0 / (2.0 * width * width);
    if parallel {
        row_segments_mut(out, n)
            .into_par_iter()
            .with_min_len(MIN_ROWS_PER_JOB)
            .enumerate()
            .for_each(|(i, seg)| {
                let xi = x[i];
                for (jj, slot) in seg.iter_mut().enumerate() {
                    let d = xi - x[i + 1 + jj];
                    *slot = (d * d * inv).exp();
                }
            });
    } else {
        let mut pos = 0;
        for i in 0..n.saturating_sub(1) {
            let xi = x[i];
            for &xj in &x[i + 1..] {
                let d = xi - xj;
                out[pos] = (d * d * inv).exp();
                pos += 1;
            }
        }
    }
}

/// `hsic_test_gamma(X, Y, bw_method="mdbs")` → `(test_stat, p_value)`.
pub fn hsic_test_gamma(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    let nf = n as f64;
    if n >= PAR_THRESHOLD {
        pool::install(|| hsic_inner(x, y, n, nf, true))
    } else {
        hsic_inner(x, y, n, nf, false)
    }
}

fn hsic_inner(x: &[f64], y: &[f64], n: usize, nf: f64, parallel: bool) -> (f64, f64) {
    let wx = kernel_width(x);
    let wy = kernel_width(y);

    // Packed strict upper triangles of the (symmetric, unit-diagonal) RBF grams.
    let m = n * n.saturating_sub(1) / 2;
    let mut ku = vec![0.0f64; m];
    let mut lu = vec![0.0f64; m];
    build_upper(x, wx, &mut ku, parallel);
    build_upper(y, wy, &mut lu, parallel);

    let offs = if parallel { row_offsets(n) } else { Vec::new() };

    // Pass A: off-diagonal row sums of K and L, plus S = sum_{i<j} K_ij L_ij.
    let (mut rk, mut rl, skl_off) = if parallel {
        (0..n.saturating_sub(1))
            .into_par_iter()
            .with_min_len(MIN_ROWS_PER_JOB)
            .fold(
                || (vec![0.0f64; n], vec![0.0f64; n], 0.0f64),
                |(mut rk, mut rl, mut skl), i| {
                    let s = offs[i];
                    let len = n - 1 - i;
                    let krow = &ku[s..s + len];
                    let lrow = &lu[s..s + len];
                    let (mut ri_k, mut ri_l) = (0.0, 0.0);
                    for jj in 0..len {
                        let k = krow[jj];
                        let l = lrow[jj];
                        rk[i + 1 + jj] += k;
                        rl[i + 1 + jj] += l;
                        ri_k += k;
                        ri_l += l;
                        skl += k * l;
                    }
                    rk[i] += ri_k;
                    rl[i] += ri_l;
                    (rk, rl, skl)
                },
            )
            .reduce(
                || (vec![0.0f64; n], vec![0.0f64; n], 0.0f64),
                |(mut ak, mut al, sa), (bk, bl, sb)| {
                    for t in 0..n {
                        ak[t] += bk[t];
                        al[t] += bl[t];
                    }
                    (ak, al, sa + sb)
                },
            )
    } else {
        let mut rk = vec![0.0f64; n];
        let mut rl = vec![0.0f64; n];
        let mut skl = 0.0;
        let mut pos = 0;
        for i in 0..n.saturating_sub(1) {
            for j in (i + 1)..n {
                let k = ku[pos];
                let l = lu[pos];
                pos += 1;
                rk[i] += k;
                rk[j] += k;
                rl[i] += l;
                rl[j] += l;
                skl += k * l;
            }
        }
        (rk, rl, skl)
    };

    // Promote to full row sums by adding the unit diagonal (K_ii = L_ii = 1).
    for t in 0..n {
        rk[t] += 1.0;
        rl[t] += 1.0;
    }
    let ak: f64 = rk.iter().sum();
    let al: f64 = rl.iter().sum();
    let mk = ak / (nf * nf);
    let ml = al / (nf * nf);

    // test_stat = 1/n * ( sum_ij K_ij L_ij  -  2 sum_i rk_i (rl_i / n)  +  mL aK ),
    // using <HKH, HLH>_F = <K, HLH>_F so only L needs centering.
    let s_kl = nf + 2.0 * skl_off; // + unit diagonal (1*1) per row
    let cross: f64 = rk.iter().zip(&rl).map(|(&a, &b)| a * (b / nf)).sum();
    let stat = (s_kl - 2.0 * cross + ml * ak) / nf;

    // var = 2/36 * sum_{i<j} (Kc_ij Lc_ij)^2 / (n (n-1)), then the H0 scaling.
    // Kc_ij = K_ij - rk_i/n - rk_j/n + mK  (reconstructed on the fly).
    let var_off = if parallel {
        (0..n.saturating_sub(1))
            .into_par_iter()
            .with_min_len(MIN_ROWS_PER_JOB)
            .map(|i| {
                let s = offs[i];
                let len = n - 1 - i;
                let krow = &ku[s..s + len];
                let lrow = &lu[s..s + len];
                let ui = rk[i] / nf;
                let vi = rl[i] / nf;
                let mut acc = 0.0;
                for jj in 0..len {
                    let kc = krow[jj] - ui - rk[i + 1 + jj] / nf + mk;
                    let lc = lrow[jj] - vi - rl[i + 1 + jj] / nf + ml;
                    let p = kc * lc;
                    acc += p * p;
                }
                acc
            })
            .sum::<f64>()
    } else {
        let mut acc = 0.0;
        let mut pos = 0;
        for i in 0..n.saturating_sub(1) {
            let ui = rk[i] / nf;
            let vi = rl[i] / nf;
            for j in (i + 1)..n {
                let kc = ku[pos] - ui - rk[j] / nf + mk;
                let lc = lu[pos] - vi - rl[j] / nf + ml;
                pos += 1;
                let p = kc * lc;
                acc += p * p;
            }
        }
        acc
    };

    let mut var = 2.0 / 36.0 * var_off / (nf * (nf - 1.0));
    var *= 72.0 * (nf - 4.0) * (nf - 5.0) / (nf * (nf - 1.0) * (nf - 2.0) * (nf - 3.0));

    // mean under H0: sum_{i!=j} K = aK - n thanks to the unit diagonal.
    let mu_x = (ak - nf) / (nf * (nf - 1.0));
    let mu_y = (al - nf) / (nf * (nf - 1.0));
    let mean = (1.0 + mu_x * mu_y - mu_x - mu_y) / nf;

    let alpha = mean * mean / var;
    let beta = var * nf / mean;
    let p = gamma_sf(stat, alpha, beta);

    (stat, p)
}

/// `ruslingam.hsic_test_gamma(X, Y)` — gamma-approximation HSIC independence test,
/// matching `lingam.hsic.hsic_test_gamma(X, Y, bw_method="mdbs")`.
///
/// `X` and `Y` are 1-D arrays of equal length; returns `(test_stat, p_value)`.
#[pyfunction]
#[pyo3(name = "hsic_test_gamma")]
pub fn hsic_test_gamma_py(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    let xv: Vec<f64> = x.as_array().iter().copied().collect();
    let yv: Vec<f64> = y.as_array().iter().copied().collect();
    if xv.len() != yv.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "X and Y must have the same length.",
        ));
    }
    Ok(hsic_test_gamma(&xv, &yv))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rtol: f64) {
        assert!(
            (a - b).abs() <= rtol * (1.0 + b.abs()),
            "{a} vs {b} (rtol {rtol})"
        );
    }

    #[test]
    fn serial_and_parallel_paths_agree() {
        let n = 700usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).cos() + 0.2 * x[i]).collect();
        let (s_stat, s_p) = hsic_inner(&x, &y, n, n as f64, false);
        let (p_stat, p_p) = hsic_inner(&x, &y, n, n as f64, true);
        close(s_stat, p_stat, 1e-9);
        close(s_p, p_p, 1e-9);
    }

    #[test]
    fn separates_dependent_from_independent() {
        let n = 400usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i.wrapping_mul(2_654_435_761) % 1000) as f64) / 500.0 - 1.0)
            .collect();
        let indep: Vec<f64> = (0..n)
            .map(|i| ((i.wrapping_mul(40_503) % 997) as f64) / 498.0 - 1.0)
            .collect();
        let dep: Vec<f64> = x.iter().map(|v| v * v).collect();

        let (_, p_indep) = hsic_test_gamma(&x, &indep);
        let (_, p_dep) = hsic_test_gamma(&x, &dep);
        assert!(p_dep < p_indep, "p_dep {p_dep} !< p_indep {p_indep}");
        assert!(p_dep < 0.05, "p_dep {p_dep} not significant");
    }
}
