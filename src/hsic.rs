//! Port of `lingam/hsic.py` restricted to `bw_method="mdbs"` (median-distance
//! bandwidth), used by `get_error_independence_p_values`.

use ndarray::{Array2, Axis};

use crate::util::{gamma_sf, median};

/// `get_kernel_width` for a single-column input: sqrt(0.5 * median of the
/// positive squared pairwise distances over the first <=100 points).
fn kernel_width(x: &[f64]) -> f64 {
    let n = x.len().min(100);
    let xm = &x[..n];
    let mut dists = Vec::with_capacity(n * (n.saturating_sub(1)) / 2);
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

/// `get_gram_matrix`: returns `(K, Kc)` where `Kc` is the doubly-centered gram matrix.
fn gram(x: &[f64], width: f64) -> (Array2<f64>, Array2<f64>) {
    let n = x.len();
    let w2 = 2.0 * width * width;
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let h = (x[i] - x[j]).powi(2);
            k[[i, j]] = (-h / w2).exp();
        }
    }
    let colsums = k.sum_axis(Axis(0));
    let rowsums = k.sum_axis(Axis(1));
    let allsum = rowsums.sum();
    let nf = n as f64;
    let mut kc = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            kc[[i, j]] = k[[i, j]] - (colsums[j] + rowsums[i]) / nf + allsum / (nf * nf);
        }
    }
    (k, kc)
}

/// `hsic_test_gamma(X, Y, bw_method="mdbs")` → `(test_stat, p_value)`.
pub fn hsic_test_gamma(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    let nf = n as f64;

    let wx = kernel_width(x);
    let wy = kernel_width(y);
    let (k, kc) = gram(x, wx);
    let (l, lc) = gram(y, wy);

    // test_stat = 1/n * sum(Kc.T * Lc)
    let mut stat = 0.0;
    for i in 0..n {
        for j in 0..n {
            stat += kc[[j, i]] * lc[[i, j]];
        }
    }
    stat /= nf;

    // var = (1/6 Kc Lc)^2 ; var = 1/n/(n-1) (sum(var) - trace(var)) ; var = 72(n-4)(n-5)/... var
    let mut sum_var = 0.0;
    let mut trace_var = 0.0;
    for i in 0..n {
        for j in 0..n {
            let v = (kc[[i, j]] * lc[[i, j]] / 6.0).powi(2);
            sum_var += v;
            if i == j {
                trace_var += v;
            }
        }
    }
    let mut var = (sum_var - trace_var) / (nf * (nf - 1.0));
    var *= 72.0 * (nf - 4.0) * (nf - 5.0) / (nf * (nf - 1.0) * (nf - 2.0) * (nf - 3.0));

    // mean, using K and L with zeroed diagonal
    let mut ksum = 0.0;
    let mut lsum = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                ksum += k[[i, j]];
                lsum += l[[i, j]];
            }
        }
    }
    let mu_x = ksum / (nf * (nf - 1.0));
    let mu_y = lsum / (nf * (nf - 1.0));
    let mean = (1.0 + mu_x * mu_y - mu_x - mu_y) / nf;

    let alpha = mean * mean / var;
    let beta = var * nf / mean;
    let p = gamma_sf(stat, alpha, beta);

    (stat, p)
}
