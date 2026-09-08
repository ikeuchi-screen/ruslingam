//! Core DirectLiNGAM causal-ordering (`measure="pwling"`) — a direct port of
//! `lingam/direct_lingam.py` with `prior_knowledge=None`
//! (so `Uc = U` and `Vj = []` everywhere).

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rayon::prelude::*;

use crate::adjacency::estimate_adjacency_matrix;

fn mean(v: &ArrayView1<f64>) -> f64 {
    v.sum() / v.len() as f64
}

fn pop_std(v: &ArrayView1<f64>) -> f64 {
    let m = mean(v);
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
    var.sqrt()
}

/// `xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj`
pub fn residual(xi: &ArrayView1<f64>, xj: &ArrayView1<f64>) -> Array1<f64> {
    let n = xi.len() as f64;
    let mi = mean(xi);
    let mj = mean(xj);
    let cov = xi
        .iter()
        .zip(xj.iter())
        .map(|(a, b)| (a - mi) * (b - mj))
        .sum::<f64>()
        / n;
    let var_j = xj.iter().map(|b| (b - mj) * (b - mj)).sum::<f64>() / n;
    let k = cov / var_j;
    // element t: xi[t] - (k * xj[t]) — same arithmetic as `xi - xj.mapv(|b| k * b)`
    // but without the two intermediate allocations.
    Array1::from_shape_fn(xi.len(), |t| xi[t] - k * xj[t])
}

/// Maximum-entropy approximation of differential entropy (`_entropy`).
fn entropy(u: &ArrayView1<f64>) -> f64 {
    const K1: f64 = 79.047;
    const K2: f64 = 7.4129;
    const GAMMA: f64 = 0.37457;
    let n = u.len() as f64;
    let t1 = u.iter().map(|x| x.cosh().ln()).sum::<f64>() / n;
    let t2 = u.iter().map(|x| x * (-(x * x) / 2.0).exp()).sum::<f64>() / n;
    (1.0 + (2.0 * std::f64::consts::PI).ln()) / 2.0
        - K1 * (t1 - GAMMA).powi(2)
        - K2 * t2.powi(2)
}

fn standardize(col: &ArrayView1<f64>) -> Array1<f64> {
    let m = mean(col);
    let s = pop_std(col);
    col.mapv(|v| (v - m) / s)
}

/// Index of the first maximum (matches `numpy.argmax`).
fn argmax_first(v: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// `_diff_mutual_info` with the two single-column entropies supplied by the caller
/// (they only depend on one standardized column, so they are hoisted out of the
/// O(|u|^2) pair loop and computed once per column).
fn diff_mutual_info_pre(
    ent_xi: f64,
    ent_xj: f64,
    ri_j: &ArrayView1<f64>,
    rj_i: &ArrayView1<f64>,
) -> f64 {
    let ri = ri_j.to_owned() / pop_std(ri_j);
    let rj = rj_i.to_owned() / pop_std(rj_i);
    (ent_xj + entropy(&ri.view())) - (ent_xi + entropy(&rj.view()))
}

/// `_search_causal_order` (no prior knowledge).
///
/// Two changes from a literal port, both bit-for-bit equivalent to the Python:
///   * `_diff_mutual_info` is antisymmetric (`d(j, i) == -d(i, j)` exactly, since
///     the two calls swap an operand pair of a single subtraction), so each
///     unordered pair is evaluated once and its score added to both entries.
///   * the per-pair work is spread across cores with rayon; the reduction is then
///     replayed serially in the original `for j in U` order so the accumulated
///     `M` values — and therefore `argmax` tie-breaking — are unchanged.
fn search_causal_order(x: &Array2<f64>, u: &[usize]) -> usize {
    let len = u.len();
    if len == 1 {
        return u[0];
    }

    let std_cols: Vec<Array1<f64>> = u.iter().map(|&i| standardize(&x.column(i))).collect();
    let ent_std: Vec<f64> = std_cols.iter().map(|c| entropy(&c.view())).collect();

    let pairs: Vec<(usize, usize)> = (0..len)
        .flat_map(|a| (a + 1..len).map(move |b| (a, b)))
        .collect();

    let score = |&(a, b): &(usize, usize)| -> (f64, f64) {
        let xa = std_cols[a].view();
        let xb = std_cols[b].view();
        let ra_b = residual(&xa, &xb);
        let rb_a = residual(&xb, &xa);
        let d = diff_mutual_info_pre(ent_std[a], ent_std[b], &ra_b.view(), &rb_a.view());
        (d.min(0.0).powi(2), (-d).min(0.0).powi(2))
    };

    let contribs: Vec<(f64, f64)> = if pairs.len() >= 16 {
        crate::pool::install(|| pairs.par_iter().map(score).collect())
    } else {
        pairs.iter().map(score).collect()
    };

    // Replay the accumulation in the original `for i in U: for j in U` order so
    // each `m[i]` is summed exactly as the Python loop sums it: for index `i`,
    // the pairs `(j, i)` for `j < i` land first (ascending `j`), then `(i, j)`
    // for `j > i`.
    let mut m = vec![0.0f64; len];
    let mut idx = 0;
    for a in 0..len {
        for b in (a + 1)..len {
            let (ca, cb) = contribs[idx];
            idx += 1;
            m[a] += ca;
            m[b] += cb;
        }
    }

    let m_list: Vec<f64> = m.iter().map(|&v| -v).collect();
    u[argmax_first(&m_list)]
}

/// Full DirectLiNGAM fit. `x` is the raw (unscaled) data. Returns the causal
/// order and the estimated adjacency matrix `B`.
pub fn fit(x: &Array2<f64>, adaptive_lasso: bool) -> (Vec<usize>, Array2<f64>) {
    let n_features = x.ncols();
    let mut u: Vec<usize> = (0..n_features).collect();
    let mut k: Vec<usize> = Vec::with_capacity(n_features);
    let mut x_ = x.to_owned();

    for _ in 0..n_features {
        let m = search_causal_order(&x_, &u);
        let xm = x_.column(m).to_owned();
        for &i in &u {
            if i != m {
                let xi = x_.column(i).to_owned();
                let r = residual(&xi.view(), &xm.view());
                x_.column_mut(i).assign(&r);
            }
        }
        k.push(m);
        u.retain(|&v| v != m);
    }

    let b = estimate_adjacency_matrix(x, &k, adaptive_lasso);
    (k, b)
}

/// `X - np.dot(B, X.T).T` — the estimated error/residual terms.
pub fn error_terms(x: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    x - &x.dot(&b.t())
}

/// Column means, used by callers that need `numpy.mean(X, axis=0)`.
#[allow(dead_code)]
pub fn col_means(x: &Array2<f64>) -> Array1<f64> {
    x.mean_axis(Axis(0)).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn toy_data() -> Array2<f64> {
        // x0 exogenous, x1 = 2 x0 + e1, x2 = -3 x1 + e2 ; deterministic-ish "noise"
        let n = 200;
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = i as f64;
            let e0 = (t * 0.7).sin();
            let e1 = (t * 1.3).cos() * 0.3;
            let e2 = (t * 2.1).sin() * 0.2;
            let x0 = e0;
            let x1 = 2.0 * x0 + e1;
            let x2 = -3.0 * x1 + e2;
            x[[i, 0]] = x0;
            x[[i, 1]] = x1;
            x[[i, 2]] = x2;
        }
        x
    }

    #[test]
    fn residual_removes_linear_dependence() {
        let x = toy_data();
        let r = residual(&x.column(1), &x.column(0));
        // cov(residual, x0) should be ~0
        let rm = r.mean().unwrap();
        let x0 = x.column(0);
        let x0m = x0.mean().unwrap();
        let cov = r
            .iter()
            .zip(x0.iter())
            .map(|(a, b)| (a - rm) * (b - x0m))
            .sum::<f64>()
            / r.len() as f64;
        assert!(cov.abs() < 1e-10, "cov = {cov}");
    }

    #[test]
    fn fit_finds_chain_order() {
        let x = toy_data();
        let (order, b) = fit(&x, false);
        assert_eq!(order, vec![0, 1, 2]);
        assert!((b[[1, 0]] - 2.0).abs() < 0.2);
        assert!((b[[2, 1]] + 3.0).abs() < 0.2);
        assert_eq!(b[[0, 1]], 0.0);
        assert_eq!(b[[0, 2]], 0.0);
        assert_eq!(b[[1, 2]], 0.0);
    }
}
