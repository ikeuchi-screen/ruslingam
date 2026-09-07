//! Core DirectLiNGAM causal-ordering (`measure="pwling"`) — a direct port of
//! `lingam/direct_lingam.py` with `prior_knowledge=None`
//! (so `Uc = U` and `Vj = []` everywhere).

use ndarray::{Array1, Array2, ArrayView1, Axis};

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
    &xi.to_owned() - &xj.mapv(|b| k * b)
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

/// `_diff_mutual_info`
fn diff_mutual_info(
    xi_std: &ArrayView1<f64>,
    xj_std: &ArrayView1<f64>,
    ri_j: &ArrayView1<f64>,
    rj_i: &ArrayView1<f64>,
) -> f64 {
    let ri = ri_j.to_owned() / pop_std(ri_j);
    let rj = rj_i.to_owned() / pop_std(rj_i);
    (entropy(xj_std) + entropy(&ri.view())) - (entropy(xi_std) + entropy(&rj.view()))
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

/// `_search_causal_order` (no prior knowledge).
fn search_causal_order(x: &Array2<f64>, u: &[usize]) -> usize {
    if u.len() == 1 {
        return u[0];
    }

    let std_cols: Vec<Array1<f64>> = u.iter().map(|&i| standardize(&x.column(i))).collect();

    let mut m_list = Vec::with_capacity(u.len());
    for (ii, &_i) in u.iter().enumerate() {
        let xi_std = &std_cols[ii];
        let mut m = 0.0f64;
        for (jj, &_j) in u.iter().enumerate() {
            if ii == jj {
                continue;
            }
            let xj_std = &std_cols[jj];
            let ri_j = residual(&xi_std.view(), &xj_std.view());
            let rj_i = residual(&xj_std.view(), &xi_std.view());
            let d = diff_mutual_info(
                &xi_std.view(),
                &xj_std.view(),
                &ri_j.view(),
                &rj_i.view(),
            );
            m += d.min(0.0).powi(2);
        }
        m_list.push(-m); // Python: M_list.append(-1.0 * M)
    }
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
