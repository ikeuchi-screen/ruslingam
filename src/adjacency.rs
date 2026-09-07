//! Adjacency-matrix estimation: `_estimate_adjacency_matrix` (base.py) and
//! `predict_adaptive_lasso` (utils/__init__.py).

use ndarray::{Array1, Array2, Axis};

use crate::lars::lasso_lars_ic_bic;
use crate::util::linreg_coef;

fn select_columns(x: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let n = x.nrows();
    let mut out = Array2::zeros((n, cols.len()));
    for (j, &c) in cols.iter().enumerate() {
        out.column_mut(j).assign(&x.column(c));
    }
    out
}

/// `sklearn.preprocessing.StandardScaler().fit_transform` — centre and scale each
/// column by its population standard deviation (zero-variance columns keep scale 1).
fn standard_scale(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows() as f64;
    let means = x.mean_axis(Axis(0)).unwrap();
    let mut out = x.clone();
    for j in 0..x.ncols() {
        let m = means[j];
        let var = x.column(j).iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
        let mut s = var.sqrt();
        if s == 0.0 {
            s = 1.0;
        }
        out.column_mut(j).mapv_inplace(|v| (v - m) / s);
    }
    out
}

/// `predict_adaptive_lasso(X, predictors, target, gamma=1.0)`.
fn predict_adaptive_lasso(x: &Array2<f64>, predictors: &[usize], target: usize) -> Array1<f64> {
    let x_std = standard_scale(x);
    let xp_std = select_columns(&x_std, predictors);
    let y_std = x_std.column(target).to_owned();

    let lr_coef = linreg_coef(&xp_std, &y_std);
    let weight = lr_coef.mapv(|c| c.abs()); // ** gamma, gamma = 1.0

    let mut design = xp_std.clone();
    for j in 0..design.ncols() {
        let w = weight[j];
        design.column_mut(j).mapv_inplace(|v| v * w);
    }

    let reg_coef = lasso_lars_ic_bic(&design, &y_std);

    let pruned: Vec<bool> = reg_coef
        .iter()
        .zip(weight.iter())
        .map(|(c, w)| (c * w).abs() > 0.0)
        .collect();

    let mut coef = Array1::zeros(predictors.len());
    let sel: Vec<usize> = pruned
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();
    if !sel.is_empty() {
        let orig_cols: Vec<usize> = sel.iter().map(|&i| predictors[i]).collect();
        let xsel = select_columns(x, &orig_cols);
        let y = x.column(target).to_owned();
        let c = linreg_coef(&xsel, &y);
        for (k, &i) in sel.iter().enumerate() {
            coef[i] = c[k];
        }
    }
    coef
}

/// `_estimate_adjacency_matrix(X, prior_knowledge=None, adaptive_lasso=...)`.
pub fn estimate_adjacency_matrix(
    x: &Array2<f64>,
    causal_order: &[usize],
    adaptive_lasso: bool,
) -> Array2<f64> {
    let p = x.ncols();
    let mut b = Array2::zeros((p, p));

    for i in 1..causal_order.len() {
        let target = causal_order[i];
        let predictors = &causal_order[..i];
        if predictors.is_empty() {
            continue;
        }

        let coef = if adaptive_lasso {
            predict_adaptive_lasso(x, predictors, target)
        } else {
            let xp = select_columns(x, predictors);
            let y = x.column(target).to_owned();
            linreg_coef(&xp, &y)
        };

        for (idx, &pred) in predictors.iter().enumerate() {
            b[[target, pred]] = coef[idx];
        }
    }
    b
}

/// `estimate_total_effect` (base.py): regress `to` on `[from] + parents(from)`,
/// return the coefficient on `from`. Returns `(effect, order_ok)` where
/// `order_ok` is false when `from` comes after `to` in the causal order.
pub fn estimate_total_effect(
    x: &Array2<f64>,
    b: &Array2<f64>,
    causal_order: &[usize],
    from_index: usize,
    to_index: usize,
) -> (f64, bool) {
    let from_order = causal_order.iter().position(|&v| v == from_index).unwrap();
    let to_order = causal_order.iter().position(|&v| v == to_index).unwrap();
    let order_ok = from_order <= to_order;

    let mut predictors = vec![from_index];
    for k in 0..b.ncols() {
        if b[[from_index, k]].abs() > 0.0 {
            predictors.push(k);
        }
    }

    let xp = select_columns(x, &predictors);
    let y = x.column(to_index).to_owned();
    let coef = linreg_coef(&xp, &y);
    (coef[0], order_ok)
}
