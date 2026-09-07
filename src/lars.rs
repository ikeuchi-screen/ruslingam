//! `LassoLarsIC(criterion="bic")` reproduced without scikit-learn:
//! a LARS-LASSO coefficient path followed by BIC model selection over its knots.

use ndarray::{Array1, Array2};

use crate::util::{linreg_coef, solve};

/// LARS-LASSO regularisation path (`sklearn.linear_model.lars_path(..., method="lasso",
/// alpha_min=0.0)`). Returns the coefficient vector at every knot, starting with the
/// all-zeros solution. `x` and `y` are expected to be mean-centered.
fn lars_path_lasso(x: &Array2<f64>, y: &Array1<f64>) -> Vec<Array1<f64>> {
    let p = x.ncols();
    let mut beta = Array1::<f64>::zeros(p);
    let mut path = vec![beta.clone()];
    if p == 0 {
        return path;
    }

    let mut mu = Array1::<f64>::zeros(x.nrows());
    let mut active: Vec<usize> = Vec::new();
    let mut sign: Vec<f64> = Vec::new();
    let mut dropped_prev = false;

    let tol = 1e-11;
    let max_iter = 8 * p + 200;

    for _ in 0..max_iter {
        let resid = y - &mu;
        let c = x.t().dot(&resid);
        let big_c = c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if big_c < tol {
            break;
        }

        if !dropped_prev {
            let mut j_add = None;
            let mut best = big_c - tol;
            for j in 0..p {
                if active.contains(&j) {
                    continue;
                }
                if c[j].abs() > best {
                    best = c[j].abs();
                    j_add = Some(j);
                }
            }
            match j_add {
                Some(j) => {
                    active.push(j);
                    sign.push(if c[j] >= 0.0 { 1.0 } else { -1.0 });
                }
                None => break,
            }
        }

        let k = active.len();
        // sign-adjusted active design
        let mut xa = Array2::<f64>::zeros((x.nrows(), k));
        for (col, (&j, &s)) in active.iter().zip(sign.iter()).enumerate() {
            let sj = s;
            xa.column_mut(col)
                .assign(&x.column(j).mapv(|v| v * sj));
        }
        let g = xa.t().dot(&xa);
        let ones = Array1::<f64>::ones(k);
        let ua_raw = match solve(&g, &ones) {
            Some(v) => v,
            None => break,
        };
        let sum_ua = ua_raw.sum();
        if sum_ua <= 0.0 {
            break;
        }
        let aa = 1.0 / sum_ua.sqrt();
        let w = ua_raw.mapv(|v| v * aa); // (k,)
        let u = xa.dot(&w); // (n,)
        let a = x.t().dot(&u); // (p,)

        // LARS step length: smallest positive blocking value among inactive vars.
        let mut gamma = big_c / aa;
        for j in 0..p {
            if active.contains(&j) {
                continue;
            }
            for &(num, den) in &[(big_c - c[j], aa - a[j]), (big_c + c[j], aa + a[j])] {
                if den > tol {
                    let t = num / den;
                    if t > tol && t < gamma {
                        gamma = t;
                    }
                }
            }
        }

        // LASSO modification: a coefficient hitting zero before `gamma`.
        let mut gamma_tilde = f64::INFINITY;
        let mut drop_k = None;
        for (idx, &j) in active.iter().enumerate() {
            let d = sign[idx] * w[idx];
            if d.abs() < tol {
                continue;
            }
            let t = -beta[j] / d;
            if t > tol && t < gamma_tilde {
                gamma_tilde = t;
                drop_k = Some(idx);
            }
        }

        let (step, do_drop) = if gamma_tilde < gamma {
            (gamma_tilde, true)
        } else {
            (gamma, false)
        };

        for (idx, &j) in active.iter().enumerate() {
            beta[j] += step * sign[idx] * w[idx];
        }
        mu = &mu + &u.mapv(|v| v * step);

        if do_drop {
            let idx = drop_k.unwrap();
            beta[active[idx]] = 0.0;
            active.remove(idx);
            sign.remove(idx);
        }

        path.push(beta.clone());
        dropped_prev = do_drop;

        if !do_drop && active.len() == p {
            break;
        }
    }

    path
}

/// `LassoLarsIC(criterion="bic").fit(x, y).coef_`.
pub fn lasso_lars_ic_bic(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    if p == 0 {
        return Array1::zeros(0);
    }

    // _preprocess_data: centre X and y.
    let col_means = x.mean_axis(ndarray::Axis(0)).unwrap();
    let y_mean = y.mean().unwrap();
    let mut xc = x.clone();
    for j in 0..p {
        let m = col_means[j];
        xc.column_mut(j).mapv_inplace(|v| v - m);
    }
    let yc: Array1<f64> = y.mapv(|v| v - y_mean);

    // Noise variance: OLS (no intercept, data already centered), RSS / (n - p - 1).
    // sklearn raises when n <= p + 1; we fall back to plain OLS coefficients instead.
    if n <= p + 1 {
        return linreg_coef(x, y);
    }
    let ols = linreg_coef(x, y); // intercept handled internally == centered no-intercept
    let pred = xc.dot(&ols);
    let rss_full: f64 = yc
        .iter()
        .zip(pred.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let noise_var = rss_full / (n as f64 - p as f64 - 1.0);
    if noise_var <= 0.0 || noise_var.is_nan() {
        return ols;
    }

    let path = lars_path_lasso(&xc, &yc);

    let n_f = n as f64;
    let factor = n_f.ln(); // BIC
    let eps = f64::EPSILON;
    let mut best_k = 0usize;
    let mut best_crit = f64::INFINITY;
    for (k, coef) in path.iter().enumerate() {
        let pred_k = xc.dot(coef);
        let rss: f64 = yc
            .iter()
            .zip(pred_k.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let df = coef.iter().filter(|&&v| v.abs() > eps).count() as f64;
        let crit = n_f * (2.0 * std::f64::consts::PI * noise_var).ln() + rss / noise_var
            + factor * df;
        if crit < best_crit {
            best_crit = crit;
            best_k = k;
        }
    }

    path[best_k].clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn ic_bic_drops_irrelevant_predictor() {
        // y depends only on column 0; column 1 is pure noise.
        let n = 400;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = ndarray::Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64;
            let a = (t * 0.37).sin();
            let noise = (t * 1.91).cos() * 0.9;
            x[[i, 0]] = a;
            x[[i, 1]] = noise;
            y[i] = 4.0 * a + (t * 0.05).sin() * 0.01;
        }
        let coef = lasso_lars_ic_bic(&x, &y);
        assert!((coef[0] - 4.0).abs() < 0.1, "coef0 = {}", coef[0]);
        assert_eq!(coef[1], 0.0, "irrelevant predictor should be pruned");
    }

    #[test]
    fn ic_bic_keeps_two_real_predictors() {
        let n = 500;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = ndarray::Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64;
            let a = (t * 0.37).sin();
            let b = (t * 0.11).cos();
            x[[i, 0]] = a;
            x[[i, 1]] = b;
            y[i] = 2.0 * a - 1.5 * b;
        }
        let coef = lasso_lars_ic_bic(&x, &y);
        assert!((coef[0] - 2.0).abs() < 0.05);
        assert!((coef[1] + 1.5).abs() < 0.05);
    }
}
