//! Small self-contained numerical helpers so the crate stays free of heavy
//! linear-algebra / statistics dependencies (keeps the manylinux CI matrix happy).

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// splitmix64 -> xoshiro256** RNG. Only what `bootstrap` needs: an unbiased-enough
/// `usize` in `0..n`. This is *not* NumPy/sklearn's stream, so per-seed resampled
/// indices differ from `lingam` (documented).
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n` (n > 0).
    pub fn gen_index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Natural log of the Gamma function (Lanczos approximation, g = 7).
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    #[allow(clippy::excessive_precision)]
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection formula
        (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Regularized lower incomplete gamma P(a, x) via series expansion (x < a + 1).
fn gser(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let gln = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..1000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

/// Regularized upper incomplete gamma Q(a, x) via continued fraction (x >= a + 1).
fn gcf(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).
/// Equivalent to `scipy.special.gammaincc(a, x)`.
pub fn gammq(a: f64, x: f64) -> f64 {
    if !a.is_finite() || !x.is_finite() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gser(a, x)
    } else {
        gcf(a, x)
    }
}

/// `scipy.stats.gamma.sf(x, a, scale=scale)`.
pub fn gamma_sf(x: f64, a: f64, scale: f64) -> f64 {
    gammq(a, x / scale)
}

/// Median of a slice (linear interpolation of the two middle values for even n),
/// matching `numpy.median`.
pub fn median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Indices that would sort `keys` in descending order, stable on ties
/// (a small deviation from `numpy.argsort` which is not stable).
pub fn argsort_desc(keys: &[i64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by(|&i, &j| keys[j].cmp(&keys[i]));
    idx
}

/// Indices that would sort `keys` in descending order, stable on ties.
pub fn argsort_desc_f64(keys: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by(|&i, &j| keys[j].partial_cmp(&keys[i]).unwrap());
    idx
}

/// Lexicographically sorted unique rows and their counts, matching
/// `numpy.unique(rows, axis=0, return_counts=True)`.
pub fn unique_rows(mut rows: Vec<Vec<i64>>) -> (Vec<Vec<i64>>, Vec<i64>) {
    rows.sort();
    let mut uniq: Vec<Vec<i64>> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();
    for r in rows {
        if uniq.last().map(|u| u == &r).unwrap_or(false) {
            *counts.last_mut().unwrap() += 1;
        } else {
            uniq.push(r);
            counts.push(1);
        }
    }
    (uniq, counts)
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
/// Returns `None` if the matrix is (numerically) singular.
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());
    debug_assert_eq!(n, b.len());
    let mut m = a.clone();
    let mut rhs = b.clone();

    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = m[[col, col]].abs();
        for r in (col + 1)..n {
            let v = m[[r, col]].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                m.swap([col, c], [piv, c]);
            }
            rhs.swap(col, piv);
        }
        let pivot = m[[col, col]];
        for r in (col + 1)..n {
            let factor = m[[r, col]] / pivot;
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                let sub = factor * m[[col, c]];
                m[[r, c]] -= sub;
            }
            rhs[r] -= factor * rhs[col];
        }
    }

    // back substitution
    let mut x = Array1::zeros(n);
    for row in (0..n).rev() {
        let mut s = rhs[row];
        for c in (row + 1)..n {
            s -= m[[row, c]] * x[c];
        }
        x[row] = s / m[[row, row]];
    }
    Some(x)
}

/// Ordinary least squares with intercept (like `sklearn.linear_model.LinearRegression`),
/// returning only the coefficients (intercept discarded). Solved through the normal
/// equations on mean-centered data; for the well-conditioned sub-problems DirectLiNGAM
/// produces this matches `scipy.linalg.lstsq` closely.
pub fn linreg_coef(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let p = x.ncols();
    if p == 0 {
        return Array1::zeros(0);
    }
    let col_means: Array1<f64> = x.mean_axis(ndarray::Axis(0)).unwrap();
    let y_mean = y.mean().unwrap();

    let mut xc = x.clone();
    for j in 0..p {
        let m = col_means[j];
        xc.column_mut(j).mapv_inplace(|v| v - m);
    }
    let yc: Array1<f64> = y.mapv(|v| v - y_mean);

    let xtx = xc.t().dot(&xc);
    let xty = xc.t().dot(&yc);

    if let Some(sol) = solve(&xtx, &xty) {
        return sol;
    }
    // Tikhonov fallback for rank-deficient design.
    let trace: f64 = (0..p).map(|i| xtx[[i, i]]).sum();
    let ridge = (trace / p as f64).max(1.0) * 1e-10;
    let mut xtx_r = xtx;
    for i in 0..p {
        xtx_r[[i, i]] += ridge;
    }
    solve(&xtx_r, &xty).unwrap_or_else(|| Array1::zeros(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gammq_matches_known_values() {
        // Q(1, x) = exp(-x)
        assert!((gammq(1.0, 2.0) - (-2.0f64).exp()).abs() < 1e-12);
        assert!((gammq(1.0, 0.5) - (-0.5f64).exp()).abs() < 1e-12);
        // Q(a, 0) = 1
        assert_eq!(gammq(3.2, 0.0), 1.0);
        // Q(0.5, 1.0) = erfc(1) = 0.15729920705028513
        assert!((gammq(0.5, 1.0) - 0.157_299_207_050_285).abs() < 1e-9);
    }

    #[test]
    fn median_even_and_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn solve_2x2() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![3.0, 5.0];
        let x = solve(&a, &b).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12);
        assert!((x[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn linreg_recovers_slope_and_ignores_intercept() {
        // y = 2 x0 - 3 x1 + 5
        let x = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 4.0]
        ];
        let y = x.dot(&array![2.0, -3.0]).mapv(|v| v + 5.0);
        let coef = linreg_coef(&x, &y);
        assert!((coef[0] - 2.0).abs() < 1e-9);
        assert!((coef[1] + 3.0).abs() < 1e-9);
    }
}
