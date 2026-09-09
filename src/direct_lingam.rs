//! Core DirectLiNGAM causal-ordering (`measure="pwling"`) — a direct port of
//! `lingam/direct_lingam.py`.
//!
//! `prior_knowledge` is supported in both modes:
//!   * hard (`apply_prior_knowledge_softly=False`, the default) — partial orders
//!     are extracted from the knowledge matrix and restrict the candidate set
//!     `Uc` at each step (`_extract_partial_orders` / `_search_candidate`);
//!   * soft (`apply_prior_knowledge_softly=True`) — `_search_candidate` derives
//!     `Uc` / `Vj` from the (possibly `NaN`-valued) knowledge matrix and the
//!     pairwise search substitutes `x_std` for a residual when a variable is in
//!     `Vj` (`_search_causal_order`, soft branch).
//!
//! With `prior_knowledge=None` this reduces to `Uc = U`, `Vj = []` everywhere.

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rayon::prelude::*;
use std::collections::HashSet;

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

/// `_search_causal_order` for the no-prior-knowledge and hard-prior-knowledge
/// cases (the pairwise scores never substitute `x_std` for a residual, so the
/// antisymmetry optimisation below is valid). `candidates` is the candidate set
/// `Uc`: all of `U` without prior knowledge, or the subset with no as-yet-unplaced
/// ancestor when hard prior knowledge is in effect. The inner `M` sum still ranges
/// over the whole of `U`; only the final `argmax` is restricted to `Uc`.
///
/// Two changes from a literal port, both bit-for-bit equivalent to the Python:
///   * `_diff_mutual_info` is antisymmetric (`d(j, i) == -d(i, j)` exactly, since
///     the two calls swap an operand pair of a single subtraction), so each
///     unordered pair is evaluated once and its score added to both entries.
///   * the per-pair work is spread across cores with rayon; the reduction is then
///     replayed serially in the original `for j in U` order so the accumulated
///     `M` values — and therefore `argmax` tie-breaking — are unchanged.
fn search_causal_order(x: &Array2<f64>, u: &[usize], candidates: &[usize]) -> usize {
    if candidates.len() == 1 {
        return candidates[0];
    }
    let len = u.len();

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

    // `Uc[np.argmax(M_list)]` — `M_list` is built in `U` order over the candidate
    // set only, and `np.argmax` returns the first maximum.
    let cand: HashSet<usize> = candidates.iter().copied().collect();
    let mut best: Option<usize> = None;
    for pos in 0..len {
        if !cand.contains(&u[pos]) {
            continue;
        }
        match best {
            None => best = Some(pos),
            Some(b) if -m[pos] > -m[b] => best = Some(pos),
            _ => {}
        }
    }
    u[best.expect("search_causal_order: empty candidate set (inconsistent prior knowledge?)")]
}

/// `_search_candidate` — soft prior knowledge branch. Returns `(Uc, Vj)` derived
/// from the knowledge matrix `aknw` (with `-1` already replaced by `NaN`).
/// A plain `sum() == 0` test treats any `NaN` as "not zero" (mirroring numpy);
/// `nansum` skips `NaN`s.
fn search_candidate_soft(u: &[usize], aknw: &Array2<f64>) -> (Vec<usize>, Vec<usize>) {
    let sum_row = |j: usize| -> f64 { u.iter().filter(|&&x| x != j).map(|&c| aknw[[j, c]]).sum() };

    // Exogenous features: no known and no unknown incoming edge among `U`.
    let mut uc: Vec<usize> = u.iter().copied().filter(|&j| sum_row(j) == 0.0).collect();

    if uc.is_empty() {
        let mut u_end: Vec<usize> = Vec::new();
        // Endogenous features (a known incoming edge among `U`).
        for &j in u {
            let nansum: f64 = u
                .iter()
                .filter(|&&x| x != j)
                .map(|&c| aknw[[j, c]])
                .filter(|v| !v.is_nan())
                .sum();
            if nansum > 0.0 {
                u_end.push(j);
            }
        }
        // Sink features (no outgoing edge among `U`).
        for &i in u {
            let s: f64 = u.iter().filter(|&&x| x != i).map(|&r| aknw[[r, i]]).sum();
            if s == 0.0 {
                u_end.push(i);
            }
        }
        let end: HashSet<usize> = u_end.into_iter().collect();
        uc = u.iter().copied().filter(|i| !end.contains(i)).collect();
    }

    let uc_set: HashSet<usize> = uc.iter().copied().collect();
    let mut vj: Vec<usize> = Vec::new();
    for &i in u {
        if uc_set.contains(&i) {
            continue;
        }
        let s: f64 = uc.iter().map(|&c| aknw[[i, c]]).sum();
        if s == 0.0 {
            vj.push(i);
        }
    }
    (uc, vj)
}

/// `_search_causal_order` — soft prior knowledge branch. A literal port: when
/// `i in Vj and j in Uc` the "residual of `xi` on `xj`" is replaced by `xi_std`
/// itself, which breaks the antisymmetry the fast path above relies on, so this
/// is a plain double loop over `Uc` × `U`.
fn search_causal_order_soft(x: &Array2<f64>, u: &[usize], aknw: &Array2<f64>) -> usize {
    let (uc, vj) = search_candidate_soft(u, aknw);
    if uc.len() == 1 {
        return uc[0];
    }
    let uc_set: HashSet<usize> = uc.iter().copied().collect();
    let vj_set: HashSet<usize> = vj.iter().copied().collect();

    // Standardised columns and their single-column entropies, keyed by the
    // original column index (identical arithmetic to recomputing per pair).
    let std: Vec<(usize, Array1<f64>)> =
        u.iter().map(|&i| (i, standardize(&x.column(i)))).collect();
    let col = |i: usize| -> &Array1<f64> { &std.iter().find(|(k, _)| *k == i).unwrap().1 };
    let ent: Vec<(usize, f64)> = std.iter().map(|(i, c)| (*i, entropy(&c.view()))).collect();
    let ent_of = |i: usize| -> f64 { ent.iter().find(|(k, _)| *k == i).unwrap().1 };

    let mut m_list: Vec<f64> = Vec::with_capacity(uc.len());
    for &i in &uc {
        let mut m = 0.0f64;
        let xi = col(i);
        for &j in u {
            if i == j {
                continue;
            }
            let xj = col(j);
            let ri_j = if vj_set.contains(&i) && uc_set.contains(&j) {
                xi.clone()
            } else {
                residual(&xi.view(), &xj.view())
            };
            let rj_i = if vj_set.contains(&j) && uc_set.contains(&i) {
                xj.clone()
            } else {
                residual(&xj.view(), &xi.view())
            };
            let d = diff_mutual_info_pre(ent_of(i), ent_of(j), &ri_j.view(), &rj_i.view());
            m += d.min(0.0).powi(2);
        }
        m_list.push(-m);
    }
    uc[argmax_first(&m_list)]
}

/// Validated prior-knowledge state carried through a `fit`.
pub struct PriorKnowledge {
    /// `_Aknw` with `np.where(Aknw < 0, np.nan, Aknw)` already applied.
    aknw: Array2<f64>,
    apply_softly: bool,
    /// `_extract_partial_orders` output as `(from, to)` pairs ("from" is an
    /// ancestor of "to"). Empty in soft mode.
    partial_orders: Vec<(usize, usize)>,
}

impl PriorKnowledge {
    /// `check_array` has already run (finite, 2D) and negatives are `NaN`. Here we
    /// do the `fit`-time square-shape check and, for hard mode, extract the
    /// partial orders (which can fail on an inconsistent matrix).
    pub fn new(
        aknw: Array2<f64>,
        apply_softly: bool,
        n_features: usize,
    ) -> Result<Self, String> {
        if aknw.shape() != [n_features, n_features] {
            return Err(
                "The shape of prior knowledge must be (n_features, n_features)".to_string(),
            );
        }
        let partial_orders = if apply_softly {
            Vec::new()
        } else {
            extract_partial_orders(&aknw)?
        };
        Ok(Self {
            aknw,
            apply_softly,
            partial_orders,
        })
    }
}

/// Row/column index pairs `(i, j)` where `pk[i, j] == val`, in C (row-major)
/// order — i.e. `np.array(np.where(pk == val)).transpose()`.
fn where_eq(pk: &Array2<f64>, val: f64) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..pk.nrows() {
        for j in 0..pk.ncols() {
            if pk[[i, j]] == val {
                out.push((i, j));
            }
        }
    }
    out
}

fn format_index_pairs(rows: &[Vec<i64>]) -> String {
    let inner: Vec<String> = rows.iter().map(|r| format!("[{}, {}]", r[0], r[1])).collect();
    format!("[{}]", inner.join(", "))
}

/// `_extract_partial_orders` — returns the partial orders as `(from, to)` pairs.
/// Errors (like the Python `ValueError`) when the matrix asserts both `i -> j`
/// and `j -> i`.
fn extract_partial_orders(pk: &Array2<f64>) -> Result<Vec<(usize, usize)>, String> {
    let path_pairs = where_eq(pk, 1.0);
    let mut no_path_pairs = where_eq(pk, 0.0);

    // Inconsistencies among pairs with a path: (i, j) and (j, i) both present.
    if !path_pairs.is_empty() {
        let mut check: Vec<Vec<i64>> = Vec::with_capacity(path_pairs.len() * 2);
        for &(i, j) in &path_pairs {
            check.push(vec![i as i64, j as i64]);
        }
        for &(i, j) in &path_pairs {
            check.push(vec![j as i64, i as i64]);
        }
        let (uniq, counts) = crate::util::unique_rows(check);
        let bad: Vec<Vec<i64>> = uniq
            .into_iter()
            .zip(counts)
            .filter(|(_, c)| *c > 1)
            .map(|(r, _)| r)
            .collect();
        if !bad.is_empty() {
            return Err(format!(
                "The prior knowledge contains inconsistencies at the following indices: {}",
                format_index_pairs(&bad)
            ));
        }
    }

    // Pairs without a path that are symmetric (`pk[i, j] == pk[j, i] == 0`)
    // "cancel out and are not ordered".
    if !no_path_pairs.is_empty() {
        let mut check: Vec<Vec<i64>> = Vec::with_capacity(no_path_pairs.len() * 2);
        for &(i, j) in &no_path_pairs {
            check.push(vec![i as i64, j as i64]);
        }
        for &(i, j) in &no_path_pairs {
            check.push(vec![j as i64, i as i64]);
        }
        let (uniq, counts) = crate::util::unique_rows(check);
        let mut check2: Vec<Vec<i64>> = no_path_pairs
            .iter()
            .map(|&(i, j)| vec![i as i64, j as i64])
            .collect();
        for (r, c) in uniq.into_iter().zip(counts) {
            if c > 1 {
                check2.push(r);
            }
        }
        let (uniq2, counts2) = crate::util::unique_rows(check2);
        no_path_pairs = uniq2
            .into_iter()
            .zip(counts2)
            .filter(|(_, c)| *c < 2)
            .map(|(r, _)| (r[0] as usize, r[1] as usize))
            .collect();
    }

    // `check_pairs` rows are `[to, from]`: path pairs `[i, j]` (meaning `j -> i`)
    // and the surviving no-path pairs swapped.
    let mut check_pairs: Vec<Vec<i64>> = Vec::new();
    for &(i, j) in &path_pairs {
        check_pairs.push(vec![i as i64, j as i64]);
    }
    for &(i, j) in &no_path_pairs {
        check_pairs.push(vec![j as i64, i as i64]);
    }
    if check_pairs.is_empty() {
        return Ok(Vec::new());
    }
    let (uniq, _) = crate::util::unique_rows(check_pairs);
    Ok(uniq
        .into_iter()
        .map(|r| (r[1] as usize, r[0] as usize)) // [to, from] -> (from, to)
        .collect())
}

/// Full DirectLiNGAM fit. `x` is the raw (unscaled) data. Returns the causal
/// order and the estimated adjacency matrix `B`.
pub fn fit(
    x: &Array2<f64>,
    adaptive_lasso: bool,
    prior_knowledge: Option<&PriorKnowledge>,
) -> (Vec<usize>, Array2<f64>) {
    let n_features = x.ncols();
    let mut u: Vec<usize> = (0..n_features).collect();
    let mut k: Vec<usize> = Vec::with_capacity(n_features);
    let mut x_ = x.to_owned();

    let hard_pk = prior_knowledge.filter(|p| !p.apply_softly);
    let soft_pk = prior_knowledge.filter(|p| p.apply_softly);
    // Mutable copy of the partial orders, trimmed as variables are placed.
    let mut partial_orders: Vec<(usize, usize)> =
        hard_pk.map(|p| p.partial_orders.clone()).unwrap_or_default();

    for _ in 0..n_features {
        let m = if let Some(p) = soft_pk {
            search_causal_order_soft(&x_, &u, &p.aknw)
        } else if hard_pk.is_some() && !partial_orders.is_empty() {
            // `Uc = [i for i in U if i not in partial_orders[:, 1]]`.
            let candidates: Vec<usize> = u
                .iter()
                .copied()
                .filter(|i| !partial_orders.iter().any(|&(_, to)| to == *i))
                .collect();
            search_causal_order(&x_, &u, &candidates)
        } else {
            search_causal_order(&x_, &u, &u)
        };

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
        if hard_pk.is_some() {
            // `partial_orders = partial_orders[partial_orders[:, 0] != m]`.
            partial_orders.retain(|&(from, _)| from != m);
        }
    }

    let b = estimate_adjacency_matrix(
        x,
        &k,
        adaptive_lasso,
        prior_knowledge.map(|p| &p.aknw),
    );
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
        let (order, b) = fit(&x, false, None);
        assert_eq!(order, vec![0, 1, 2]);
        assert!((b[[1, 0]] - 2.0).abs() < 0.2);
        assert!((b[[2, 1]] + 3.0).abs() < 0.2);
        assert_eq!(b[[0, 1]], 0.0);
        assert_eq!(b[[0, 2]], 0.0);
        assert_eq!(b[[1, 2]], 0.0);
    }

    /// `pk[i][j] == 1` means `x_j -> x_i`.
    fn pk_unknown(p: usize) -> Array2<f64> {
        Array2::from_elem((p, p), f64::NAN)
    }

    #[test]
    fn extract_partial_orders_from_paths() {
        let mut pk = pk_unknown(3);
        pk[[1, 0]] = 1.0; // 0 -> 1
        pk[[2, 1]] = 1.0; // 1 -> 2
        let po = extract_partial_orders(&pk).unwrap();
        assert_eq!(po, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn extract_partial_orders_from_no_path() {
        let mut pk = pk_unknown(2);
        pk[[0, 1]] = 0.0; // 1 does NOT -> 0
        // pk[1, 0] stays unknown, so 0 -> 1 is possible: order (0, 1).
        let po = extract_partial_orders(&pk).unwrap();
        assert_eq!(po, vec![(0, 1)]);
    }

    #[test]
    fn extract_partial_orders_rejects_inconsistency() {
        let mut pk = pk_unknown(2);
        pk[[0, 1]] = 1.0; // 1 -> 0
        pk[[1, 0]] = 1.0; // 0 -> 1
        assert!(extract_partial_orders(&pk).is_err());
    }

    #[test]
    fn hard_prior_knowledge_forces_reverse_order() {
        let x = toy_data(); // data-generating order is 0 -> 1 -> 2
        let mut pk = pk_unknown(3);
        pk[[0, 2]] = 1.0; // 2 -> 0
        pk[[1, 2]] = 1.0; // 2 -> 1
        pk[[0, 1]] = 1.0; // 1 -> 0
        let pknw = PriorKnowledge::new(pk, false, 3).unwrap();
        let (order, _b) = fit(&x, false, Some(&pknw));
        assert_eq!(order, vec![2, 1, 0]);
    }

    #[test]
    fn hard_prior_knowledge_prunes_adjacency() {
        let x = toy_data();
        let mut pk = pk_unknown(3);
        // Symmetric "no path" between 1 and 2: cancels out as an ordering
        // constraint but still prunes the 1 -> 2 edge from B.
        pk[[2, 1]] = 0.0;
        pk[[1, 2]] = 0.0;
        let pknw = PriorKnowledge::new(pk, false, 3).unwrap();
        let (order, b) = fit(&x, false, Some(&pknw));
        assert_eq!(order, vec![0, 1, 2]);
        assert_eq!(b[[2, 1]], 0.0);
        // the 2 <- 0 edge is still free to be estimated
        assert!(b[[2, 0]].abs() > 0.0);
    }

    #[test]
    fn soft_prior_knowledge_orders_chain() {
        let x = toy_data();
        let mut pk = pk_unknown(3);
        pk[[1, 0]] = 1.0; // 0 -> 1
        pk[[2, 1]] = 1.0; // 1 -> 2
        let pknw = PriorKnowledge::new(pk, true, 3).unwrap();
        let (order, _b) = fit(&x, false, Some(&pknw));
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn prior_knowledge_shape_is_checked() {
        assert!(PriorKnowledge::new(pk_unknown(3), false, 4).is_err());
    }
}
