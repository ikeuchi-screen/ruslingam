//! `ruslingam.CAMUV` — a Rust port of `lingam.CAMUV` (`independence="hsic"`
//! only; see the constructor). Structured to mirror `lingam/camuv.py` function
//! by function, the same convention `direct_lingam.rs` uses for `DirectLiNGAM`.
//!
//! Unlike `DirectLiNGAM`, `lingam.CAMUV` exposes only `fit(X)` and
//! `adjacency_matrix_` (confirmed by reading the full Python source — no
//! `causal_order_`, `bootstrap`, or `estimate_total_effect`), so that's all this
//! module implements.

use ndarray::{Array1, Array2, ArrayView2};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::gam::gam_residual;
use crate::hsic::{hsic_test_gamma, hsic_test_gamma_1d};

/// `prior_knowledge` parsed into `_make_pk_dict`'s shape: keyed by the
/// "child"/`to` index, valued by the set of `from` indices forbidden as a cause
/// of it. Kept as `i64` (rather than `usize`) so that out-of-range entries (the
/// Python version never validates `prior_knowledge` contents) are simply inert —
/// they can never equal a real `0..d` column index — instead of needing a
/// fallible cast.
pub type PriorKnowledge = HashMap<i64, HashSet<i64>>;

/// Ascending-order combinations of size `t` chosen from `0..d`. Equivalent to
/// `itertools.combinations(range(d), t)`, and — since CPython's `set` of small
/// non-negative ints iterates in ascending order (verified empirically) — to
/// `itertools.combinations(set(range(d)), t)`, which is what `_find_parents`
/// actually calls. `t == 0` never occurs in practice (the search always starts
/// at `t = 2`), but is handled for completeness.
fn combinations(d: usize, t: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if t == 0 {
        out.push(Vec::new());
        return out;
    }
    if t > d {
        return out;
    }
    let mut combo: Vec<usize> = (0..t).collect();
    loop {
        out.push(combo.clone());
        let mut i = t as isize - 1;
        while i >= 0 && combo[i as usize] == i as usize + d - t {
            i -= 1;
        }
        if i < 0 {
            break;
        }
        let i = i as usize;
        combo[i] += 1;
        for j in (i + 1)..t {
            combo[j] = combo[j - 1] + 1;
        }
    }
    out
}

/// `_get_residual(X, explained_i, explanatory_ids)`. `explanatory` need not be
/// sorted (column order doesn't change the GAM fit — each column gets its own
/// independent spline term — but callers pass it sorted for determinism).
fn get_residual(x: &Array2<f64>, explained: usize, explanatory: &[usize]) -> Array1<f64> {
    if explanatory.is_empty() {
        return x.column(explained).to_owned();
    }
    let n = x.nrows();
    let mut xe = Array2::<f64>::zeros((n, explanatory.len()));
    for (c, &col) in explanatory.iter().enumerate() {
        xe.column_mut(c).assign(&x.column(col));
    }
    gam_residual(xe.view(), x.column(explained))
}

/// `_is_independent` / `_is_independent_by` for `independence="hsic"`:
/// independent iff the HSIC gamma-test p-value exceeds `alpha`.
fn is_independent_1d(alpha: f64, x: &[f64], y: &[f64]) -> bool {
    hsic_test_gamma_1d(x, y).1 > alpha
}

/// `_check_prior_knowledge(xj_list, xi)`.
fn check_prior_knowledge(pk: Option<&PriorKnowledge>, xj_list: &HashSet<usize>, xi: usize) -> bool {
    let Some(pk) = pk else { return false };
    let Some(forbidden) = pk.get(&(xi as i64)) else {
        return false;
    };
    xj_list.iter().any(|&xj| forbidden.contains(&(xj as i64)))
}

/// `_check_correlation(child, parents, N)`.
fn check_correlation(
    child: usize,
    parents: &HashSet<usize>,
    neighborhoods: &[HashSet<usize>],
) -> bool {
    parents.iter().all(|p| neighborhoods[child].contains(p))
}

/// `_check_identified_causality(variables_set, P)`. `variables_set` must be in
/// ascending order (as produced by [`combinations`]), matching Python's
/// `list(variables_set)` over a `set` of small ints.
fn check_identified_causality(variables_set: &[usize], p: &[HashSet<usize>]) -> bool {
    for a in 0..variables_set.len() {
        for b in (a + 1)..variables_set.len() {
            let (i, j) = (variables_set[a], variables_set[b]);
            if p[i].contains(&j) || p[j].contains(&i) {
                return false;
            }
        }
    }
    true
}

/// `_get_neighborhoods(X)` — all-pairs HSIC test on raw columns. Each pair is
/// independent of every other (no shared mutable state), so this runs on the
/// thread pool.
fn get_neighborhoods(x: &Array2<f64>, alpha: f64) -> Vec<HashSet<usize>> {
    let d = x.ncols();
    let cols: Vec<Vec<f64>> = (0..d).map(|c| x.column(c).to_vec()).collect();
    let pairs: Vec<(usize, usize)> = (0..d)
        .flat_map(|i| ((i + 1)..d).map(move |j| (i, j)))
        .collect();
    let dependent: Vec<bool> = crate::pool::install(|| {
        pairs
            .par_iter()
            .map(|&(i, j)| !is_independent_1d(alpha, &cols[i], &cols[j]))
            .collect()
    });
    let mut neighborhoods = vec![HashSet::new(); d];
    for (&(i, j), &dep) in pairs.iter().zip(dependent.iter()) {
        if dep {
            neighborhoods[i].insert(j);
            neighborhoods[j].insert(i);
        }
    }
    neighborhoods
}

/// `_get_child(X, variables_set, P, N, Y)` — pick the child (among
/// `variables_set`) whose residual (regressed on the rest of `variables_set`
/// plus its already-known parents) is *most* independent of `Y[:, parents]`;
/// first-strictly-greater wins ties (same idiom as
/// `direct_lingam::search_causal_order`'s argmax). `parents` can have more than
/// one column once `num_explanatory_vals > 2`, hence the general (not
/// single-column) [`hsic_test_gamma`].
fn get_child(
    x: &Array2<f64>,
    variables_set: &[usize],
    p: &[HashSet<usize>],
    neighborhoods: &[HashSet<usize>],
    y: &Array2<f64>,
    alpha: f64,
    pk: Option<&PriorKnowledge>,
) -> (Option<usize>, bool) {
    let n = x.nrows();
    let mut prev_independence = 0.0f64;
    let mut max_child: Option<usize> = None;

    for &child in variables_set {
        let parents: HashSet<usize> = variables_set
            .iter()
            .copied()
            .filter(|&v| v != child)
            .collect();
        if check_prior_knowledge(pk, &parents, child) {
            continue;
        }
        if !check_correlation(child, &parents, neighborhoods) {
            continue;
        }

        let mut explanatory: Vec<usize> = parents.union(&p[child]).copied().collect();
        explanatory.sort_unstable();
        let residual = get_residual(x, child, &explanatory);

        let mut parent_cols: Vec<usize> = parents.iter().copied().collect();
        parent_cols.sort_unstable();
        let mut y_sel = Array2::<f64>::zeros((n, parent_cols.len()));
        for (c, &pc) in parent_cols.iter().enumerate() {
            y_sel.column_mut(c).assign(&y.column(pc));
        }

        let x_view = ArrayView2::from_shape((n, 1), residual.as_slice().unwrap()).unwrap();
        let (_, value) = hsic_test_gamma(x_view, y_sel.view());
        if value > prev_independence {
            prev_independence = value;
            max_child = Some(child);
        }
    }

    let is_independent = prev_independence > alpha;
    (max_child, is_independent)
}

/// `_check_independence_withou_K(parents, child, P, N, Y)`.
fn check_independence_without_k(
    parents: &HashSet<usize>,
    child: usize,
    y: &Array2<f64>,
    alpha: f64,
) -> bool {
    let child_col = y.column(child).to_vec();
    for &parent in parents {
        let parent_col = y.column(parent).to_vec();
        if is_independent_1d(alpha, &child_col, &parent_col) {
            return false;
        }
    }
    true
}

/// `_get_residuals_matrix(X, Y, P, child)`, applied in place.
fn update_residual_column(
    x: &Array2<f64>,
    y: &mut Array2<f64>,
    p: &[HashSet<usize>],
    child: usize,
) {
    let mut explanatory: Vec<usize> = p[child].iter().copied().collect();
    explanatory.sort_unstable();
    let residual = get_residual(x, child, &explanatory);
    y.column_mut(child).assign(&residual);
}

/// `_find_parents(X, maxnum_vals, N)`.
fn find_parents(
    x: &Array2<f64>,
    max_vals: usize,
    neighborhoods: &[HashSet<usize>],
    alpha: f64,
    pk: Option<&PriorKnowledge>,
) -> Vec<HashSet<usize>> {
    let d = x.ncols();
    let mut p: Vec<HashSet<usize>> = vec![HashSet::new(); d];
    let mut t = 2usize;
    let mut y = x.clone();

    // The combinatorial search mutates `p`/`y` as it goes, and later
    // combinations in the *same* pass observe earlier mutations (a combination
    // can become newly identifiable, or its residuals can change, mid-pass) —
    // this must stay sequential to match the reference exactly.
    loop {
        let mut changed = false;
        for combo in combinations(d, t) {
            if !check_identified_causality(&combo, &p) {
                continue;
            }
            let (child, is_independence_with_k) =
                get_child(x, &combo, &p, neighborhoods, &y, alpha, pk);
            let Some(child) = child else { continue };
            if !is_independence_with_k {
                continue;
            }
            let parents: HashSet<usize> = combo.iter().copied().filter(|&v| v != child).collect();
            if !check_independence_without_k(&parents, child, &y, alpha) {
                continue;
            }
            for &parent in &parents {
                p[child].insert(parent);
                changed = true;
                update_residual_column(x, &mut y, &p, child);
            }
        }
        if changed {
            t = 2;
        } else {
            t += 1;
            if t > max_vals {
                break;
            }
        }
    }

    // Drop any parent that turns out to be independent of the child once
    // excluded from the child's own regression (a parent can become redundant
    // as later parents are added).
    for i in 0..d {
        let mut non_parents = Vec::new();
        for &j in &p[i] {
            let mut reduced: Vec<usize> = p[i].iter().copied().filter(|&v| v != j).collect();
            reduced.sort_unstable();
            let residual_i = get_residual(x, i, &reduced);

            let mut pj: Vec<usize> = p[j].iter().copied().collect();
            pj.sort_unstable();
            let residual_j = get_residual(x, j, &pj);

            if is_independent_1d(
                alpha,
                residual_i.as_slice().unwrap(),
                residual_j.as_slice().unwrap(),
            ) {
                non_parents.push(j);
            }
        }
        for j in non_parents {
            p[i].remove(&j);
        }
    }

    p
}

/// `_estimate_adjacency_matrix(X, P, U)` combined with the `fit` loop that finds
/// `U`: full `CAMUV.fit(X)`, returning the `(p, p)` adjacency matrix (`1` =
/// discovered parent edge, `NaN` = suspected shared latent confounder, `0`
/// otherwise).
pub fn fit(
    x: &Array2<f64>,
    alpha: f64,
    num_explanatory_vals: usize,
    pk: Option<&PriorKnowledge>,
) -> Array2<f64> {
    let d = x.ncols();
    let neighborhoods = get_neighborhoods(x, alpha);
    let p = find_parents(x, num_explanatory_vals, &neighborhoods, alpha, pk);

    // Confounded-pair candidates: no direct edge either way, and mutually
    // dependent in the raw-column neighborhoods. Each candidate's independence
    // test only reads the now-final `p`, so this loop is embarrassingly
    // parallel.
    let candidates: Vec<(usize, usize)> = (0..d)
        .flat_map(|i| ((i + 1)..d).map(move |j| (i, j)))
        .filter(|&(i, j)| {
            !(p[j].contains(&i) || p[i].contains(&j))
                && neighborhoods[j].contains(&i)
                && neighborhoods[i].contains(&j)
        })
        .collect();

    let confounded: Vec<bool> = crate::pool::install(|| {
        candidates
            .par_iter()
            .map(|&(i, j)| {
                let mut pi: Vec<usize> = p[i].iter().copied().collect();
                pi.sort_unstable();
                let mut pj: Vec<usize> = p[j].iter().copied().collect();
                pj.sort_unstable();
                let ri = get_residual(x, i, &pi);
                let rj = get_residual(x, j, &pj);
                !is_independent_1d(alpha, ri.as_slice().unwrap(), rj.as_slice().unwrap())
            })
            .collect()
    });

    let mut b = Array2::<f64>::zeros((d, d));
    for (i, parents) in p.iter().enumerate() {
        for &parent in parents {
            b[[i, parent]] = 1.0;
        }
    }
    for (&(i, j), &conf) in candidates.iter().zip(confounded.iter()) {
        if conf {
            b[[i, j]] = f64::NAN;
            b[[j, i]] = f64::NAN;
        }
    }
    b
}

/// `_make_pk_dict(prior_knowledge)`: `prior_knowledge` is an iterable of
/// `(from, to)` pairs ("`from` cannot be a cause of `to`"). Mirrors the Python
/// version's lack of validation — any pair is accepted verbatim.
fn parse_prior_knowledge(obj: &Bound<'_, PyAny>) -> PyResult<PriorKnowledge> {
    let mut map: PriorKnowledge = HashMap::new();
    for item in obj.try_iter()? {
        let pair: Vec<i64> = item?.extract()?;
        if pair.len() != 2 {
            return Err(PyValueError::new_err(
                "each prior_knowledge entry must be a (from, to) pair",
            ));
        }
        map.entry(pair[1]).or_default().insert(pair[0]);
    }
    Ok(map)
}

#[pyclass]
pub struct CAMUV {
    #[pyo3(get)]
    alpha: f64,
    #[pyo3(get)]
    num_explanatory_vals: usize,
    #[pyo3(get)]
    independence: String,
    #[pyo3(get)]
    ind_corr: f64,
    prior_knowledge: Option<PriorKnowledge>,
    adjacency_matrix: Option<Array2<f64>>,
}

#[pymethods]
impl CAMUV {
    #[new]
    #[pyo3(signature = (
        alpha=0.01,
        num_explanatory_vals=2,
        independence="hsic".to_string(),
        ind_corr=0.5,
        prior_knowledge=None,
    ))]
    fn new(
        py: Python<'_>,
        alpha: f64,
        num_explanatory_vals: i64,
        independence: String,
        ind_corr: f64,
        prior_knowledge: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if num_explanatory_vals <= 0 {
            return Err(PyValueError::new_err("num_explanatory_vals must be > 0."));
        }
        if alpha < 0.0 {
            return Err(PyValueError::new_err("alpha must be >= 0."));
        }
        if independence != "hsic" && independence != "fcorr" {
            return Err(PyValueError::new_err(
                "independence must be 'hsic' or 'fcorr'.",
            ));
        }
        if ind_corr < 0.0 {
            return Err(PyValueError::new_err(
                "ind_corr must be an float greater than 0.",
            ));
        }
        if independence == "fcorr" {
            return Err(PyNotImplementedError::new_err(
                "ruslingam only implements independence='hsic' (got 'fcorr').",
            ));
        }

        let prior_knowledge = match prior_knowledge {
            Some(obj) => Some(parse_prior_knowledge(obj.bind(py))?),
            None => None,
        };

        Ok(Self {
            alpha,
            num_explanatory_vals: num_explanatory_vals as usize,
            independence,
            ind_corr,
            prior_knowledge,
            adjacency_matrix: None,
        })
    }

    /// Fit the model to `X`; returns `self`.
    fn fit<'py>(slf: Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let data = crate::pyutil::to_array2(py, x)?;
        let (alpha, num_explanatory_vals, pk) = {
            let me = slf.borrow();
            (
                me.alpha,
                me.num_explanatory_vals,
                me.prior_knowledge.clone(),
            )
        };
        let b = fit(&data, alpha, num_explanatory_vals, pk.as_ref());
        {
            let mut me = slf.borrow_mut();
            me.adjacency_matrix = Some(b);
        }
        Ok(slf)
    }

    #[getter]
    fn adjacency_matrix_<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        match &self.adjacency_matrix {
            Some(b) => b.clone().into_pyarray(py).into_any().unbind(),
            None => py.None(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_match_itertools_order() {
        // itertools.combinations(range(4), 2)
        let expected = vec![
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 3],
        ];
        assert_eq!(combinations(4, 2), expected);
    }

    #[test]
    fn combinations_size_equals_d() {
        assert_eq!(combinations(3, 3), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn combinations_t_greater_than_d_is_empty() {
        assert!(combinations(2, 3).is_empty());
    }

    #[test]
    fn check_identified_causality_detects_known_edge() {
        let mut p = vec![HashSet::new(), HashSet::new(), HashSet::new()];
        p[1].insert(0); // 0 is a parent of 1
        assert!(!check_identified_causality(&[0, 1], &p));
        assert!(check_identified_causality(&[0, 2], &p));
    }

    #[test]
    fn prior_knowledge_blocks_forbidden_parent() {
        let mut pk: PriorKnowledge = HashMap::new();
        pk.entry(3).or_default().insert(0); // 0 cannot cause 3
        let xj_list: HashSet<usize> = [0, 2].into_iter().collect();
        assert!(check_prior_knowledge(Some(&pk), &xj_list, 3));
        assert!(!check_prior_knowledge(Some(&pk), &xj_list, 1));
    }

    /// End-to-end smoke test: a clearly-dependent pair with a strong nonlinear
    /// additive relationship should end up with at least a neighborhood/edge
    /// relationship, and two independent columns should not.
    #[test]
    fn fit_smoke_test_separates_dependent_from_independent() {
        let n = 400usize;
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = i as f64;
            let e0 = ((i.wrapping_mul(2_654_435_761) % 1000) as f64) / 500.0 - 1.0;
            let e2 = ((i.wrapping_mul(40_503) % 997) as f64) / 498.0 - 1.0;
            x[[i, 0]] = e0;
            x[[i, 1]] = e0 * e0 * 2.0 + 0.01 * (t * 0.7).sin(); // x1 strongly depends on x0
            x[[i, 2]] = e2; // independent of both
        }
        let b = fit(&x, 0.01, 2, None);
        assert_eq!(b[[2, 0]], 0.0);
        assert_eq!(b[[0, 2]], 0.0);
        assert_eq!(b[[2, 1]], 0.0);
        assert_eq!(b[[1, 2]], 0.0);
        // x1 should show *some* relationship to x0: either a direct edge or a
        // flagged confounded pair, i.e. not simply left at 0 in both cells.
        assert!(b[[1, 0]] != 0.0 || b[[0, 1]].is_nan());
    }
}
