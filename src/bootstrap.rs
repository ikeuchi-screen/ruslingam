//! Port of `lingam/bootstrap.py` (`BootstrapMixin.bootstrap` + `BootstrapResult`)
//! and the `find_all_paths` / `calculate_total_effect` helpers from `lingam/utils`.

use ndarray::{Array2, Array3, Axis};
use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::util::{argsort_desc, argsort_desc_f64, median, unique_rows};

/// `find_all_paths(dag, from_index, to_index, min_causal_effect)`.
/// Edge `j -> i` exists when `|dag[i, j]| > min_causal_effect`.
pub fn find_all_paths(
    dag: &Array2<f64>,
    from_index: usize,
    to_index: usize,
    min_causal_effect: f64,
) -> (Vec<Vec<usize>>, Vec<f64>) {
    let p = dag.nrows();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); p];
    for i in 0..p {
        for k in 0..p {
            if dag[[k, i]].abs() > min_causal_effect {
                succ[i].push(k);
            }
        }
    }

    let mut paths: Vec<Vec<usize>> = Vec::new();
    let mut stack = vec![from_index];
    let mut on_path = vec![false; p];
    on_path[from_index] = true;
    dfs(&succ, from_index, to_index, &mut stack, &mut on_path, &mut paths);

    let effects = paths
        .iter()
        .map(|path| {
            let mut prod = 1.0;
            for w in path.windows(2) {
                prod *= dag[[w[1], w[0]]];
            }
            prod
        })
        .collect();

    (paths, effects)
}

fn dfs(
    succ: &[Vec<usize>],
    cur: usize,
    to: usize,
    stack: &mut Vec<usize>,
    on_path: &mut [bool],
    paths: &mut Vec<Vec<usize>>,
) {
    if cur == to {
        paths.push(stack.clone());
        return;
    }
    for &nx in &succ[cur] {
        if on_path[nx] {
            continue; // guard against cycles (DirectLiNGAM output is a DAG)
        }
        stack.push(nx);
        on_path[nx] = true;
        dfs(succ, nx, to, stack, on_path, paths);
        on_path[nx] = false;
        stack.pop();
    }
}

/// `calculate_total_effect(adjacency_matrix, from_index, to_index)` — sum of the
/// products of coefficients along every directed path from `from` to `to`.
pub fn calculate_total_effect(dag: &Array2<f64>, from_index: usize, to_index: usize) -> f64 {
    let (_, effects) = find_all_paths(dag, from_index, to_index, 0.0);
    effects.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn paths_and_total_effect_on_3x3() {
        // edges: 0->1 (2.0), 1->2 (-3.0), 0->2 (0.5)  [B[i,j] is edge j->i]
        let b = array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, -3.0, 0.0]];
        let (paths, effects) = find_all_paths(&b, 0, 2, 0.0);
        assert_eq!(paths, vec![vec![0, 1, 2], vec![0, 2]]);
        assert!((effects[0] - (2.0 * -3.0)).abs() < 1e-12);
        assert!((effects[1] - 0.5).abs() < 1e-12);
        // total effect = -6.0 + 0.5 = -5.5
        assert!((calculate_total_effect(&b, 0, 2) - (-5.5)).abs() < 1e-12);
    }

    #[test]
    fn min_effect_filters_edges() {
        let b = array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.05, -3.0, 0.0]];
        let (paths, _) = find_all_paths(&b, 0, 2, 0.1);
        assert_eq!(paths, vec![vec![0, 1, 2]]);
    }
}

fn check_min_causal_effect(v: Option<f64>) -> PyResult<f64> {
    match v {
        None => Ok(0.0),
        Some(x) if x > 0.0 => Ok(x),
        Some(_) => Err(PyValueError::new_err(
            "min_causal_effect must be an value greater than 0.",
        )),
    }
}

/// `BootstrapResult`.
#[pyclass]
pub struct BootstrapResult {
    adjacency_matrices: Vec<Array2<f64>>,
    total_effects: Vec<Array2<f64>>,
    resampled_indices: Vec<Vec<usize>>,
    n_features: usize,
}

impl BootstrapResult {
    pub fn new(
        adjacency_matrices: Vec<Array2<f64>>,
        total_effects: Vec<Array2<f64>>,
        resampled_indices: Vec<Vec<usize>>,
        n_features: usize,
    ) -> Self {
        Self {
            adjacency_matrices,
            total_effects,
            resampled_indices,
            n_features,
        }
    }

    fn stack3(mats: &[Array2<f64>], p: usize) -> Array3<f64> {
        let mut out = Array3::<f64>::zeros((mats.len(), p, p));
        for (i, m) in mats.iter().enumerate() {
            out.index_axis_mut(Axis(0), i).assign(m);
        }
        out
    }
}

#[pymethods]
impl BootstrapResult {
    #[getter]
    fn adjacency_matrices_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        Self::stack3(&self.adjacency_matrices, self.n_features).into_pyarray(py)
    }

    #[getter]
    fn total_effects_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        Self::stack3(&self.total_effects, self.n_features).into_pyarray(py)
    }

    #[getter]
    fn resampled_indices_<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let rows: Vec<Vec<usize>> = self.resampled_indices.clone();
        PyList::new(py, rows.iter().map(|r| PyList::new(py, r).unwrap())).unwrap()
    }

    #[pyo3(signature = (n_directions=None, min_causal_effect=None, split_by_causal_effect_sign=false))]
    fn get_causal_direction_counts<'py>(
        &self,
        py: Python<'py>,
        n_directions: Option<usize>,
        min_causal_effect: Option<f64>,
        split_by_causal_effect_sign: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let min_effect = check_min_causal_effect(min_causal_effect)?;

        let mut rows: Vec<Vec<i64>> = Vec::new();
        for am in &self.adjacency_matrices {
            for i in 0..self.n_features {
                for j in 0..self.n_features {
                    if am[[i, j]].abs() > min_effect {
                        let mut row = vec![i as i64, j as i64];
                        if split_by_causal_effect_sign {
                            row.push(am[[i, j]].signum() as i64);
                        }
                        rows.push(row);
                    }
                }
            }
        }

        let dict = PyDict::new(py);
        if rows.is_empty() {
            dict.set_item("from", PyList::empty(py))?;
            dict.set_item("to", PyList::empty(py))?;
            dict.set_item("count", PyList::empty(py))?;
            if split_by_causal_effect_sign {
                dict.set_item("sign", PyList::empty(py))?;
            }
            return Ok(dict);
        }

        let (uniq, counts) = unique_rows(rows);
        let mut order = argsort_desc(&counts);
        if let Some(n) = n_directions {
            order.truncate(n);
        }

        let from: Vec<i64> = order.iter().map(|&k| uniq[k][1]).collect();
        let to: Vec<i64> = order.iter().map(|&k| uniq[k][0]).collect();
        let count: Vec<i64> = order.iter().map(|&k| counts[k]).collect();
        dict.set_item("from", from)?;
        dict.set_item("to", to)?;
        dict.set_item("count", count)?;
        if split_by_causal_effect_sign {
            let sign: Vec<i64> = order.iter().map(|&k| uniq[k][2]).collect();
            dict.set_item("sign", sign)?;
        }
        Ok(dict)
    }

    #[pyo3(signature = (n_dags=None, min_causal_effect=None, split_by_causal_effect_sign=false))]
    fn get_directed_acyclic_graph_counts<'py>(
        &self,
        py: Python<'py>,
        n_dags: Option<usize>,
        min_causal_effect: Option<f64>,
        split_by_causal_effect_sign: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let min_effect = check_min_causal_effect(min_causal_effect)?;
        let p = self.n_features;

        let mut flats: Vec<Vec<i64>> = Vec::with_capacity(self.adjacency_matrices.len());
        for am in &self.adjacency_matrices {
            let mut flat = Vec::with_capacity(p * p);
            for i in 0..p {
                for j in 0..p {
                    let hit = am[[i, j]].abs() > min_effect;
                    if split_by_causal_effect_sign {
                        flat.push(if hit { am[[i, j]].signum() as i64 } else { 0 });
                    } else {
                        flat.push(hit as i64);
                    }
                }
            }
            flats.push(flat);
        }

        let (uniq, counts) = unique_rows(flats);
        let mut order = argsort_desc(&counts);
        if let Some(n) = n_dags {
            order.truncate(n);
        }

        let dag_list = PyList::empty(py);
        for &k in &order {
            let flat = &uniq[k];
            let mut from = Vec::new();
            let mut to = Vec::new();
            let mut sign = Vec::new();
            for i in 0..p {
                for j in 0..p {
                    let v = flat[i * p + j];
                    if v != 0 {
                        from.push(j as i64);
                        to.push(i as i64);
                        sign.push(v);
                    }
                }
            }
            let d = PyDict::new(py);
            d.set_item("from", from)?;
            d.set_item("to", to)?;
            if split_by_causal_effect_sign {
                d.set_item("sign", sign)?;
            }
            dag_list.append(d)?;
        }

        let count: Vec<i64> = order.iter().map(|&k| counts[k]).collect();
        let dict = PyDict::new(py);
        dict.set_item("dag", dag_list)?;
        dict.set_item("count", count)?;
        Ok(dict)
    }

    #[pyo3(signature = (min_causal_effect=None))]
    fn get_probabilities<'py>(
        &self,
        py: Python<'py>,
        min_causal_effect: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let min_effect = check_min_causal_effect(min_causal_effect)?;
        let p = self.n_features;
        let mut bp = Array2::<f64>::zeros((p, p));
        for am in &self.adjacency_matrices {
            for i in 0..p {
                for j in 0..p {
                    if am[[i, j]].abs() > min_effect {
                        bp[[i, j]] += 1.0;
                    }
                }
            }
        }
        let n = self.adjacency_matrices.len().max(1) as f64;
        bp.mapv_inplace(|v| v / n);
        Ok(bp.into_pyarray(py))
    }

    #[pyo3(signature = (min_causal_effect=None))]
    fn get_total_causal_effects<'py>(
        &self,
        py: Python<'py>,
        min_causal_effect: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let min_effect = check_min_causal_effect(min_causal_effect)?;
        let p = self.n_features;
        let n = self.total_effects.len().max(1) as f64;

        let mut probs = Array2::<f64>::zeros((p, p));
        for te in &self.total_effects {
            for i in 0..p {
                for j in 0..p {
                    if te[[i, j]].abs() > min_effect {
                        probs[[i, j]] += 1.0;
                    }
                }
            }
        }
        probs.mapv_inplace(|v| v / n);

        // directions where probability > 0, row-major (to = row, from = col)
        let mut dirs: Vec<(usize, usize)> = Vec::new();
        for i in 0..p {
            for j in 0..p {
                if probs[[i, j]].abs() > 0.0 {
                    dirs.push((i, j));
                }
            }
        }

        let prob_sel: Vec<f64> = dirs.iter().map(|&(i, j)| probs[[i, j]]).collect();
        let effects: Vec<f64> = dirs
            .iter()
            .map(|&(to, from_)| {
                let vals: Vec<f64> = self
                    .total_effects
                    .iter()
                    .map(|te| te[[to, from_]])
                    .filter(|v| v.abs() > 0.0)
                    .collect();
                if vals.is_empty() {
                    f64::NAN
                } else {
                    median(&vals)
                }
            })
            .collect();

        let order = argsort_desc_f64(&prob_sel);
        let from: Vec<i64> = order.iter().map(|&k| dirs[k].1 as i64).collect();
        let to: Vec<i64> = order.iter().map(|&k| dirs[k].0 as i64).collect();
        let effect: Vec<f64> = order.iter().map(|&k| effects[k]).collect();
        let probability: Vec<f64> = order.iter().map(|&k| prob_sel[k]).collect();

        let dict = PyDict::new(py);
        dict.set_item("from", from)?;
        dict.set_item("to", to)?;
        dict.set_item("effect", effect)?;
        dict.set_item("probability", probability)?;
        Ok(dict)
    }

    #[pyo3(signature = (from_index, to_index, min_causal_effect=None))]
    fn get_paths<'py>(
        &self,
        py: Python<'py>,
        from_index: usize,
        to_index: usize,
        min_causal_effect: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let min_effect = check_min_causal_effect(min_causal_effect)?;

        let mut path_strings: Vec<String> = Vec::new();
        let mut effect_values: Vec<f64> = Vec::new();
        for am in &self.adjacency_matrices {
            let (paths, effects) = find_all_paths(am, from_index, to_index, min_effect);
            for (path, eff) in paths.iter().zip(effects.iter()) {
                let s = path
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join("_");
                path_strings.push(s);
                effect_values.push(*eff);
            }
        }

        // np.unique on the string reprs -> lexicographically sorted unique strings
        let mut uniq: Vec<String> = path_strings.clone();
        uniq.sort();
        uniq.dedup();

        let counts: Vec<i64> = uniq
            .iter()
            .map(|u| path_strings.iter().filter(|s| *s == u).count() as i64)
            .collect();
        let order = argsort_desc(&counts);
        let n = self.adjacency_matrices.len().max(1) as f64;

        let path_out = PyList::empty(py);
        let mut effect_out: Vec<f64> = Vec::new();
        let mut prob_out: Vec<f64> = Vec::new();
        for &k in &order {
            let u = &uniq[k];
            let nodes: Vec<i64> = u.split('_').map(|s| s.parse::<i64>().unwrap()).collect();
            path_out.append(nodes)?;
            let vals: Vec<f64> = path_strings
                .iter()
                .zip(effect_values.iter())
                .filter(|(s, _)| *s == u)
                .map(|(_, e)| *e)
                .collect();
            effect_out.push(median(&vals));
            prob_out.push(counts[k] as f64 / n);
        }

        let dict = PyDict::new(py);
        dict.set_item("path", path_out)?;
        dict.set_item("effect", effect_out)?;
        dict.set_item("probability", prob_out)?;
        Ok(dict)
    }
}
