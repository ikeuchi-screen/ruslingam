//! `ruslingam.DirectLiNGAM` — the Python-facing class, a drop-in for
//! `lingam.DirectLiNGAM` (see README for the supported subset).

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;


use crate::bootstrap::{calculate_total_effect, BootstrapResult};
use crate::util::Rng;

fn to_array2<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Array2<f64>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float64")?;
    let arr = np.getattr("asarray")?.call((obj,), Some(&kwargs))?;

    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim != 2 {
        return Err(PyValueError::new_err(format!(
            "Expected a 2D array for X, got a {ndim}D array."
        )));
    }

    let ro: PyReadonlyArray2<f64> = arr.extract()?;
    let owned = ro.as_array().to_owned();

    if owned.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "Input X contains NaN, infinity or a value too large.",
        ));
    }
    if owned.nrows() < 1 || owned.ncols() < 1 {
        return Err(PyValueError::new_err(
            "Found array with 0 sample(s) or 0 feature(s).",
        ));
    }
    Ok(owned)
}

/// `check_array(prior_knowledge)` followed by `np.where(Aknw < 0, np.nan, Aknw)`:
/// require a finite 2D array, then map every negative entry (the `-1` "unknown"
/// sentinel) to `NaN`. The square-shape check happens later, at `fit` time.
fn parse_prior_knowledge<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Array2<f64>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float64")?;
    let arr = np.getattr("asarray")?.call((obj,), Some(&kwargs))?;

    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim != 2 {
        return Err(PyValueError::new_err(format!(
            "Expected a 2D array for prior_knowledge, got a {ndim}D array."
        )));
    }

    let ro: PyReadonlyArray2<f64> = arr.extract()?;
    let mut owned = ro.as_array().to_owned();
    if owned.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "prior_knowledge contains NaN, infinity or a value too large.",
        ));
    }
    owned.mapv_inplace(|v| if v < 0.0 { f64::NAN } else { v });
    Ok(owned)
}

#[pyclass]
pub struct DirectLiNGAM {
    random_state: Option<u64>,
    #[pyo3(get)]
    measure: String,
    #[pyo3(get)]
    adaptive_lasso: bool,
    /// `_Aknw` with negatives already replaced by `NaN`; `None` when no prior
    /// knowledge was supplied.
    prior_knowledge: Option<Array2<f64>>,
    apply_prior_knowledge_softly: bool,
    causal_order: Option<Vec<usize>>,
    adjacency_matrix: Option<Array2<f64>>,
}

#[pymethods]
impl DirectLiNGAM {
    #[new]
    #[pyo3(signature = (
        random_state=None,
        prior_knowledge=None,
        apply_prior_knowledge_softly=false,
        measure="pwling".to_string(),
        adaptive_lasso=true,
    ))]
    fn new(
        py: Python<'_>,
        random_state: Option<i64>,
        prior_knowledge: Option<Py<PyAny>>,
        apply_prior_knowledge_softly: bool,
        measure: String,
        adaptive_lasso: bool,
    ) -> PyResult<Self> {
        if measure != "pwling" {
            return Err(PyNotImplementedError::new_err(format!(
                "ruslingam only implements measure='pwling' (got {measure:?})."
            )));
        }
        let prior_knowledge = match prior_knowledge {
            Some(obj) => Some(parse_prior_knowledge(py, obj.bind(py))?),
            None => None,
        };
        Ok(Self {
            random_state: random_state.map(|v| v as u64),
            measure,
            adaptive_lasso,
            prior_knowledge,
            apply_prior_knowledge_softly,
            causal_order: None,
            adjacency_matrix: None,
        })
    }

    /// Fit the model to `X`; returns `self`.
    fn fit<'py>(slf: Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let data = to_array2(py, x)?;
        let (adaptive, pk) = {
            let me = slf.borrow();
            (me.adaptive_lasso, me.build_prior_knowledge(data.ncols())?)
        };
        let (order, b) = crate::direct_lingam::fit(&data, adaptive, pk.as_ref());
        {
            let mut me = slf.borrow_mut();
            me.causal_order = Some(order);
            me.adjacency_matrix = Some(b);
        }
        Ok(slf)
    }

    #[getter]
    fn causal_order_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(match &self.causal_order {
            Some(k) => {
                let v: Vec<i64> = k.iter().map(|&x| x as i64).collect();
                PyList::new(py, v)?.into_any().unbind()
            }
            None => py.None(),
        })
    }

    #[getter]
    fn adjacency_matrix_<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        match &self.adjacency_matrix {
            Some(b) => b.clone().into_pyarray(py).into_any().unbind(),
            None => py.None(),
        }
    }

    /// `estimate_total_effect(X, from_index, to_index)`.
    #[pyo3(signature = (x, from_index, to_index))]
    fn estimate_total_effect<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        from_index: usize,
        to_index: usize,
    ) -> PyResult<f64> {
        let order = self
            .causal_order
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("fit() must be called before estimate_total_effect()."))?;
        let b = self.adjacency_matrix.as_ref().unwrap();
        let data = to_array2(py, x)?;
        let (effect, order_ok) =
            crate::adjacency::estimate_total_effect(&data, b, order, from_index, to_index);
        if !order_ok {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "The estimated causal effect may be incorrect because the causal \
                     order of the destination variable (to_index={to_index}) is earlier \
                     than the source variable (from_index={from_index})."
                ),),
            )?;
        }
        Ok(effect)
    }

    /// `get_error_independence_p_values(X)`.
    fn get_error_independence_p_values<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let b = self
            .adjacency_matrix
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("fit() must be called first."))?;
        let data = to_array2(py, x)?;
        let e = crate::direct_lingam::error_terms(&data, b);
        let p = data.ncols();
        let cols: Vec<Vec<f64>> = (0..p).map(|j| e.column(j).to_vec()).collect();

        // The p(p-1)/2 pair tests are independent; run them on the thread pool.
        let pairs: Vec<(usize, usize)> = (0..p)
            .flat_map(|i| ((i + 1)..p).map(move |j| (i, j)))
            .collect();
        let results: Vec<(usize, usize, f64)> = crate::pool::install(|| {
            pairs
                .par_iter()
                .map(|&(i, j)| {
                    let (_, pval) = crate::hsic::hsic_test_gamma(&cols[i], &cols[j]);
                    (i, j, pval)
                })
                .collect()
        });

        let mut pv = Array2::<f64>::zeros((p, p));
        for (i, j, pval) in results {
            pv[[i, j]] = pval;
            pv[[j, i]] = pval;
        }
        Ok(pv.into_pyarray(py))
    }

    /// `bootstrap(X, n_sampling)` → `BootstrapResult`.
    #[pyo3(signature = (x, n_sampling))]
    fn bootstrap<'py>(
        slf: Bound<'py, Self>,
        x: &Bound<'py, PyAny>,
        n_sampling: i64,
    ) -> PyResult<BootstrapResult> {
        let py = slf.py();
        if n_sampling <= 0 {
            return Err(PyValueError::new_err(
                "n_sampling must be an integer greater than 0.",
            ));
        }
        let n_sampling = n_sampling as usize;

        let data = to_array2(py, x)?;
        let (adaptive, seed, pk) = {
            let me = slf.borrow();
            (
                me.adaptive_lasso,
                me.random_state,
                me.build_prior_knowledge(data.ncols())?,
            )
        };
        let n_samples = data.nrows();
        let p = data.ncols();

        let seed = seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x1234_5678)
        });
        let mut rng = Rng::seed_from_u64(seed);

        let mut ams = Vec::with_capacity(n_sampling);
        let mut tes = Vec::with_capacity(n_sampling);
        let mut resampled = Vec::with_capacity(n_sampling);
        let mut last_order: Vec<usize> = Vec::new();
        let mut last_b: Array2<f64> = Array2::zeros((p, p));

        for _ in 0..n_sampling {
            let idx: Vec<usize> = (0..n_samples).map(|_| rng.gen_index(n_samples)).collect();
            let mut rx = Array2::<f64>::zeros((n_samples, p));
            for (r, &s) in idx.iter().enumerate() {
                rx.row_mut(r).assign(&data.row(s));
            }

            let (order, b) = crate::direct_lingam::fit(&rx, adaptive, pk.as_ref());

            let mut te = Array2::<f64>::zeros((p, p));
            for (c, &from_) in order.iter().enumerate() {
                for &to in &order[c + 1..] {
                    te[[to, from_]] = calculate_total_effect(&b, from_, to);
                }
            }

            ams.push(b.clone());
            tes.push(te);
            resampled.push(idx);
            last_order = order;
            last_b = b;
        }

        {
            let mut me = slf.borrow_mut();
            me.causal_order = Some(last_order);
            me.adjacency_matrix = Some(last_b);
        }

        Ok(BootstrapResult::new(ams, tes, resampled, p))
    }
}

impl DirectLiNGAM {
    /// Build the validated `PriorKnowledge` for a `fit` on `n_features` columns.
    /// The `fit`-time checks (`_extract_partial_orders` inconsistencies, wrong
    /// shape) surface as `ValueError`, matching `lingam`.
    fn build_prior_knowledge(
        &self,
        n_features: usize,
    ) -> PyResult<Option<crate::direct_lingam::PriorKnowledge>> {
        match &self.prior_knowledge {
            Some(a) => Ok(Some(
                crate::direct_lingam::PriorKnowledge::new(
                    a.clone(),
                    self.apply_prior_knowledge_softly,
                    n_features,
                )
                .map_err(PyValueError::new_err)?,
            )),
            None => Ok(None),
        }
    }
}
