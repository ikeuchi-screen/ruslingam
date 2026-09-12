//! Shared PyO3/numpy conversion helpers used by both `DirectLiNGAM` and `CAMUV`.

use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `check_array(X)`: `numpy.asarray(X, dtype="float64")`, then require a finite
/// 2-D array with at least one sample and one feature. Both `DirectLiNGAM.fit`
/// and `CAMUV.fit` (via `sklearn.utils.check_array` in the Python package) apply
/// exactly this validation to their input.
pub fn to_array2<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Array2<f64>> {
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
