//! ruslingam — LiNGAM causal-discovery algorithms implemented in Rust,
//! exposing the same Python API as the `lingam` package.

use pyo3::prelude::*;

mod adjacency;
mod bootstrap;
mod direct_lingam;
mod hsic;
mod lars;
mod pool;
mod pyclass;
mod util;

/// The Python module `ruslingam`.
#[pymodule]
fn ruslingam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pyclass::DirectLiNGAM>()?;
    m.add_class::<bootstrap::BootstrapResult>()?;
    m.add_function(wrap_pyfunction!(pool::set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(pool::get_num_threads, m)?)?;
    Ok(())
}
