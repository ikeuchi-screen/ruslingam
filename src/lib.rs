//! ruslingam — LiNGAM causal-discovery algorithms implemented in Rust,
//! exposing the same Python API as the `lingam` package.

use pyo3::prelude::*;

mod adjacency;
mod bootstrap;
mod camuv;
mod direct_lingam;
mod gam;
mod hsic;
mod lars;
mod pool;
mod pyclass;
mod pyutil;
mod util;

/// The Python module `ruslingam`.
#[pymodule]
fn ruslingam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pyclass::DirectLiNGAM>()?;
    m.add_class::<bootstrap::BootstrapResult>()?;
    m.add_class::<camuv::CAMUV>()?;
    m.add_function(wrap_pyfunction!(pool::set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(pool::get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(hsic::hsic_test_gamma_py, m)?)?;
    Ok(())
}
