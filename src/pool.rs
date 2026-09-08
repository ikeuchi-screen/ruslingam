//! Optional user-controlled rayon thread pool for the parallel sections.
//!
//! `ruslingam.set_num_threads(n)` installs a dedicated pool of `n` worker threads
//! that the parallel causal-order search runs inside. Never calling it (or passing
//! `0`) leaves the work on rayon's default global pool, which sizes itself to the
//! machine and honours the `RAYON_NUM_THREADS` environment variable.

use std::sync::{Arc, OnceLock, RwLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::ThreadPool;

fn slot() -> &'static RwLock<Option<Arc<ThreadPool>>> {
    static SLOT: OnceLock<RwLock<Option<Arc<ThreadPool>>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(None))
}

/// Run `f` on the user-configured pool if one has been set with
/// [`set_num_threads`], otherwise directly (which keeps `rayon` iterators on the
/// default global pool).
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let pool = slot().read().unwrap().clone();
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// `ruslingam.set_num_threads(n)`.
///
/// * `n >= 1` — route the parallel causal-order search through a dedicated pool of
///   `n` worker threads, replacing any pool from an earlier call.
/// * `n == 0` — drop the dedicated pool and fall back to rayon's default global
///   pool.
///
/// Safe to call at any point: it does not touch rayon's build-once global pool,
/// so it still takes effect after `fit()` has already run.
#[pyfunction]
pub fn set_num_threads(n: isize) -> PyResult<()> {
    if n < 0 {
        return Err(PyValueError::new_err("n must be >= 0"));
    }
    let new_pool = if n == 0 {
        None
    } else {
        let p = rayon::ThreadPoolBuilder::new()
            .num_threads(n as usize)
            .thread_name(|i| format!("ruslingam-{i}"))
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to build thread pool: {e}")))?;
        Some(Arc::new(p))
    };
    *slot().write().unwrap() = new_pool;
    Ok(())
}

/// `ruslingam.get_num_threads()` — worker count of the pool the parallel search
/// will use: the dedicated one if [`set_num_threads`] set it, else the default
/// global pool.
#[pyfunction]
pub fn get_num_threads() -> usize {
    match slot().read().unwrap().as_ref() {
        Some(p) => p.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}
