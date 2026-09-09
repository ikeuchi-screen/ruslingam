# ruslingam

Accelerating LiNGAM with Rust.

`ruslingam` reimplements algorithms from the Python
[`lingam`](https://github.com/ikeuchi-screen/lingam) package in Rust (via
[PyO3](https://pyo3.rs) / [maturin](https://www.maturin.rs)) and exposes the same
Python API, so it can be used as a drop-in replacement.

## Installation

`ruslingam` is not on PyPI yet, so it is built from source. There is no
pre-built wheel: you need a **Rust toolchain** in addition to Python.

### Prerequisites

* **Python** >= 3.8 with `pip`
* **Rust** >= 1.85 (the crate uses edition 2024). Install it with
  [rustup](https://rustup.rs):

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

  On Windows, install "Rust" and the "Desktop development with C++" workload
  from the Visual Studio Build Tools instead.

### Option A — install straight from GitHub (recommended)

This builds the extension and installs it into your current environment. Run it
inside a virtualenv (or a conda env) so it does not touch the system Python:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install "git+https://github.com/ikeuchi-screen/ruslingam.git"
```

`pip` picks up the `maturin` build backend automatically; you do not need to
install `maturin` yourself. To pin a specific commit or branch, append
`@<ref>`, e.g. `...ruslingam.git@main`.

### Option B — clone and build for development

Use this if you want to hack on the Rust code. `maturin develop` compiles the
crate and installs it into the **currently active** virtualenv, so activate one
first.

```bash
git clone https://github.com/ikeuchi-screen/ruslingam.git
cd ruslingam

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install maturin
maturin develop --release            # drop --release for a faster debug build
```

Re-run `maturin develop` after editing the Rust sources to rebuild.

### Verify

```bash
python -c "from ruslingam import DirectLiNGAM; print('ok')"
```

## `DirectLiNGAM`

```python
import numpy as np
from ruslingam import DirectLiNGAM        # instead of: from lingam import DirectLiNGAM

X = np.loadtxt("data.csv", delimiter=",")

model = DirectLiNGAM()
model.fit(X)

model.causal_order_                       # list[int]         - estimated topological order
model.adjacency_matrix_                   # np.ndarray (p, p) - estimated coefficient matrix B

model.estimate_total_effect(X, 0, 2)      # float
model.get_error_independence_p_values(X)  # np.ndarray (p, p)

result = model.bootstrap(X, n_sampling=100)
result.adjacency_matrices_               # np.ndarray (n_sampling, p, p)
result.total_effects_                    # np.ndarray (n_sampling, p, p)
result.get_probabilities()
result.get_causal_direction_counts(n_directions=8, split_by_causal_effect_sign=True)
result.get_directed_acyclic_graph_counts(n_dags=5)
result.get_total_causal_effects(min_causal_effect=0.01)
result.get_paths(0, 2)
```

### Constructor

```python
DirectLiNGAM(
    random_state=None,
    prior_knowledge=None,
    apply_prior_knowledge_softly=False,
    measure="pwling",
    adaptive_lasso=True,
)
```

* `measure="pwling"` (the default, entropy-based pairwise likelihood ratio) is the
  only supported measure. `"kernel"` and `"pwling_fast"` raise `NotImplementedError`.
* `adaptive_lasso=True` prunes `B` with an Adaptive Lasso (`LassoLarsIC('bic')`
  reproduced in Rust); `adaptive_lasso=False` uses ordinary least squares.
* `random_state` seeds the internal resampler used by `bootstrap`.
* `prior_knowledge` is an `(n_features, n_features)` matrix of `0` / `1` / `-1`
  entries (`pk[i, j] == 1`: `x_j` has a directed path to `x_i`; `0`: it does not;
  `-1`: unknown). It restricts the causal-order search and prunes `B`, both in the
  default hard mode and with `apply_prior_knowledge_softly=True`. An inconsistent
  matrix (asserting both `i -> j` and `j -> i`) raises `ValueError`.

## Threading

The causal-order search runs in parallel. By default it uses one worker per
logical core; set `RAYON_NUM_THREADS` in the environment before the first `fit()`
to change that, or control it from Python at any time:

```python
import ruslingam

ruslingam.set_num_threads(4)   # run the search on a dedicated pool of 4 threads
ruslingam.set_num_threads(0)   # back to the default (all cores / RAYON_NUM_THREADS)
ruslingam.get_num_threads()    # -> worker count currently in effect
```

Results (`causal_order_`, `adjacency_matrix_`) are identical regardless of the
thread count.

## Differences from `lingam`

1. **Adaptive Lasso** matches scikit-learn within numerical tolerance (~1e-6 on the
   test suite); the selected edge set agrees on well-conditioned data. For exact
   OLS parity use `adaptive_lasso=False`.
2. **`bootstrap`** uses an internal RNG rather than scikit-learn's `resample`, so
   the resampled indices differ from `lingam` for a given `random_state`. Aggregate
   statistics are equivalent in distribution.
3. `measure="kernel"` and `measure="pwling_fast"` are not implemented yet.
4. `BootstrapResult` count rankings use a stable sort; entries with equal counts
   may be ordered differently from NumPy's `argsort`.

## Development

```bash
cargo test              # Rust unit tests
maturin develop         # (re)build the extension
pytest                  # Python tests; tests/test_parity.py compares against `lingam`
                        # and is skipped automatically if `lingam` is not installed
```
