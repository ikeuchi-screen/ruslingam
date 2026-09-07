# ruslingam

Accelerating LiNGAM with Rust.

`ruslingam` reimplements algorithms from the Python
[`lingam`](https://github.com/ikeuchi-screen/lingam) package in Rust (via
[PyO3](https://pyo3.rs) / [maturin](https://www.maturin.rs)) and exposes the same
Python API, so it can be used as a drop-in replacement.

## Installation

```bash
pip install maturin
maturin develop --release        # build & install into the current venv
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
* `prior_knowledge` / `apply_prior_knowledge_softly` are accepted for signature
  compatibility only and raise `NotImplementedError` when set.

## Differences from `lingam`

1. **Adaptive Lasso** matches scikit-learn within numerical tolerance (~1e-6 on the
   test suite); the selected edge set agrees on well-conditioned data. For exact
   OLS parity use `adaptive_lasso=False`.
2. **`bootstrap`** uses an internal RNG rather than scikit-learn's `resample`, so
   the resampled indices differ from `lingam` for a given `random_state`. Aggregate
   statistics are equivalent in distribution.
3. `measure="kernel"`, `measure="pwling_fast"` and `prior_knowledge` are not
   implemented yet.
4. `BootstrapResult` count rankings use a stable sort; entries with equal counts
   may be ordered differently from NumPy's `argsort`.

## Development

```bash
cargo test              # Rust unit tests
maturin develop         # (re)build the extension
pytest                  # Python tests; tests/test_parity.py compares against `lingam`
                        # and is skipped automatically if `lingam` is not installed
```
