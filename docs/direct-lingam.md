---
outline: deep
---

# `DirectLiNGAM`

```python
from ruslingam import DirectLiNGAM
```

A drop-in replacement for `lingam.DirectLiNGAM`. Only the subset below is
implemented; see [Differences from `lingam`](/differences).

## Constructor

```python
DirectLiNGAM(
    random_state=None,
    prior_knowledge=None,
    apply_prior_knowledge_softly=False,
    measure="pwling",
    adaptive_lasso=True,
)
```

| Parameter | Description |
| --- | --- |
| `random_state` | Seeds the internal resampler used by [`bootstrap`](#bootstrap-x-n-sampling). `None` seeds from the system clock. |
| `prior_knowledge` | An `(n_features, n_features)` matrix of `0` / `1` / `-1` entries, or `None`. See [Prior knowledge](#prior-knowledge). |
| `apply_prior_knowledge_softly` | `False` (default): prior knowledge hard-constrains the causal-order search. `True`: it is applied as a soft penalty. |
| `measure` | Only `"pwling"` (entropy-based pairwise likelihood ratio) is supported. `"kernel"` and `"pwling_fast"` raise `NotImplementedError` from the constructor. |
| `adaptive_lasso` | `True`: prune `B` with an Adaptive Lasso (`LassoLarsIC('bic')`, reproduced in Rust). `False`: use ordinary least squares. |

`measure` and `adaptive_lasso` are readable back as attributes on the instance.

## `fit(X)`

Estimate the causal order and adjacency matrix from `X`, an
`(n_samples, n_features)` array-like of finite floats. Returns `self`.

```python
model = DirectLiNGAM(random_state=0)
model.fit(X)
```

`X` is coerced with `numpy.asarray(X, dtype="float64")`; a non-2D array, or one
containing `NaN` / `inf`, raises `ValueError`.

### Fitted attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `causal_order_` | `list[int]` | Estimated topological order of the variables. `None` before `fit`. |
| `adjacency_matrix_` | `np.ndarray` `(p, p)` | Estimated coefficient matrix `B`, where `B[i, j]` is the direct effect of `x_j` on `x_i`. `None` before `fit`. |

## `estimate_total_effect(X, from_index, to_index)`

Total causal effect of `x[from_index]` on `x[to_index]`, computed from `X` and the
fitted order. Returns a `float`.

```python
model.estimate_total_effect(X, 0, 2)
```

If `to_index` comes **before** `from_index` in `causal_order_`, the effect is
still returned but a `UserWarning` is emitted (the value may be meaningless).

## `get_error_independence_p_values(X)`

Return a `(p, p)` array of p-values from a gamma-approximation HSIC test between
each pair of residuals `e_i`, `e_j` (`e = X - X @ B.T`). The matrix is symmetric
with a zero diagonal. Requires `fit` to have been called.

```python
p_values = model.get_error_independence_p_values(X)
```

The `p(p-1)/2` pair tests run on the [Rayon thread pool](/threading).

## `bootstrap(X, n_sampling)`

Resample `X` with replacement `n_sampling` times, refit on each resample, and
return a [`BootstrapResult`](/bootstrap). `n_sampling` must be a positive integer.

```python
result = model.bootstrap(X, n_sampling=100)
result.adjacency_matrices_          # np.ndarray (n_sampling, p, p)
result.total_effects_              # np.ndarray (n_sampling, p, p)
```

The resampling RNG is seeded from the constructor's `random_state`. After
`bootstrap` returns, `causal_order_` and `adjacency_matrix_` hold the fit from the
**last** resample.

## Prior knowledge

`prior_knowledge` is an `(n_features, n_features)` integer matrix:

| `pk[i, j]` | Meaning |
| --- | --- |
| `0` | `x_j` does **not** have a directed path to `x_i`. |
| `1` | `x_j` has a directed path to `x_i`. |
| `-1` | Unknown (no constraint). |

It restricts the causal-order search and prunes `B`, in both the default hard mode
and with `apply_prior_knowledge_softly=True`. The matrix is validated at `fit`
time: a non-square shape, or an inconsistent matrix that asserts both `i -> j` and
`j -> i`, raises `ValueError`.

```python
import numpy as np

# 4 variables; force x0 to be exogenous (nothing points into it).
pk = np.array([
    [ 0,  0,  0,  0],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
])
model = DirectLiNGAM(prior_knowledge=pk)
model.fit(X)
```
