---
outline: deep
---

# Module functions

```python
import ruslingam
```

## `set_num_threads(n)`

Set the worker count for the parallel causal-order search and the
error-independence HSIC tests.

- `n > 0` — run on a dedicated pool of `n` threads.
- `n == 0` — use the default global pool (all logical cores, or
  `RAYON_NUM_THREADS`).
- `n < 0` — raises `ValueError`.

Safe to call at any time, including after `fit()` has already run. See
[Threading](/threading).

## `get_num_threads()`

Return the worker count currently in effect: the dedicated pool's size if
`set_num_threads` set one, otherwise the default global pool's size.

## `hsic_test_gamma(x, y)`

Gamma-approximation HSIC independence test, matching
`lingam.hsic.hsic_test_gamma(x, y, bw_method="mdbs")`.

`x` and `y` are 1-D arrays of equal length (unequal lengths raise `ValueError`).
Returns a `(test_stat, p_value)` tuple of floats.

```python
import numpy as np
from ruslingam import hsic_test_gamma

rng = np.random.default_rng(0)
x = rng.standard_normal(500)
y = rng.standard_normal(500)

stat, p = hsic_test_gamma(x, y)
```
