---
outline: deep
---

# `BootstrapResult`

```python
result = DirectLiNGAM(random_state=0).bootstrap(X, n_sampling=100)
```

The object returned by [`DirectLiNGAM.bootstrap`](/direct-lingam#bootstrap-x-n-sampling).
It mirrors `lingam.bootstrap.BootstrapResult` for the ported subset. `p` below is
the number of features and `n` is `n_sampling`.

## Attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `adjacency_matrices_` | `np.ndarray` `(n, p, p)` | Estimated `B` from each resample. |
| `total_effects_` | `np.ndarray` `(n, p, p)` | Total-effect matrix from each resample; `[k, to, from]` is the effect of `from` on `to`. |
| `resampled_indices_` | `list[list[int]]` | The row indices drawn (with replacement) for each resample. |

## `get_probabilities(min_causal_effect=None)`

`(p, p)` array. Entry `[i, j]` is the fraction of resamples in which
`abs(B[i, j]) > min_causal_effect`, i.e. the bootstrap probability of the edge
`x_j -> x_i`. `min_causal_effect` defaults to `0.0` and must be non-negative.

## `get_causal_direction_counts(n_directions=None, min_causal_effect=None, split_by_causal_effect_sign=False)`

Count how often each directed edge appears across resamples, sorted by count
descending. Returns a dict:

```python
{
    "from":  [...],   # source variable index
    "to":    [...],   # destination variable index
    "count": [...],   # number of resamples containing the edge
    "sign":  [...],   # only when split_by_causal_effect_sign=True: +1 / -1
}
```

`n_directions` truncates to the top entries (default: all). With
`split_by_causal_effect_sign=True`, positive and negative edges are counted
separately.

## `get_directed_acyclic_graph_counts(n_dags=None, min_causal_effect=None, split_by_causal_effect_sign=False)`

Count how often each **full DAG** (thresholded adjacency pattern) appears, sorted
by count descending. Returns a dict:

```python
{
    "dag":   [{"from": [...], "to": [...], "sign": [...]}, ...],
    "count": [...],
}
```

`n_dags` truncates to the top entries (default: all). `sign` is present in each
DAG entry only when `split_by_causal_effect_sign=True`.

## `get_total_causal_effects(min_causal_effect=None)`

Bootstrap summary of total effects, over every direction that appears with
non-zero probability, sorted by probability descending. Returns a dict:

```python
{
    "from":        [...],
    "to":          [...],
    "effect":      [...],   # median total effect across resamples where it is non-zero
    "probability": [...],   # fraction of resamples with abs(total effect) > min_causal_effect
}
```

## `get_paths(from_index, to_index, min_causal_effect=None)`

Enumerate every directed path from `from_index` to `to_index` across resamples,
sorted by frequency descending. Returns a dict:

```python
{
    "path":        [[from_index, ..., to_index], ...],
    "effect":      [...],   # median product-of-coefficients effect along the path
    "probability": [...],   # fraction of resamples containing the path
}
```
