---
outline: deep
---

# `CAMUV`

```python
from ruslingam import CAMUV
```

A drop-in replacement for `lingam.CAMUV` (`independence="hsic"` only; see
[Differences from `lingam`](/differences)). CAMUV ("Causal Additive Models with
Unobserved Variables") finds, for each variable, a set of direct causes, and
also flags pairs of variables suspected to share an unobserved (latent)
confounder — unlike `DirectLiNGAM`, it does not require every pair of variables
to be fully causally ordered.

## Constructor

```python
CAMUV(
    alpha=0.01,
    num_explanatory_vals=2,
    independence="hsic",
    ind_corr=0.5,
    prior_knowledge=None,
)
```

| Parameter | Description |
| --- | --- |
| `alpha` | Significance level for the HSIC independence test. Must be `>= 0`. |
| `num_explanatory_vals` | Maximum number of explanatory variables considered together during the combinatorial parent search. Must be `> 0`. |
| `independence` | Only `"hsic"` (gamma-approximation HSIC test) is supported. `"fcorr"` raises `NotImplementedError` from the constructor. |
| `ind_corr` | Threshold used to determine independence when `independence="fcorr"` (kept for API compatibility; unused since `"fcorr"` is not implemented). Must be `>= 0`. |
| `prior_knowledge` | An iterable of `(from, to)` index pairs, or `None`. See [Prior knowledge](#prior-knowledge). Note this is a *different shape* from `DirectLiNGAM`'s `(n_features, n_features)` matrix. |

`alpha`, `num_explanatory_vals`, `independence`, and `ind_corr` are readable
back as attributes on the instance.

## `fit(X)`

Find each variable's parents and any suspected latent-confounder pairs from
`X`, an `(n_samples, n_features)` array-like of finite floats. Returns `self`.

```python
model = CAMUV()
model.fit(X)
```

`X` is coerced with `numpy.asarray(X, dtype="float64")`; a non-2D array, or one
containing `NaN` / `inf`, raises `ValueError`.

### Fitted attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `adjacency_matrix_` | `np.ndarray` `(p, p)` | `B[i, j] == 1`: `x_j` is a discovered direct cause of `x_i`. `B[i, j] == B[j, i] == NaN`: `x_i` and `x_j` are suspected to share an unobserved confounder (no direct edge between them). `0` otherwise. `None` before `fit`. |

Unlike `DirectLiNGAM`, `CAMUV` has no `causal_order_`, `bootstrap`, or
`estimate_total_effect` — the reference `lingam.CAMUV` doesn't implement them
either, since a `NaN` pair means the direction (and the very existence of a
direct causal path) is genuinely unresolved.

## Prior knowledge

`prior_knowledge` is an iterable of `(from, to)` index pairs, each meaning
"`x_from` cannot be a direct cause of `x_to`":

```python
# 3 variables; x0 can never directly cause x2.
model = CAMUV(prior_knowledge=[(0, 2)])
model.fit(X)
```

Unlike `DirectLiNGAM`'s prior knowledge, this is not validated for consistency
or shape — any iterable of pairs is accepted, matching `lingam.CAMUV`. If a
forbidden pair is also mutually dependent in the data, `CAMUV` falls back to
flagging it as a suspected confounded pair (`NaN`) instead of a direct edge.
