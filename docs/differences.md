---
outline: deep
---

# Differences from `lingam`

`ruslingam` aims to be a drop-in replacement for `lingam.DirectLiNGAM`, but a few
behaviours differ.

1. **Adaptive Lasso** matches scikit-learn within numerical tolerance (~1e-6 on
   the test suite); the selected edge set agrees on well-conditioned data. For
   exact OLS parity use `adaptive_lasso=False`.

2. **`bootstrap`** uses an internal RNG rather than scikit-learn's `resample`, so
   the resampled indices differ from `lingam` for a given `random_state`.
   Aggregate statistics are equivalent in distribution.

3. **`measure="kernel"` and `measure="pwling_fast"` are not implemented.** Passing
   either raises `NotImplementedError` from the `DirectLiNGAM` constructor. Only
   `measure="pwling"` is supported.

4. **Count rankings use a stable sort.** In `BootstrapResult`, entries with equal
   counts may be ordered differently from NumPy's `argsort`.

5. **Scope.** Only `DirectLiNGAM` (plus its `BootstrapResult`) and
   `hsic_test_gamma` are ported. Other `lingam` estimators — `ICALiNGAM`,
   `VARLiNGAM`, `RCD`, `LiM`, and so on — are not available.

## Input validation

`X` and `prior_knowledge` are coerced with `numpy.asarray(..., dtype="float64")`
and must be 2-D and finite; violations raise `ValueError` with a message close to
scikit-learn's. A non-square or self-contradictory `prior_knowledge` matrix raises
`ValueError` at `fit` time.
