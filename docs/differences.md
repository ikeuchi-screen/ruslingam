---
outline: deep
---

# Differences from `lingam`

`ruslingam` aims to be a drop-in replacement for `lingam.DirectLiNGAM` and
`lingam.CAMUV`, but a few behaviours differ.

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

5. **`independence="fcorr"` is not implemented for `CAMUV`.** Passing it raises
   `NotImplementedError` from the constructor. Only `independence="hsic"` (the
   default) is supported.

6. **`CAMUV`'s internal GAM regression uses a normal-equations solve**, not
   `pygam`'s QR-then-SVD path. The two agree closely (coefficients within
   ~1e-7, predictions within ~1e-14 in testing) whenever the regression is
   well-determined — i.e. the sample count is comfortably larger than
   `20 * num_explanatory_vals + 1` — which is always true for realistic
   causal-discovery datasets (`CAMUV` always regresses on the full dataset; it
   has no `bootstrap()` that would shrink the effective sample size). For a
   sample count smaller than the coefficient count, the two solves can diverge
   substantially.

7. **Scope.** `DirectLiNGAM` (plus its `BootstrapResult`), `CAMUV`, and
   `hsic_test_gamma` are ported. Other `lingam` estimators — `ICALiNGAM`,
   `VARLiNGAM`, `RCD`, `LiM`, and so on — are not available.

## Input validation

`X` (for both `DirectLiNGAM` and `CAMUV`) and `DirectLiNGAM`'s `prior_knowledge`
are coerced with `numpy.asarray(..., dtype="float64")` and must be 2-D and
finite; violations raise `ValueError` with a message close to scikit-learn's. A
non-square or self-contradictory `prior_knowledge` matrix raises `ValueError`
at `fit` time for `DirectLiNGAM`. `CAMUV`'s `prior_knowledge` (a list of
`(from, to)` pairs, not a matrix — see [its docs](/camuv#prior-knowledge)) is
not validated at all, matching `lingam.CAMUV`.
