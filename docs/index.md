---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "ruslingam"
  text: "Accelerating LiNGAM with Rust."
  tagline: A drop-in Rust reimplementation of the Python lingam package, exposing the same API via PyO3.
  actions:
    - theme: brand
      text: Getting Started
      link: /getting-started
    - theme: alt
      text: DirectLiNGAM API
      link: /direct-lingam
    - theme: alt
      text: View on GitHub
      link: https://github.com/ikeuchi-screen/ruslingam

features:
  - title: Drop-in replacement
    details: >-
      Import `DirectLiNGAM` from `ruslingam` instead of `lingam`. The constructor,
      `fit`, `bootstrap`, and the fitted attributes match the reference API.
  - title: Rust core
    details: >-
      The causal-order search, Adaptive Lasso pruning (LassoLarsIC "bic"), total-effect
      estimation, and the gamma-approximation HSIC test are reimplemented in Rust.
  - title: Parallel search
    details: >-
      The pairwise-likelihood search and the error-independence HSIC tests run on a
      Rayon thread pool. Set the worker count from Python or via `RAYON_NUM_THREADS`.
  - title: Numerically faithful
    details: >-
      Results are validated against `lingam` in `tests/test_parity.py`; causal order
      and edge selection agree on well-conditioned data.
---
