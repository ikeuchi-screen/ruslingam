---
outline: deep
---

# Getting Started

`ruslingam` reimplements algorithms from the Python
[`lingam`](https://github.com/ikeuchi-screen/lingam) package in Rust (via
[PyO3](https://pyo3.rs) / [maturin](https://www.maturin.rs)) and exposes the same
Python API, so it can be used as a drop-in replacement.

## Installation

`ruslingam` is not on PyPI yet, so it is built from source. There is no pre-built
wheel: you need a **Rust toolchain** in addition to Python.

### Prerequisites

- **Python** &ge; 3.8 with `pip`
- **Rust** &ge; 1.85 (the crate uses edition 2024). Install it with
  [rustup](https://rustup.rs):

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

  On Windows, install "Rust" and the "Desktop development with C++" workload from
  the Visual Studio Build Tools instead.

### Option A — install straight from GitHub (recommended)

This builds the extension and installs it into your current environment. Run it
inside a virtualenv (or a conda env) so it does not touch the system Python:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install "git+https://github.com/ikeuchi-screen/ruslingam.git"
```

`pip` picks up the `maturin` build backend automatically; you do not need to
install `maturin` yourself. To pin a specific commit or branch, append `@<ref>`,
e.g. `...ruslingam.git@main`.

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
python -c "from ruslingam import DirectLiNGAM, CAMUV; print('ok')"
```

## Quick start

```python
import numpy as np
from ruslingam import DirectLiNGAM        # instead of: from lingam import DirectLiNGAM

X = np.loadtxt("data.csv", delimiter=",")

model = DirectLiNGAM()
model.fit(X)

model.causal_order_                       # list[int]         - estimated topological order
model.adjacency_matrix_                   # np.ndarray (p, p) - estimated coefficient matrix B
```

See [DirectLiNGAM](/direct-lingam) for the full estimator API,
[CAMUV](/camuv) for the confounder-aware estimator, and
[BootstrapResult](/bootstrap) for the object returned by `model.bootstrap(...)`.

## What is implemented

| Area | Status |
| --- | --- |
| `DirectLiNGAM` with `measure="pwling"` | ✅ |
| Adaptive Lasso pruning (`LassoLarsIC('bic')`) and OLS | ✅ |
| `prior_knowledge` (hard and soft) | ✅ |
| `estimate_total_effect`, `get_error_independence_p_values` | ✅ |
| `bootstrap` and the full `BootstrapResult` API | ✅ |
| `CAMUV` with `independence="hsic"` | ✅ |
| `ruslingam.hsic_test_gamma(x, y)` | ✅ |
| `measure="kernel"`, `measure="pwling_fast"` | ❌ `NotImplementedError` |
| `CAMUV`'s `independence="fcorr"` | ❌ `NotImplementedError` |
| Other `lingam` estimators (`ICALiNGAM`, `VARLiNGAM`, …) | ❌ not ported |

See [Differences from `lingam`](/differences) for the behavioural fine print.

## Development

```bash
cargo test              # Rust unit tests
maturin develop         # (re)build the extension
pytest                  # Python tests; tests/test_parity.py compares against `lingam`
                        # and is skipped automatically if `lingam` is not installed
```

There are also speed benchmarks against the reference implementation:

```bash
python benchmarks/bench_direct_lingam.py   # lingam.DirectLiNGAM vs ruslingam.DirectLiNGAM
python benchmarks/bench_camuv.py           # lingam.CAMUV vs ruslingam.CAMUV
python benchmarks/bench_hsic.py            # lingam.hsic.hsic_test_gamma vs ruslingam.hsic_test_gamma
```
