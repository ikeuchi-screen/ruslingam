import numpy as np
import pytest


@pytest.fixture
def linear_non_gaussian_data():
    """A 4-variable linear non-Gaussian SEM with a known DAG.

    x0 -> x1 -> x2,  x0 -> x2,  x1 -> x3,  x2 -> x3
    """
    rng = np.random.default_rng(0)
    n = 2000

    def noise():
        return rng.uniform(-1.0, 1.0, n) ** 3  # sub-Gaussian

    x0 = noise()
    x1 = 3.0 * x0 + noise()
    x2 = -2.0 * x1 + 0.5 * x0 + noise()
    x3 = 1.0 * x1 - 1.0 * x2 + noise()

    X = np.c_[x0, x1, x2, x3]
    B_true = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.5, -2.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
        ]
    )
    return X, B_true
