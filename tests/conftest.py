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


@pytest.fixture
def confounded_data():
    """A 5-variable dataset for CAMUV: a direct parent edge, a pair sharing an
    unobserved latent confounder, and an unrelated column.

    x0 -> x1 (direct edge, x1 = 2*x0 + noise).

    x2 <- L -> x3 (L is dropped from X): L feeds x2/x3 through two *different*
    non-monotonic functions (sin/cos of different frequencies), so neither
    x2 = f(x3) nor x3 = f(x2) is a well-defined smooth function — a purely
    linear shared latent (x2 = a*L, x3 = b*L) is trivially invertible and gets
    identified as a direct edge instead, so this is needed to actually exercise
    CAMUV's confounded-pair detection (verified against real `lingam.CAMUV`).

    x4 is independent of everything else.
    """
    rng = np.random.default_rng(1)
    n = 600

    def noise(scale=1.0):
        return scale * rng.uniform(-1.0, 1.0, n) ** 3

    x0 = noise()
    x1 = 2.0 * x0 + noise()

    latent = rng.uniform(-2.0, 2.0, n)
    x2 = np.sin(2 * latent) + 0.2 * noise()
    x3 = np.cos(3 * latent) + 0.2 * noise()

    x4 = noise()

    X = np.c_[x0, x1, x2, x3, x4]
    return X, {"parent_edge": (1, 0), "confounded_pair": (2, 3), "independent": 4}
