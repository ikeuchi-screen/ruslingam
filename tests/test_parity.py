"""Cross-check ruslingam.DirectLiNGAM against the reference `lingam` implementation.

Skipped automatically when `lingam` / scikit-learn are not installed, so the rest
of the suite still runs offline.
"""

import warnings

import numpy as np
import pytest

import ruslingam

lingam = pytest.importorskip("lingam")


def _random_sem(seed, n=800, p=6, density=0.4):
    rng = np.random.default_rng(seed)
    B = np.zeros((p, p))
    for i in range(p):
        for j in range(i):
            if rng.random() < density:
                B[i, j] = rng.uniform(-2.5, 2.5)
    X = np.zeros((n, p))
    for i in range(p):
        e = rng.uniform(-1.0, 1.0, n) ** 3
        X[:, i] = e + X @ B[i]
    perm = rng.permutation(p)
    return X[:, perm]


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("adaptive_lasso", [False, True])
def test_matches_lingam(seed, adaptive_lasso):
    X = _random_sem(seed)

    ref = lingam.DirectLiNGAM(adaptive_lasso=adaptive_lasso).fit(X)
    rus = ruslingam.DirectLiNGAM(adaptive_lasso=adaptive_lasso).fit(X)

    assert list(map(int, rus.causal_order_)) == list(map(int, ref.causal_order_))

    # identical sparsity pattern (adaptive lasso pruning agrees)
    ref_edges = np.abs(ref.adjacency_matrix_) > 0
    rus_edges = np.abs(rus.adjacency_matrix_) > 0
    assert np.array_equal(ref_edges, rus_edges)

    assert np.allclose(ref.adjacency_matrix_, rus.adjacency_matrix_, atol=1e-6)

    p_ref = ref.get_error_independence_p_values(X)
    p_rus = rus.get_error_independence_p_values(X)
    assert np.allclose(p_ref, p_rus, atol=1e-6)


def test_total_effect_matches_lingam():
    X = _random_sem(0)
    ref = lingam.DirectLiNGAM().fit(X)
    rus = ruslingam.DirectLiNGAM().fit(X)
    order = list(map(int, ref.causal_order_))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                f, t = order[a], order[b]
                assert ref.estimate_total_effect(X, f, t) == pytest.approx(
                    rus.estimate_total_effect(X, f, t), abs=1e-6
                )
