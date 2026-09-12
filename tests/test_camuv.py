"""Self-contained tests for ruslingam.CAMUV (no scikit-learn / lingam needed)."""

import numpy as np
import pytest

import ruslingam


def test_fit_recovers_parent_and_confounded_pair(confounded_data):
    X, truth = confounded_data
    model = ruslingam.CAMUV().fit(X)
    B = model.adjacency_matrix_
    p = X.shape[1]
    assert B.shape == (p, p)

    child, parent = truth["parent_edge"]
    assert B[child, parent] == 1.0
    assert B[parent, child] == 0.0

    i, j = truth["confounded_pair"]
    assert np.isnan(B[i, j])
    assert np.isnan(B[j, i])

    k = truth["independent"]
    for other in range(p):
        if other == k:
            continue
        assert B[k, other] == 0.0
        assert B[other, k] == 0.0

    # everything else is exactly zero (no spurious edges/confounding)
    known = {(child, parent), (parent, child), (i, j), (j, i)}
    known |= {(k, o) for o in range(p)} | {(o, k) for o in range(p)}
    for a in range(p):
        for b in range(p):
            if a == b or (a, b) in known:
                continue
            assert B[a, b] == 0.0


def test_fit_returns_self(confounded_data):
    X, _ = confounded_data
    model = ruslingam.CAMUV()
    assert model.fit(X) is model


def test_fit_accepts_list_input(confounded_data):
    X, truth = confounded_data
    model = ruslingam.CAMUV().fit(X.tolist())
    child, parent = truth["parent_edge"]
    assert model.adjacency_matrix_[child, parent] == 1.0


def test_adjacency_matrix_none_before_fit():
    model = ruslingam.CAMUV()
    assert model.adjacency_matrix_ is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_explanatory_vals": 0},
        {"num_explanatory_vals": -1},
        {"alpha": -0.1},
        {"ind_corr": -0.1},
    ],
)
def test_bad_constructor_args_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        ruslingam.CAMUV(**kwargs)


def test_unknown_independence_raises_value_error():
    with pytest.raises(ValueError):
        ruslingam.CAMUV(independence="bogus")


def test_fcorr_not_implemented():
    with pytest.raises(NotImplementedError):
        ruslingam.CAMUV(independence="fcorr")


def test_prior_knowledge_forbids_forbidden_parent(confounded_data):
    X, truth = confounded_data
    child, parent = truth["parent_edge"]
    # forbid the true edge ((from, to) = (parent, child)); CAMUV falls back to
    # flagging the pair as a suspected latent confounder instead of a direct
    # edge (verified against real lingam.CAMUV on this exact data).
    model = ruslingam.CAMUV(prior_knowledge=[(parent, child)]).fit(X)
    B = model.adjacency_matrix_
    assert np.isnan(B[parent, child])
    assert np.isnan(B[child, parent])


def test_bad_input_raises_value_error(confounded_data):
    X, _ = confounded_data
    with pytest.raises(ValueError):
        ruslingam.CAMUV().fit(X[:, 0])  # 1-D
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        ruslingam.CAMUV().fit(bad)
