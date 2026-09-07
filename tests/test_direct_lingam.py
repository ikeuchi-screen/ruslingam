"""Self-contained tests for ruslingam.DirectLiNGAM (no scikit-learn / lingam needed)."""

import numpy as np
import pytest

import ruslingam


def _is_topological(order, B_true):
    pos = {v: i for i, v in enumerate(order)}
    p = B_true.shape[0]
    for i in range(p):
        for j in range(p):
            if B_true[i, j] != 0:  # edge j -> i
                if pos[j] > pos[i]:
                    return False
    return True


@pytest.mark.parametrize("adaptive_lasso", [False, True])
def test_fit_recovers_dag(linear_non_gaussian_data, adaptive_lasso):
    X, B_true = linear_non_gaussian_data
    model = ruslingam.DirectLiNGAM(adaptive_lasso=adaptive_lasso).fit(X)

    assert isinstance(model.causal_order_, list)
    assert sorted(model.causal_order_) == [0, 1, 2, 3]
    assert _is_topological(model.causal_order_, B_true)

    B = model.adjacency_matrix_
    assert B.shape == (4, 4)
    # strong true edges are recovered; OLS (adaptive_lasso=False) leaves small
    # spurious coefficients on non-edges, so use a generous tolerance.
    assert np.allclose(B, B_true, atol=0.3)
    for (i, j) in [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2)]:
        assert B[i, j] == pytest.approx(B_true[i, j], abs=0.15)
    # upper triangle w.r.t. the causal order must be exactly zero
    order = model.causal_order_
    for a in range(4):
        for b in range(a + 1, 4):
            assert B[order[a], order[b]] == 0.0


def test_fit_returns_self(linear_non_gaussian_data):
    X, _ = linear_non_gaussian_data
    model = ruslingam.DirectLiNGAM()
    assert model.fit(X) is model


def test_fit_accepts_list_input(linear_non_gaussian_data):
    X, _ = linear_non_gaussian_data
    model = ruslingam.DirectLiNGAM(adaptive_lasso=False).fit(X.tolist())
    assert sorted(model.causal_order_) == [0, 1, 2, 3]


def test_estimate_total_effect(linear_non_gaussian_data):
    X, _ = linear_non_gaussian_data
    model = ruslingam.DirectLiNGAM().fit(X)
    # true total effect 0 -> 2 is 0.5 + 3*(-2) = -5.5
    assert model.estimate_total_effect(X, 0, 2) == pytest.approx(-5.5, abs=0.2)
    # 0 -> 3 : via 1 and via 2 : 3*1 + (-5.5)*(-1) = 8.5
    assert model.estimate_total_effect(X, 0, 3) == pytest.approx(8.5, abs=0.3)


def test_estimate_total_effect_wrong_order_warns(linear_non_gaussian_data):
    X, _ = linear_non_gaussian_data
    model = ruslingam.DirectLiNGAM().fit(X)
    with pytest.warns(UserWarning):
        model.estimate_total_effect(X, 2, 0)


def test_error_independence_p_values(linear_non_gaussian_data):
    X, _ = linear_non_gaussian_data
    model = ruslingam.DirectLiNGAM().fit(X)
    p = model.get_error_independence_p_values(X)
    assert p.shape == (4, 4)
    assert np.allclose(p, p.T)
    assert np.allclose(np.diag(p), 0.0)
    assert np.all((p >= 0.0) & (p <= 1.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"measure": "kernel"},
        {"measure": "pwling_fast"},
        {"prior_knowledge": np.zeros((4, 4))},
        {"apply_prior_knowledge_softly": True},
    ],
)
def test_unsupported_options_raise(kwargs):
    with pytest.raises(NotImplementedError):
        ruslingam.DirectLiNGAM(**kwargs)


def test_bad_input_raises():
    with pytest.raises(ValueError):
        ruslingam.DirectLiNGAM().fit(np.array([1.0, 2.0, 3.0]))  # 1D
    with pytest.raises(ValueError):
        ruslingam.DirectLiNGAM().fit(np.array([[1.0, np.nan], [2.0, 3.0]]))


class TestBootstrap:
    def test_shapes_and_types(self, linear_non_gaussian_data):
        X, _ = linear_non_gaussian_data
        result = ruslingam.DirectLiNGAM(random_state=42).bootstrap(X, n_sampling=20)
        assert result.adjacency_matrices_.shape == (20, 4, 4)
        assert result.total_effects_.shape == (20, 4, 4)
        assert len(result.resampled_indices_) == 20
        assert len(result.resampled_indices_[0]) == X.shape[0]

    def test_n_sampling_validation(self, linear_non_gaussian_data):
        X, _ = linear_non_gaussian_data
        with pytest.raises(ValueError):
            ruslingam.DirectLiNGAM().bootstrap(X, n_sampling=0)

    def test_probabilities(self, linear_non_gaussian_data):
        X, _ = linear_non_gaussian_data
        result = ruslingam.DirectLiNGAM(random_state=1).bootstrap(X, n_sampling=30)
        bp = result.get_probabilities()
        assert bp.shape == (4, 4)
        assert np.all((bp >= 0.0) & (bp <= 1.0))
        # strong true edges should show up in every resample
        assert bp[1, 0] == pytest.approx(1.0)
        assert bp[2, 1] == pytest.approx(1.0)

    def test_causal_direction_counts(self, linear_non_gaussian_data):
        X, _ = linear_non_gaussian_data
        result = ruslingam.DirectLiNGAM(random_state=1).bootstrap(X, n_sampling=20)
        cdc = result.get_causal_direction_counts(n_directions=4)
        assert set(cdc) == {"from", "to", "count"}
        assert len(cdc["from"]) == len(cdc["to"]) == len(cdc["count"]) <= 4
        assert cdc["count"] == sorted(cdc["count"], reverse=True)

        cdc_s = result.get_causal_direction_counts(split_by_causal_effect_sign=True)
        assert "sign" in cdc_s

    def test_dag_counts(self, linear_non_gaussian_data):
        X, _ = linear_non_gaussian_data
        result = ruslingam.DirectLiNGAM(random_state=1).bootstrap(X, n_sampling=20)
        d = result.get_directed_acyclic_graph_counts(n_dags=3)
        assert set(d) == {"dag", "count"}
        assert sum(d["count"]) <= 20
        assert all(set(x) == {"from", "to"} for x in d["dag"])

    def test_total_causal_effects(self, linear_non_gaussian_data):
        X, _ = linear_non_gaussian_data
        result = ruslingam.DirectLiNGAM(random_state=1).bootstrap(X, n_sampling=20)
        tce = result.get_total_causal_effects(min_causal_effect=0.01)
        assert set(tce) == {"from", "to", "effect", "probability"}
        assert tce["probability"] == sorted(tce["probability"], reverse=True)

    def test_paths(self, linear_non_gaussian_data):
        X, _ = linear_non_gaussian_data
        result = ruslingam.DirectLiNGAM(random_state=1).bootstrap(X, n_sampling=20)
        paths = result.get_paths(0, 3)
        assert set(paths) == {"path", "effect", "probability"}
        assert [0, 1, 2, 3] in paths["path"]
        # the fully-mediated path is found in every resample
        assert paths["probability"][paths["path"].index([0, 1, 2, 3])] == pytest.approx(1.0)

    def test_min_causal_effect_validation(self, linear_non_gaussian_data):
        X, _ = linear_non_gaussian_data
        result = ruslingam.DirectLiNGAM(random_state=1).bootstrap(X, n_sampling=5)
        with pytest.raises(ValueError):
            result.get_probabilities(min_causal_effect=-1.0)
