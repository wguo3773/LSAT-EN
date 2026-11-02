import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from lsaten import (LSATBinary, LSATCox, LSATBinaryCV, LSATCoxCV,
                    make_binary_transfer, make_survival_transfer, survival_outcome,
                    binary_at_horizon, binary_metrics, survival_metrics,
                    stable_features, priority_overlap)
from lsaten._loss import logistic_loss_gradient, cox_loss_gradient, solve


@pytest.fixture
def binary():
    return make_binary_transfer(n_source=100, n_target=70, n_test=50,
                                n_features=12, block_size=4, sigma2_tl=.1)


def fit(model, data):
    return model.fit(data["X_source"], data["y_source"], data["X_target"],
                     data["y_target"], groups=data["groups"])


@pytest.mark.parametrize("endpoint", ["binary", "cox"])
def test_analytic_gradients(endpoint):
    rng = np.random.default_rng(1)
    X, beta = rng.normal(size=(20, 4)), rng.normal(size=4)
    y = rng.integers(0, 2, 20) if endpoint == "binary" else survival_outcome(
        rng.integers(1, 7, 20), rng.integers(0, 2, 20))
    fun = logistic_loss_gradient if endpoint == "binary" else cox_loss_gradient
    error = check_grad(lambda b: fun(X, y, b)[0], lambda b: fun(X, y, b)[1], beta)
    assert error < 1e-5


def test_binary_ridge_matches_sklearn(binary):
    X, y = binary["X_target"], binary["y_target"]
    lam = .1
    beta, intercept, result = solve(X, y, "binary", lam, 0, np.zeros(12), np.zeros(12), 5000, 1e-9)
    reference = LogisticRegression(C=1/(len(y)*lam), solver="lbfgs", tol=1e-10, max_iter=5000).fit(X, y)
    assert result.success
    np.testing.assert_allclose(beta, reference.coef_[0], atol=2e-5)
    np.testing.assert_allclose(intercept, reference.intercept_[0], atol=2e-5)


def test_cox_ridge_matches_sksurv():
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    data = make_survival_transfer(n_features=6, block_size=3)
    X, y = data["X_target"], data["y_target"]
    lam = .1
    beta, _, result = solve(X, y, "cox", lam, 0, np.zeros(6), np.zeros(6), 5000, 1e-9)
    ref = CoxPHSurvivalAnalysis(alpha=lam*sum(y["event"]), ties="breslow", tol=1e-10).fit(X, y)
    assert result.success
    np.testing.assert_allclose(beta, ref.coef_, atol=2e-4)


def test_tau_zero_and_stronger_anchor(binary):
    X, y = binary["X_target"], binary["y_target"]
    anchor = np.linspace(-.2, .2, 12)
    def solve_at(tau, a):
        return solve(X, y, "binary", .05, .5, np.full(12, tau), a, 5000, 1e-8)[0]
    np.testing.assert_allclose(solve_at(0, anchor), solve_at(0, np.zeros(12)), atol=1e-4)
    assert np.linalg.norm(solve_at(1000, anchor)-anchor) < np.linalg.norm(solve_at(.01, anchor)-anchor)


@pytest.mark.parametrize("strategy", ["lsat", "pooled", "target"])
def test_models_and_feature_alignment(binary, strategy):
    model = fit(LSATBinary(strategy=strategy), binary)
    p = model.predict_proba(binary["X_test"])
    assert p.shape == (50, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert model.n_features_in_ == 12
    table = model.feature_table()
    assert len(table) == 12
    np.testing.assert_array_equal(model.selected_indices_, np.arange(12))
    assert binary_metrics(binary["y_test"], p[:, 1])["auc"] > .5
    np.testing.assert_allclose(model.decision_function(binary["X_test"]),
                               binary["X_test"] @ model.coef_ + model.intercept_)


def test_target_only_ignores_source(binary):
    a = fit(LSATBinary(strategy="target"), binary)
    altered = dict(binary, X_source=binary["X_source"]*100+500, y_source=1-binary["y_source"])
    b = fit(LSATBinary(strategy="target"), altered)
    np.testing.assert_array_equal(a.selected_indices_, b.selected_indices_)
    np.testing.assert_allclose(a.coef_, b.coef_)


def test_cv_and_fold_local_preprocessing(binary):
    model = fit(LSATBinaryCV(cv=3, lambdas=(.1,),
                            tau_grid=(0., 1.), coordinate_passes=1), binary)
    assert "screening_budget" not in model.best_params_
    assert model.best_validation_loss_ == model.cv_results_.query("converged").mean_validation_loss.min()
    for (train, val), prep in zip(model.cv_splits_, model.fold_preprocessing_):
        X = np.vstack([binary["X_source"], binary["X_target"][train]])
        np.testing.assert_allclose(prep["means"], X.mean(axis=0))
        assert len(set(train) & set(val)) == 0
    assert clone(model).get_params() == model.get_params()


def test_cv_exhaustive_tau_grid(binary):
    model = fit(LSATBinaryCV(cv=2, lambdas=(.1,),
                            tau_grid=(0., 1.), tau_search="grid"), binary)
    assert len(model.cv_results_) == 8  # three layers, two tau values each
    with pytest.raises(ValueError, match="max_candidates"):
        fit(LSATBinaryCV(cv=2, tau_grid=(0., 1.),
                         tau_search="grid", max_candidates=2), binary)


def test_layer_counts_missing_values_and_frames(binary):
    data = dict(binary)
    data["X_source"] = pd.DataFrame(binary["X_source"], columns=[f"a{i}" for i in range(12)])
    data["X_target"] = pd.DataFrame(binary["X_target"], columns=data["X_source"].columns)
    data["X_source"].iloc[0, 0] = np.nan
    model = fit(LSATBinary(tau={"layer0": 0., "layer1": .1, "layer2": 1.}), data)
    assert model.n_features_in_ == 12
    assert np.all(np.isfinite(model.predict_proba(data["X_target"])))
    with pytest.raises(ValueError, match="columns"):
        model.predict_proba(data["X_target"].iloc[:, ::-1])


def test_cox_cv_survival_and_metrics():
    data = make_survival_transfer(n_source=90, n_target=60, n_test=40, n_features=6, block_size=3)
    model = fit(LSATCoxCV(cv=3, lambdas=(.1,),
                         tau_grid=(0., .5), coordinate_passes=1), data)
    risk = model.predict(data["X_test"])
    prob = model.predict_survival_function(data["X_test"], [0, 1, 2, 5])
    assert prob.shape == (40, 4)
    assert np.all(prob[:, 0] == 1) and np.all(np.diff(prob, axis=1) <= 0)
    assert np.all((prob >= 0) & (prob <= 1))
    metrics = survival_metrics(data["y_target"], data["y_test"], risk, times=[1, 2], on_error="nan")
    assert .5 < metrics["c_index"] <= 1
    assert set(metrics["ipcw_auc"]) == {1., 2.}
    invalid = survival_metrics(data["y_target"], data["y_test"], risk, times=[1e8], on_error="nan")
    assert np.isnan(invalid["ipcw_auc"][1e8]) and invalid["errors"]


def test_cox_tied_times_invariant_to_row_order():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 3))
    y = survival_outcome([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], [1, 0]*6)
    order = rng.permutation(12)
    beta = np.array([.2, -.4, .1])
    a, b = cox_loss_gradient(X, y, beta), cox_loss_gradient(X[order], y[order], beta)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_binary_horizon_censoring_convention():
    y = survival_outcome([1, 1, 2, 2, 3], [1, 0, 1, 0, 0])
    mask, labels = binary_at_horizon(y, 2)
    np.testing.assert_array_equal(mask, [1, 0, 1, 1, 1])
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])


def test_stability_and_overlap(binary):
    models = [fit(LSATBinary(lam=.01), binary) for _ in range(3)]
    table = stable_features(models)
    assert np.all(table.selection_frequency.isin([0, 1]))
    overlap = priority_overlap({"LSAT": table, "Overall": table}, top_n=5)
    assert np.all(overlap.LSAT == overlap.Overall)
    assert len(overlap) <= 5


@pytest.mark.parametrize("kwargs", [{"lam": -1}, {"alpha": 2}, {"tau": -1},
                                   {"strategy": "other"}])
def test_invalid_parameters(binary, kwargs):
    with pytest.raises(ValueError):
        fit(LSATBinary(**kwargs), binary)


def test_simulation_variance_and_reproducibility():
    a = make_binary_transfer(n_features=12, block_size=4, sigma2_tl=0)
    b = make_binary_transfer(n_features=12, block_size=4, sigma2_tl=0)
    np.testing.assert_array_equal(a["X_source"], b["X_source"])
    np.testing.assert_array_equal(a["beta_source"], a["beta_target"])


def test_all_features_retained_when_p_exceeds_n():
    data = make_binary_transfer(n_source=50, n_target=30, n_test=20,
                                n_features=80, block_size=10, random_state=7)
    data["X_source"][:, 0] = np.nan
    data["X_target"][:, 0] = np.nan
    model = fit(LSATBinary(lam=.2), data)
    assert model.n_features_in_ == 80
    assert len(model.feature_table()) == 80
    assert model.coef_standardized_[0] == 0
    assert np.all(np.isfinite(model.predict_proba(data["X_test"])))


def test_no_screening_parameters():
    assert "screening_budget" not in LSATBinary().get_params()
    assert "screening_budgets" not in LSATBinaryCV().get_params()
    with pytest.raises(TypeError):
        LSATBinaryCV(screening_budgets=(376, 566))


def test_large_l1_penalty_yields_sparse_coefficients(binary):
    model = fit(LSATBinary(lam=100, alpha=1, tau=0), binary)
    assert model.n_nonzero_ == 0


def test_censoring_before_horizon_is_never_labeled_alive():
    y = survival_outcome([.5, 1., 2., 3.], [0, 0, 0, 0])
    mask, labels = binary_at_horizon(y, 2.)
    np.testing.assert_array_equal(mask, [False, False, True, True])
    np.testing.assert_array_equal(labels, [1, 1])

