"""Tests for the expanded model catalog and universal scikit-learn access."""
import warnings

import pytest

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")


def _rec(train, outcome):
    return (recipes.Recipe(train, formula=f"{outcome} ~ .")
            .step_impute_median(all_numeric_predictors())
            .step_impute_mode(all_nominal_predictors())
            .step_dummy(all_nominal_predictors())
            .step_normalize(all_numeric_predictors()))


@pytest.fixture(scope="module")
def reg_data():
    ames = datasets.load_ames_small(n=300, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    return split.training(), split.testing()


@pytest.fixture(scope="module")
def clf_data():
    credit = datasets.load_credit_small(n=300, seed=7)
    split = rsample.initial_split(credit, prop=0.8, strata="status", seed=1)
    return split.training(), split.testing()


# -- catalog introspection -------------------------------------------------

def test_catalog_has_expected_types():
    models = set(parsnip.list_models())
    for t in ["rand_forest", "boost_tree", "hist_boost_tree", "decision_tree",
              "svm_rbf", "svm_linear", "nearest_neighbor", "mlp", "naive_Bayes",
              "discrim_linear", "linear_reg", "logistic_reg", "poisson_reg"]:
        assert t in models


def test_list_sklearn_estimators_kinds():
    alln = parsnip.list_sklearn_estimators("all")
    clf = parsnip.list_sklearn_estimators("classifier")
    reg = parsnip.list_sklearn_estimators("regressor")
    assert len(clf) > 20 and len(reg) > 20
    assert set(clf).isdisjoint(reg)
    assert set(clf) | set(reg) <= set(alln)
    assert "RandomForestClassifier" in clf
    assert "HistGradientBoostingRegressor" in reg


# -- curated regression types fit end-to-end -------------------------------

REG_TYPES = [
    parsnip.rand_forest(trees=60),
    parsnip.extra_trees(trees=60),
    parsnip.boost_tree(trees=60, learn_rate=0.1),
    parsnip.hist_boost_tree(trees=60),
    parsnip.decision_tree(tree_depth=5),
    parsnip.bag_tree(trees=30),
    parsnip.adaboost(trees=30),
    parsnip.nearest_neighbor(neighbors=5),
    parsnip.svm_rbf(cost=5),
    parsnip.svm_linear(cost=1),
    parsnip.mlp(hidden_units=16, epochs=200),
    parsnip.linear_reg(),
    parsnip.linear_reg(penalty=0.1, mixture=0.5),
    parsnip.poisson_reg(),
]


@pytest.mark.parametrize("spec", REG_TYPES, ids=lambda s: s.model_type)
def test_curated_regression_fits(spec, reg_data):
    train, test = reg_data
    wf = (workflows.Workflow().add_recipe(_rec(train, "price"))
          .add_model(spec.set_mode("regression")).fit(train))
    preds = wf.predict(test)
    assert list(preds.columns) == [".pred"]
    assert len(preds) == len(test)


# -- curated classification types fit end-to-end ---------------------------

CLF_TYPES = [
    parsnip.rand_forest(trees=60),
    parsnip.boost_tree(trees=60),
    parsnip.hist_boost_tree(trees=60),
    parsnip.decision_tree(tree_depth=5),
    parsnip.nearest_neighbor(neighbors=7),
    parsnip.svm_rbf(cost=5),
    parsnip.mlp(hidden_units=16, epochs=200),
    parsnip.logistic_reg(),
    parsnip.multinom_reg(),
    parsnip.naive_Bayes(),
    parsnip.discrim_linear(),
    parsnip.discrim_quad(),
]


@pytest.mark.parametrize("spec", CLF_TYPES, ids=lambda s: s.model_type)
def test_curated_classification_fits(spec, clf_data):
    train, test = clf_data
    wf = (workflows.Workflow().add_recipe(_rec(train, "status"))
          .add_model(spec.set_mode("classification")).fit(train))
    res = wf.augment(test)
    assert ".pred_class" in res
    acc = yardstick.accuracy(res, "status", ".pred_class")["estimate"].iloc[0]
    assert 0.0 <= acc <= 1.0


def test_svm_classification_yields_probabilities(clf_data):
    train, test = clf_data
    spec = parsnip.svm_rbf(cost=5).set_mode("classification")
    wf = (workflows.Workflow().add_recipe(_rec(train, "status"))
          .add_model(spec).fit(train))
    res = wf.augment(test)
    assert ".pred_good" in res  # probability column present -> roc_auc usable
    auc = yardstick.roc_auc(res, "status", ".pred_good")["estimate"].iloc[0]
    assert 0.0 <= auc <= 1.0


# -- universal sklearn access ----------------------------------------------

def test_sklearn_model_by_name_regression(reg_data):
    train, test = reg_data
    wf = (workflows.Workflow().add_recipe(_rec(train, "price"))
          .add_model(parsnip.sklearn_model("HuberRegressor")).fit(train))
    assert len(wf.predict(test)) == len(test)


def test_sklearn_model_by_class(reg_data):
    from sklearn.linear_model import BayesianRidge
    train, test = reg_data
    wf = (workflows.Workflow().add_recipe(_rec(train, "price"))
          .add_model(parsnip.sklearn_model(BayesianRidge)).fit(train))
    assert len(wf.predict(test)) == len(test)


def test_sklearn_model_by_instance_carries_params(reg_data):
    from sklearn.ensemble import RandomForestRegressor
    train, test = reg_data
    spec = parsnip.sklearn_model(RandomForestRegressor(n_estimators=42, random_state=0))
    wf = (workflows.Workflow().add_recipe(_rec(train, "price"))
          .add_model(spec).fit(train))
    assert wf.extract_fit_engine().n_estimators == 42


def test_sklearn_model_mode_inferred(reg_data):
    train, _ = reg_data
    wf = (workflows.Workflow().add_recipe(_rec(train, "price"))
          .add_model(parsnip.sklearn_model("KernelRidge")).fit(train))
    assert wf.extract_fit_parsnip().mode == "regression"


def test_unknown_sklearn_name_is_informative():
    with pytest.raises(ValueError, match="Unknown scikit-learn estimator"):
        parsnip.sklearn_model("RandomForstRegressor")  # typo -> suggestion


def test_unsupported_mode_for_type_raises(reg_data):
    train, _ = reg_data
    with pytest.raises(ValueError, match="does not support|regression model"):
        (workflows.Workflow().add_recipe(_rec(train, "price"))
         .add_model(parsnip.naive_Bayes().set_mode("regression")).fit(train))
