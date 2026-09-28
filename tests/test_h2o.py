"""Tests for the optional H2O engine.

Build-time guardrails run always. Training tests need a working H2O cluster
(h2o package + a Java runtime); the module skips cleanly if either is missing.
A single cluster is shared across the module and shut down at the end.
"""
import warnings

import pandas as pd
import pytest

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")

try:
    import h2o  # noqa: F401
    HAS_H2O = True
except (ImportError, OSError):
    HAS_H2O = False


@pytest.fixture(scope="module", autouse=True)
def _cluster():
    if not HAS_H2O:
        yield
        return
    try:
        parsnip.h2o_init(max_mem_size="1200M", nthreads=2)
    except Exception as e:            # no Java / cluster won't start
        pytest.skip(f"H2O cluster unavailable: {e}")
    yield
    parsnip.h2o_shutdown()


def _rec(train, outcome):
    return (recipes.Recipe(train, formula=f"{outcome} ~ .")
            .step_impute_median(all_numeric_predictors())
            .step_dummy(all_nominal_predictors())
            .step_normalize(all_numeric_predictors()))


# -- guardrails -------------------------------------------------------------

def test_list_h2o_models():
    models = parsnip.list_h2o_models()
    assert any("GBM" in m for m in models) and any("AutoML" in m for m in models)


def test_h2o_engine_rejects_unsupported_type():
    spec = parsnip.decision_tree().set_engine("h2o").set_mode("regression")
    with pytest.raises(ValueError, match="engine 'h2o' is available"):
        spec.fit_xy(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), pd.Series([1, 2, 3]))


def test_h2o_automl_requires_mode():
    with pytest.raises(ValueError, match="mode"):
        parsnip.h2o_automl(max_models=2).fit_xy(
            pd.DataFrame({"a": [1.0, 2.0, 3.0]}), pd.Series([1, 2, 3]))


# -- integration ------------------------------------------------------------

@pytest.mark.skipif(not HAS_H2O, reason="requires h2o + Java")
def test_h2o_regression():
    boston = datasets.load_boston()
    split = rsample.initial_split(boston, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    for spec in [parsnip.linear_reg(penalty=0.0),
                 parsnip.boost_tree(trees=60, tree_depth=4, learn_rate=0.1)]:
        wf = (workflows.Workflow().add_recipe(_rec(train, "medv"))
              .add_model(spec.set_engine("h2o").set_mode("regression")).fit(train))
        preds = wf.predict(test)
        assert list(preds.columns) == [".pred"]
        r2 = yardstick.rsq(wf.augment(test), "medv", ".pred")["estimate"].iloc[0]
        assert r2 > 0.5


@pytest.mark.skipif(not HAS_H2O, reason="requires h2o + Java")
def test_h2o_classification_probabilities():
    pima = datasets.load_pima()
    split = rsample.initial_split(pima, prop=0.8, strata="Outcome", seed=1)
    train, test = split.training(), split.testing()
    wf = (workflows.Workflow().add_recipe(_rec(train, "Outcome"))
          .add_model(parsnip.rand_forest(trees=80).set_engine("h2o")
                     .set_mode("classification")).fit(train))
    res = wf.augment(test)
    assert {".pred_class", ".pred_0", ".pred_1"} <= set(res.columns)
    auc = yardstick.roc_auc(res, "Outcome", ".pred_1")["estimate"].iloc[0]
    assert 0.5 < auc <= 1.0


@pytest.mark.skipif(not HAS_H2O, reason="requires h2o + Java")
def test_h2o_string_label_alignment():
    credit = datasets.load_credit_small(n=300, seed=7)
    split = rsample.initial_split(credit, prop=0.8, strata="status", seed=1)
    train, test = split.training(), split.testing()
    wf = (workflows.Workflow().add_recipe(_rec(train, "status"))
          .add_model(parsnip.boost_tree(trees=60).set_engine("h2o")
                     .set_mode("classification")).fit(train))
    res = wf.augment(test)
    # probability columns must be named/aligned by the string levels
    assert {".pred_bad", ".pred_good"} <= set(res.columns)
    auc = yardstick.roc_auc(res, "status", ".pred_good")["estimate"].iloc[0]
    assert 0.0 <= auc <= 1.0


@pytest.mark.skipif(not HAS_H2O, reason="requires h2o + Java")
def test_h2o_automl_leaderboard_and_predict():
    boston = datasets.load_boston()
    split = rsample.initial_split(boston, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    aml = parsnip.h2o_automl(mode="regression", max_models=2, seed=1)
    wf = (workflows.Workflow().add_recipe(_rec(train, "medv"))
          .add_model(aml).fit(train))
    lb = wf.extract_fit_engine().leaderboard()
    assert len(lb) >= 1 and "model_id" in lb.columns
    assert len(wf.predict(test)) == len(test)
