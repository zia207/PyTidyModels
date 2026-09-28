"""Tests for the bundled datasets."""
import warnings

import pytest

from pytidymodels import (
    datasets,
    parsnip,
    recipes,
    rsample,
    workflows,
    yardstick,
)
from pytidymodels.datasets import _bundled
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")


def test_list_datasets_overview():
    idx = datasets.list_datasets()
    assert set(["dataset", "task", "outcome", "n_rows", "n_cols",
                "description", "source"]) <= set(idx.columns)
    assert len(idx) == len(_bundled._REGISTRY) >= 14


@pytest.mark.parametrize("name", sorted(_bundled._REGISTRY))
def test_every_loader_matches_registry(name):
    df = datasets.load_dataset(name)
    meta = _bundled._REGISTRY[name]
    assert df.shape == (meta["n_rows"], meta["n_cols"])
    # the declared outcome column(s) really exist
    for col in [c.strip() for c in meta["outcome"].split(",")]:
        assert col in df.columns, f"{name}: missing outcome column {col!r}"
    assert not df.empty


def test_named_loaders_exist():
    for name in _bundled._REGISTRY:
        assert hasattr(datasets, f"load_{name}")


def test_unknown_dataset_is_informative():
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.load_dataset("not_a_dataset")


# -- modeling-readiness: bundled data flows through the pipeline -----------

def test_boston_regression_end_to_end():
    boston = datasets.load_boston()
    split = rsample.initial_split(boston, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    rec = (recipes.Recipe(train, formula="medv ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_normalize(all_numeric_predictors()))
    rf = parsnip.rand_forest(trees=80).set_engine("sklearn", random_state=0) \
               .set_mode("regression")
    wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(train)
    res = wf.augment(test)
    rsq = yardstick.rsq(res, "medv", ".pred")["estimate"].iloc[0]
    assert rsq > 0.6      # Boston is very learnable


def test_pima_classification_end_to_end():
    pima = datasets.load_pima()
    split = rsample.initial_split(pima, prop=0.8, strata="Outcome", seed=1)
    train, test = split.training(), split.testing()
    rec = (recipes.Recipe(train, formula="Outcome ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_normalize(all_numeric_predictors()))
    logit = parsnip.logistic_reg().set_mode("classification")
    wf = workflows.Workflow().add_recipe(rec).add_model(logit).fit(train)
    res = wf.augment(test)
    assert ".pred_class" in res
    acc = yardstick.accuracy(res, "Outcome", ".pred_class")["estimate"].iloc[0]
    assert acc > 0.6


def test_nmes_with_categoricals_flows():
    # NMES has string categorical predictors -> exercises step_dummy
    nmes = datasets.load_nmes()
    split = rsample.initial_split(nmes, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    rec = (recipes.Recipe(train, formula="visits ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))
    rf = parsnip.rand_forest(trees=60).set_mode("regression")
    wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(train)
    assert len(wf.predict(test)) == len(test)
