"""Tests for hyperparameter tuning."""
import warnings

import pytest

from pytidymodels import (
    datasets,
    dials,
    parsnip,
    recipes,
    rsample,
    tune,
    workflows,
    yardstick,
)
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")


def _rec(train, outcome):
    return (recipes.Recipe(train, formula=f"{outcome} ~ .")
            .step_impute_median(all_numeric_predictors())
            .step_impute_mode(all_nominal_predictors())
            .step_dummy(all_nominal_predictors())
            .step_normalize(all_numeric_predictors()))


@pytest.fixture(scope="module")
def reg_setup():
    ames = datasets.load_ames_small(n=250, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    train = split.training()
    wf = (workflows.Workflow().add_recipe(_rec(train, "price"))
          .add_model(parsnip.rand_forest(mtry=parsnip.tune(),
                                         min_n=parsnip.tune(), trees=80)
                     .set_mode("regression")))
    folds = rsample.vfold_cv(train, v=3, seed=2)
    grid = dials.grid_regular(dials.mtry(range=(2, 6)),
                              dials.min_n(range=(2, 20)), levels=2)
    return wf, folds, grid, split


def test_tune_grid_shape(reg_setup):
    wf, folds, grid, _ = reg_setup
    res = tune.tune_grid(wf, folds, grid,
                         yardstick.metric_set(yardstick.rmse, yardstick.rsq))
    summ = tune.collect_metrics(res)
    assert {"mtry", "min_n", "metric", "mean", "n", "std_err"} <= set(summ.columns)
    # one row per candidate x metric
    assert len(summ) == len(grid) * 2
    assert (summ["n"] == 3).all()          # 3 folds each


def test_show_best_respects_direction(reg_setup):
    wf, folds, grid, _ = reg_setup
    res = tune.tune_grid(wf, folds, grid, yardstick.metric_set(yardstick.rmse))
    best = tune.show_best(res, "rmse", n=3)
    # rmse is minimized -> means must be non-decreasing
    assert list(best["mean"]) == sorted(best["mean"])


def test_select_best_and_finalize(reg_setup):
    wf, folds, grid, split = reg_setup
    res = tune.tune_grid(wf, folds, grid, yardstick.metric_set(yardstick.rmse))
    best = tune.select_best(res, "rmse")
    assert set(best) == {"mtry", "min_n"}
    final = tune.finalize_workflow(wf, best)
    # tune() placeholders are now concrete values
    assert final._model.args["mtry"] == best["mtry"]
    assert not hasattr(final._model.args["mtry"], "__class__") or \
        not isinstance(final._model.args["mtry"], parsnip.tune().__class__)


def test_last_fit_returns_test_metrics(reg_setup):
    wf, folds, grid, split = reg_setup
    res = tune.tune_grid(wf, folds, grid, yardstick.metric_set(yardstick.rmse))
    final = tune.finalize_workflow(wf, tune.select_best(res, "rmse"))
    lf = tune.last_fit(final, split, yardstick.metric_set(yardstick.rmse, yardstick.rsq))
    assert set(lf.metrics["metric"]) == {"rmse", "rsq"}
    assert len(lf.predictions) == len(split.testing())


def test_select_by_one_std_err(reg_setup):
    wf, folds, grid, _ = reg_setup
    res = tune.tune_grid(wf, folds, grid, yardstick.metric_set(yardstick.rmse))
    picked = tune.select_by_one_std_err(res, "rmse", "min_n")
    assert set(picked) == {"mtry", "min_n"}


def test_tune_grid_mixed_int_float_params():
    # regression guard: mixed int/float grids must keep int params integer
    # (a pandas row-iteration gotcha once turned tree_depth into a float).
    ames = datasets.load_ames_small(n=200, seed=42)
    split = rsample.initial_split(ames, prop=0.8, seed=1)
    train = split.training()
    wf = (workflows.Workflow().add_recipe(_rec(train, "price"))
          .add_model(parsnip.boost_tree(trees=parsnip.tune(),
                                        learn_rate=parsnip.tune(),
                                        tree_depth=parsnip.tune())
                     .set_mode("regression")))
    folds = rsample.vfold_cv(train, v=2, seed=2)
    grid = dials.grid_latin_hypercube(
        dials.trees(range=(50, 150)), dials.learn_rate(range=(1e-2, 0.2)),
        dials.tree_depth(range=(2, 5)), size=4, seed=1)
    res = tune.tune_grid(wf, folds, grid, yardstick.metric_set(yardstick.rmse))
    best = tune.select_best(res, "rmse")
    assert isinstance(best["trees"], int) and isinstance(best["tree_depth"], int)
    assert isinstance(best["learn_rate"], float)


def test_tune_grid_classification():
    credit = datasets.load_credit_small(n=250, seed=7)
    split = rsample.initial_split(credit, prop=0.8, strata="status", seed=1)
    train = split.training()
    wf = (workflows.Workflow().add_recipe(_rec(train, "status"))
          .add_model(parsnip.logistic_reg(penalty=parsnip.tune())
                     .set_mode("classification")))
    folds = rsample.vfold_cv(train, v=3, strata="status", seed=2)
    grid = dials.grid_regular(dials.penalty(range=(1e-3, 1.0)), levels=3)
    res = tune.tune_grid(wf, folds, grid,
                         yardstick.metric_set(yardstick.accuracy, yardstick.roc_auc))
    best = tune.select_best(res, "roc_auc")   # maximized
    assert "penalty" in best
    top = tune.show_best(res, "roc_auc", n=3)
    assert list(top["mean"]) == sorted(top["mean"], reverse=True)
