"""Tests covering the Phase 0-1 vertical slice."""
import numpy as np
import pandas as pd
import pytest

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors


# -- rsample ---------------------------------------------------------------

def test_initial_split_partitions_all_rows():
    df = datasets.load_ames_small(n=200, seed=0)
    split = rsample.initial_split(df, prop=0.75, seed=1)
    train, test = split.training(), split.testing()
    assert len(train) + len(test) == len(df)
    assert abs(len(train) / len(df) - 0.75) < 0.05


def test_vfold_cv_covers_data_once():
    df = datasets.load_ames_small(n=100, seed=0)
    folds = rsample.vfold_cv(df, v=5, seed=1)
    assert len(folds) == 5
    seen = np.concatenate([f.assessment_idx for f in folds])
    assert sorted(seen) == list(range(len(df)))


# -- recipes ---------------------------------------------------------------

def test_recipe_prep_bake_no_missing_and_numeric_design():
    df = datasets.load_ames_small(n=200, seed=0)
    rec = (recipes.Recipe(df, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))
    prepped = rec.prep(df)
    X = prepped.bake_predictors(df)
    assert not X.isna().any().any()
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    # normalized numeric columns are ~zero mean
    assert abs(X["area"].mean()) < 1e-6


def test_recipe_predict_alignment_on_unseen_levels():
    train = datasets.load_ames_small(n=200, seed=0)
    rec = (recipes.Recipe(train, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_dummy(all_nominal_predictors()))
    prepped = rec.prep(train)
    # new data with an unseen categorical level must still align to blueprint
    new = train.head(5).copy()
    new.loc[new.index[0], "neighborhood"] = "Atlantis"
    X = prepped.bake_predictors(new)
    assert list(X.columns) == prepped.predictor_names_


# -- parsnip ---------------------------------------------------------------

def test_regression_predict_is_tidy():
    df = datasets.load_ames_small(n=150, seed=0).dropna()
    X = df[["area", "age"]]
    y = df["price"]
    fit = parsnip.linear_reg().set_engine("sklearn").set_mode("regression").fit_xy(X, y)
    preds = fit.predict(X)
    assert list(preds.columns) == [".pred"]
    assert len(preds) == len(df)


def test_unresolved_tune_raises():
    with pytest.raises(ValueError, match="tune"):
        spec = (parsnip.rand_forest(mtry=parsnip.tune())
                .set_engine("sklearn").set_mode("regression"))
        spec.fit_xy(pd.DataFrame({"a": [1, 2, 3]}), pd.Series([1, 2, 3]))


def test_unknown_engine_is_informative():
    df = datasets.load_ames_small(n=50, seed=0).dropna()
    with pytest.raises(ValueError, match="No engine"):
        (parsnip.rand_forest().set_engine("tensorflow").set_mode("regression")
         .fit_xy(df[["area"]], df["price"]))


# -- workflows + yardstick -------------------------------------------------

def test_workflow_end_to_end_regression():
    ames = datasets.load_ames_small(n=400, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    rec = (recipes.Recipe(split.training(), formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))
    rf = (parsnip.rand_forest(trees=100, min_n=5)
          .set_engine("sklearn", random_state=0).set_mode("regression"))
    wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(split.training())
    results = wf.augment(split.testing())
    out = yardstick.metric_set(yardstick.rmse, yardstick.rsq)(
        results, truth="price", estimate=".pred")
    assert set(out["metric"]) == {"rmse", "rsq"}
    rsq = out.loc[out.metric == "rsq", "estimate"].iloc[0]
    assert rsq > 0.4  # comfortably above chance: the pipeline learns real signal


def test_workflow_classification_and_roc_auc():
    credit = datasets.load_credit_small(n=400, seed=7)
    split = rsample.initial_split(credit, prop=0.8, strata="status", seed=1)
    rec = (recipes.Recipe(split.training(), formula="status ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))
    logit = parsnip.logistic_reg().set_engine("sklearn").set_mode("classification")
    wf = workflows.Workflow().add_recipe(rec).add_model(logit).fit(split.training())
    results = wf.augment(split.testing())
    assert ".pred_class" in results and ".pred_good" in results
    auc = yardstick.roc_auc(results, "status", ".pred_good")["estimate"].iloc[0]
    assert 0.5 <= auc <= 1.0


def test_extract_fit_engine_returns_sklearn_object():
    df = datasets.load_ames_small(n=200, seed=0)
    rec = (recipes.Recipe(df, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_dummy(all_nominal_predictors()))
    rf = parsnip.rand_forest(trees=50).set_engine("sklearn").set_mode("regression")
    wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(df)
    from sklearn.ensemble import RandomForestRegressor
    assert isinstance(wf.extract_fit_engine(), RandomForestRegressor)
