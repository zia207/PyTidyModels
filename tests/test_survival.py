"""Tests for survival (censored) analysis."""
import warnings

import numpy as np
import pandas as pd
import pytest

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.parsnip.survival_engine import _check_event, is_survival
from pytidymodels.recipes import all_numeric_predictors

warnings.filterwarnings("ignore")

try:
    import lifelines  # noqa: F401
    import sksurv  # noqa: F401
    HAS_SURVIVAL = True
except (ImportError, OSError):
    HAS_SURVIVAL = False


def _lung():
    lung = datasets.load_lung().copy()
    lung["event"] = (lung["status"] == 2).astype(int)
    return lung[["time", "event", "age", "sex", "ph.ecog", "ph.karno", "wt.loss"]]


def _rec(train):
    return (recipes.Recipe(train, formula="time + event ~ .")
            .step_impute_median(all_numeric_predictors())
            .step_normalize(all_numeric_predictors()))


# -- guardrails (no survival libs needed) ----------------------------------

def test_survival_specs_and_modes():
    assert is_survival("survival") and is_survival("censored regression")
    assert not is_survival("regression")
    for spec in [parsnip.proportional_hazards(), parsnip.survival_reg(dist="weibull")]:
        assert spec.set_mode("censored regression").mode == "censored regression"


def test_check_event_rejects_bad_coding():
    _check_event(np.array([0, 1, 1, 0]))          # ok
    with pytest.raises(ValueError, match="0 .censored. / 1 .event."):
        _check_event(np.array([1, 2, 2, 1]))       # raw survival codes -> error


def test_survival_needs_two_column_outcome():
    lung = _lung()
    train = rsample.initial_split(lung, prop=0.8, seed=1).training()
    rec = recipes.Recipe(train, formula="time ~ .")   # only one outcome
    wf = (workflows.Workflow().add_recipe(rec)
          .add_model(parsnip.proportional_hazards().set_mode("survival")))
    with pytest.raises(ValueError, match="two-column outcome"):
        wf.fit(train)


# -- integration (needs lifelines + scikit-survival) -----------------------

@pytest.mark.skipif(not HAS_SURVIVAL, reason="requires lifelines + scikit-survival")
def test_cox_lifelines_end_to_end():
    lung = _lung()
    split = rsample.initial_split(lung, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    wf = (workflows.Workflow().add_recipe(_rec(train))
          .add_model(parsnip.proportional_hazards(penalty=0.1)
                     .set_mode("censored regression")).fit(train))
    preds = wf.predict(test)
    assert list(preds.columns) == [".pred_time", ".pred_risk"]
    surv = wf.predict_survival(test, [100, 300, 500])
    assert surv.shape == (len(test), 3)
    assert ((surv.values >= 0) & (surv.values <= 1)).all()
    m = yardstick.survival_metrics(wf, train, test)
    c = m.loc[m.metric == "concordance", "estimate"].iloc[0]
    assert 0.5 < c <= 1.0


@pytest.mark.skipif(not HAS_SURVIVAL, reason="requires scikit-survival")
def test_random_survival_forest_sksurv():
    lung = _lung()
    split = rsample.initial_split(lung, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    rsf = (parsnip.rand_forest(trees=80, min_n=8)
           .set_engine("sksurv").set_mode("censored regression"))
    wf = workflows.Workflow().add_recipe(_rec(train)).add_model(rsf).fit(train)
    assert list(wf.predict(test).columns) == [".pred_time", ".pred_risk"]
    m = yardstick.survival_metrics(wf, train, test)
    assert "concordance" in set(m["metric"])


@pytest.mark.skipif(not HAS_SURVIVAL, reason="requires scikit-survival")
def test_concordance_survival_standalone():
    lung = _lung()
    split = rsample.initial_split(lung, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    wf = (workflows.Workflow().add_recipe(_rec(train))
          .add_model(parsnip.proportional_hazards().set_mode("survival")).fit(train))
    scored = pd.concat([test.reset_index(drop=True),
                        wf.predict(test).reset_index(drop=True)], axis=1)
    out = yardstick.concordance_survival(scored, "time", "event", ".pred_risk")
    assert out["metric"].iloc[0] == "concordance"
    assert 0.0 <= out["estimate"].iloc[0] <= 1.0


@pytest.mark.skipif(not HAS_SURVIVAL, reason="requires lifelines")
def test_aft_survival_reg():
    lung = _lung()
    split = rsample.initial_split(lung, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    aft = parsnip.survival_reg(dist="lognormal").set_mode("survival")
    wf = workflows.Workflow().add_recipe(_rec(train)).add_model(aft).fit(train)
    assert len(wf.predict(test)) == len(test)
