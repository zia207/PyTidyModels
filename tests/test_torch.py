"""Tests for the optional PyTorch engine.

Build-time guardrails run always; the actual training tests run only where
torch is installed and otherwise skip, so the suite stays green without torch.
"""
import warnings

import pandas as pd
import pytest

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except (ImportError, OSError):  # OSError: broken/partial install
    HAS_TORCH = False


def _rec(train, outcome):
    return (recipes.Recipe(train, formula=f"{outcome} ~ .")
            .step_impute_median(all_numeric_predictors())
            .step_impute_mode(all_nominal_predictors())
            .step_dummy(all_nominal_predictors())
            .step_normalize(all_numeric_predictors()))


# -- introspection (always) ------------------------------------------------

def test_torch_symbols_exist():
    assert hasattr(parsnip, "torch_model")
    assert hasattr(parsnip, "list_torch_models")
    assert isinstance(parsnip.list_torch_models(), list)


# -- build-time guardrails (always, independent of torch) ------------------

def test_torch_engine_only_for_mlp():
    spec = parsnip.rand_forest().set_engine("torch").set_mode("regression")
    with pytest.raises(ValueError, match="mlp only"):
        spec.fit_xy(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), pd.Series([1, 2, 3]))


def test_torch_model_requires_mode():
    spec = parsnip.torch_model(lambda i, o: None)  # no mode
    with pytest.raises(ValueError, match="mode"):
        spec.fit_xy(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), pd.Series([1, 2, 3]))


@pytest.mark.skipif(HAS_TORCH, reason="torch installed; the missing-torch path doesn't apply")
def test_missing_torch_is_informative():
    X = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4], "b": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    spec = parsnip.mlp(hidden_units=8, epochs=2).set_engine("torch").set_mode("regression")
    with pytest.raises(ImportError, match=r"pytidymodels\[torch\]"):
        spec.fit_xy(X, y)


# -- training integration (only where torch is installed) ------------------

@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_mlp_regression():
    ames = datasets.load_ames_small(n=300, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    train, test = split.training(), split.testing()
    spec = (parsnip.mlp(hidden_units=(64, 32), epochs=150, learn_rate=1e-2)
            .set_engine("torch", seed=0).set_mode("regression"))
    wf = workflows.Workflow().add_recipe(_rec(train, "price")).add_model(spec).fit(train)
    res = wf.augment(test)
    assert list(wf.predict(test).columns) == [".pred"]
    r2 = yardstick.rsq(res, "price", ".pred")["estimate"].iloc[0]
    assert r2 > 0.3  # learns real signal


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_mlp_classification_probabilities():
    credit = datasets.load_credit_small(n=300, seed=7)
    split = rsample.initial_split(credit, prop=0.8, strata="status", seed=1)
    train, test = split.training(), split.testing()
    spec = (parsnip.mlp(hidden_units=32, epochs=150)
            .set_engine("torch", seed=0).set_mode("classification"))
    wf = workflows.Workflow().add_recipe(_rec(train, "status")).add_model(spec).fit(train)
    res = wf.augment(test)
    assert ".pred_class" in res and ".pred_good" in res
    auc = yardstick.roc_auc(res, "status", ".pred_good")["estimate"].iloc[0]
    assert 0.0 <= auc <= 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_model_wraps_arbitrary_module():
    import torch.nn as nn

    ames = datasets.load_ames_small(n=200, seed=0)
    split = rsample.initial_split(ames, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    rec = _rec(train, "price")

    # a factory that adapts to the design-matrix width at fit time
    def factory(in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, 48), nn.ReLU(),
            nn.Linear(48, out_features),
        )

    spec = parsnip.torch_model(factory, mode="regression", epochs=100, seed=0)
    wf = workflows.Workflow().add_recipe(rec).add_model(spec).fit(train)
    assert len(wf.predict(test)) == len(test)
