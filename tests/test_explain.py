"""Tests for SHAP interpretation."""
import warnings

import pytest

from pytidymodels import datasets, explain, parsnip, recipes, rsample, workflows
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")

try:
    import shap  # noqa: F401
    HAS_SHAP = True
except (ImportError, OSError):
    HAS_SHAP = False


def _fitted(model, outcome="price", n=200):
    if outcome == "price":
        data = datasets.load_ames_small(n=n, seed=42)
    else:
        data = datasets.load_credit_small(n=n, seed=7)
    split = rsample.initial_split(data, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    rec = (recipes.Recipe(train, formula=f"{outcome} ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))
    wf = workflows.Workflow().add_recipe(rec).add_model(model).fit(train)
    return wf, test


@pytest.mark.skipif(HAS_SHAP, reason="shap installed; missing-shap path doesn't apply")
def test_missing_shap_is_informative():
    wf, test = _fitted(parsnip.rand_forest(trees=30).set_mode("regression"))
    with pytest.raises(ImportError, match=r"pytidymodels\[shap\]"):
        explain.explain(wf, test)


@pytest.mark.skipif(not HAS_SHAP, reason="requires shap")
def test_explain_tree_model_importance():
    wf, test = _fitted(parsnip.rand_forest(trees=60).set_mode("regression"))
    exp = explain.explain(wf, test, seed=0)
    imp = exp.importance()
    assert set(imp.columns) == {"feature", "importance"}
    assert (imp["importance"] >= 0).all()
    # importance is sorted descending
    assert list(imp["importance"]) == sorted(imp["importance"], reverse=True)
    # SHAP matrix has a value per row x feature
    assert exp.values.shape == (len(test), len(exp.features))


@pytest.mark.skipif(not HAS_SHAP, reason="requires shap")
def test_explain_to_frame_long():
    wf, test = _fitted(parsnip.rand_forest(trees=40).set_mode("regression"))
    exp = explain.explain(wf, test, seed=0)
    long = exp.to_frame(long=True)
    assert set(long.columns) == {"row", "feature", "shap_value"}
    assert len(long) == len(test) * len(exp.features)


@pytest.mark.skipif(not HAS_SHAP, reason="requires shap")
def test_explain_model_agnostic_linear():
    wf, test = _fitted(parsnip.linear_reg().set_mode("regression"))
    exp = explain.explain(wf, test.head(40), n_background=40, seed=0)
    assert exp.values.shape[0] == 40


@pytest.mark.skipif(not HAS_SHAP, reason="requires shap")
def test_vip_convenience_and_classification():
    wf, test = _fitted(parsnip.rand_forest(trees=60).set_mode("classification"),
                       outcome="status")
    imp = explain.vip(wf, test.head(60), seed=0)
    assert set(imp.columns) == {"feature", "importance"}
    assert len(imp) == len(wf.extract_recipe().predictor_names_)


@pytest.mark.skipif(not HAS_SHAP, reason="requires shap")
def test_summary_plot_saves_png(tmp_path):
    wf, test = _fitted(parsnip.rand_forest(trees=40).set_mode("regression"))
    exp = explain.explain(wf, test, seed=0)
    out = exp.summary_plot(path=str(tmp_path / "shap.png"))
    assert out.endswith("shap.png")
    assert (tmp_path / "shap.png").stat().st_size > 0
