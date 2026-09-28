"""Workflows: bundle preprocessing + model into one fit/predict object.

A workflow is what you actually fit and carry around. It preps the recipe,
bakes the design matrix, fits the model spec, and gives back tidy
predictions -- while still exposing the raw fitted engine and a genuine
``sklearn.Pipeline`` for anyone who needs to escape the abstraction.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from ..parsnip.spec import Spec
from ..pipe import PipeMixin
from ..recipes.recipe import Recipe


class Workflow(PipeMixin):
    def __init__(self):
        self._recipe: Optional[Recipe] = None
        self._model: Optional[Spec] = None
        self._fitted = False
        self._prepped_recipe: Optional[Recipe] = None
        self._fit_model = None

    # -- builder ------------------------------------------------------------

    def add_recipe(self, recipe: Recipe) -> "Workflow":
        self._recipe = recipe
        return self

    def add_model(self, model: Spec) -> "Workflow":
        self._model = model
        return self

    # -- fit / predict ------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "Workflow":
        if self._recipe is None or self._model is None:
            raise ValueError("A workflow needs both add_recipe() and add_model().")
        from ..parsnip.survival_engine import is_survival
        prepped = self._recipe.prep(data)
        X = prepped.bake_predictors(data)
        if is_survival(self._model.mode):
            outs = prepped._outcomes
            if len(outs) < 2:
                raise ValueError(
                    "Survival models need a two-column outcome (time, event); "
                    "use Recipe(data, formula='time + event ~ .').")
            y = prepped.bake(data)[outs]          # column order: time, event
        else:
            y = prepped.get_outcome(data)
        self._prepped_recipe = prepped
        self._fit_model = self._model.fit_xy(X, y)
        self._fitted = True
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("Workflow must be fit() before predicting.")

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        X = self._prepped_recipe.bake_predictors(new_data)
        return self._fit_model.predict(X)

    def predict_survival(self, new_data: pd.DataFrame, times) -> pd.DataFrame:
        """Survival probabilities S(t) at ``times`` (censored/survival models)."""
        self._require_fitted()
        X = self._prepped_recipe.bake_predictors(new_data)
        return self._fit_model.predict_survival(X, times)

    def augment(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Return ``new_data`` with prediction columns appended."""
        preds = self.predict(new_data).reset_index(drop=True)
        return pd.concat([new_data.reset_index(drop=True), preds], axis=1)

    def pipe(self, func, *args, **kwargs):
        """Apply ``func(self, *args, **kwargs)`` -- pandas-style threading."""
        return func(self, *args, **kwargs)

    # -- extractors / interop ----------------------------------------------

    def extract_recipe(self) -> Recipe:
        self._require_fitted()
        return self._prepped_recipe

    def extract_fit_parsnip(self):
        self._require_fitted()
        return self._fit_model

    def extract_fit_engine(self):
        """The raw underlying estimator (sklearn/xgboost/... object)."""
        self._require_fitted()
        return self._fit_model.estimator

    def to_sklearn_pipeline(self, data: pd.DataFrame) -> Pipeline:
        """Compile to a native sklearn Pipeline (recipe transformer + estimator).

        Demonstrates the design claim that a workflow *is* an sklearn
        Pipeline underneath; useful for SHAP, calibration, MLflow logging.
        """
        from ..parsnip import engines
        self._model._check_resolved()
        estimator, _mode = engines.build(self._model)
        pipe = Pipeline([
            ("recipe", self._recipe.to_sklearn_transformer()),
            ("model", estimator),
        ])
        y = self._recipe.prep(data).get_outcome(data)
        return pipe.fit(data, y)

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return (f"<Workflow recipe={self._recipe is not None} "
                f"model={self._model!r} [{state}]>")


# -- free-function forms (pipe-friendly) ------------------------------------

def fit(workflow: "Workflow", data) -> "Workflow":
    """Functional form of ``workflow.fit(data)`` for use in pipes."""
    return workflow.fit(data)


def predict(workflow: "Workflow", new_data):
    """Functional form of ``workflow.predict(new_data)`` for use in pipes."""
    return workflow.predict(new_data)


def augment(workflow: "Workflow", new_data):
    """Functional form of ``workflow.augment(new_data)`` for use in pipes."""
    return workflow.augment(new_data)
