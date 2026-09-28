"""The Recipe object: declare preprocessing, ``prep`` to learn, ``bake`` to apply.

A recipe is built up with chained ``step_*`` calls, then ``prep``-ed on the
training data (learning every parameter, sequentially, so each step sees the
data as transformed by the ones before it), then ``bake``-d on any data. It
also exposes ``to_sklearn_transformer`` so the whole thing drops into a plain
``sklearn.Pipeline`` -- the interoperability hinge of the design.
"""
from __future__ import annotations

import copy
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..hardhat.blueprint import Blueprint
from ..hardhat.schema import ColumnInfo, parse_formula
from ..pipe import PipeMixin
from . import steps as _steps


class Recipe(PipeMixin):
    def __init__(self, data: pd.DataFrame, formula: Optional[str] = None,
                 outcome: Optional[str] = None,
                 outcomes: Optional[list[str]] = None):
        self._template = data.head(0).copy()
        if formula is not None:
            outs, _preds = parse_formula(formula, data)
            outcomes = outs
        self._outcomes = outcomes or ([] if outcome is None else [outcome])
        self._steps: list = []
        self._prepped = False
        # discovered at prep time:
        self.info_: Optional[ColumnInfo] = None
        self.predictor_names_: list[str] = []
        self.blueprint_: Optional[Blueprint] = None

    # -- builder ------------------------------------------------------------

    def _add(self, step):
        self._steps.append(step)
        return self

    def step_impute_median(self, selector):
        return self._add(_steps.step_impute_median(selector))

    def step_impute_mode(self, selector):
        return self._add(_steps.step_impute_mode(selector))

    def step_log(self, selector, base=None, offset=0.0):
        import numpy as np
        return self._add(_steps.step_log(selector, base or np.e, offset))

    def step_other(self, selector, threshold=0.05, other="other"):
        return self._add(_steps.step_other(selector, threshold, other))

    def step_dummy(self, selector, drop_first=True):
        return self._add(_steps.step_dummy(selector, drop_first))

    def step_normalize(self, selector):
        return self._add(_steps.step_normalize(selector))

    def step_zv(self, selector):
        return self._add(_steps.step_zv(selector))

    # -- prep / bake --------------------------------------------------------

    def prep(self, training: pd.DataFrame) -> "Recipe":
        """Learn all step parameters from ``training``; returns a prepped clone."""
        rec = copy.deepcopy(self)
        info = ColumnInfo.from_frame(training, outcomes=rec._outcomes)
        data = training.copy()
        for step in rec._steps:
            step.prep(data, info)
            data = step.bake(data)
        rec.info_ = info
        rec.predictor_names_ = info.predictors()
        rec.blueprint_ = Blueprint(
            predictors=list(rec.predictor_names_),
            outcome=rec._outcomes[0] if rec._outcomes else None,
        )
        rec._prepped = True
        return rec

    def _require_prepped(self):
        if not self._prepped:
            raise RuntimeError("Recipe must be prep()-ed before baking.")

    def bake(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transformation; returns outcome + predictors."""
        self._require_prepped()
        data = new_data.copy()
        for step in self._steps:
            data = step.bake(data)
        return data

    def bake_predictors(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Baked design matrix (predictors only), aligned to the blueprint."""
        baked = self.bake(new_data)
        present = [c for c in self.predictor_names_ if c in baked.columns]
        X = baked[present]
        return self.blueprint_.align(X)

    def get_outcome(self, new_data: pd.DataFrame) -> pd.Series:
        self._require_prepped()
        baked = self.bake(new_data)
        return baked[self.blueprint_.outcome]

    # -- interop ------------------------------------------------------------

    def to_sklearn_transformer(self) -> "RecipeTransformer":
        return RecipeTransformer(self)

    def pipe(self, func, *args, **kwargs):
        """Apply ``func(self, *args, **kwargs)`` -- pandas-style threading."""
        return func(self, *args, **kwargs)

    def __repr__(self) -> str:
        state = "prepped" if self._prepped else "unprepped"
        return f"<Recipe steps={len(self._steps)} outcome={self._outcomes} [{state}]>"


class RecipeTransformer(BaseEstimator, TransformerMixin):
    """Adapts a Recipe to the scikit-learn transformer API.

    Lets a recipe live inside a native ``sklearn.Pipeline``; ``fit`` preps on
    the training frame, ``transform`` returns the baked design matrix.
    """

    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self._prepped = None

    def fit(self, X: pd.DataFrame, y=None):
        data = X.copy()
        if y is not None and self.recipe._outcomes:
            data[self.recipe._outcomes[0]] = list(y)
        self._prepped = self.recipe.prep(data)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rec = self._prepped or self.recipe
        return rec.bake_predictors(X)
