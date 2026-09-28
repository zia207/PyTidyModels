"""Abstract contracts shared across the ecosystem.

Every recipe step is prep/bake; every fitted model predicts a tidy
frame. Stating those contracts here is what keeps the "consistent
interface" promise enforceable rather than aspirational.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .schema import ColumnInfo


class PipeMixin:
    """Provides a pandas-style ``.pipe(func, *args, **kwargs)`` for threading
    an object through a free function mid-chain: ``x.pipe(f, y)`` -> ``f(x, y)``."""

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)


class Step(ABC):
    """A single preprocessing operation in a recipe."""

    #: set at prep time to the concrete columns this step acts on
    columns_: list[str]

    @abstractmethod
    def prep(self, data: pd.DataFrame, info: ColumnInfo) -> None:
        """Learn parameters from ``data`` and update ``info`` in place."""

    @abstractmethod
    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transformation and return a new frame."""


class ModelSpec(ABC):
    """An engine-agnostic declaration of a model."""

    @abstractmethod
    def fit_xy(self, X: pd.DataFrame, y: pd.Series) -> "FittedModel":
        ...


class FittedModel(ABC):
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return predictions as a tidy, labelled frame."""
