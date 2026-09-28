"""Preprocessing steps.

Each step is prep (learn from training data, then rewrite the schema) then
bake (apply). Steps store their learned parameters so the same transform
replays identically on new data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..hardhat import roles
from ..hardhat.base import Step
from ..hardhat.schema import NUMERIC, ColumnInfo
from .selectors import Selector, to_selector


class _SelectorStep(Step):
    def __init__(self, selector):
        self.selector: Selector = to_selector(selector)
        self.columns_: list[str] = []


class step_impute_median(_SelectorStep):
    """Fill missing numeric values with the training median."""

    def prep(self, data, info):
        self.columns_ = self.selector.resolve(info)
        self.medians_ = {c: float(data[c].median()) for c in self.columns_}

    def bake(self, data):
        data = data.copy()
        for c, m in self.medians_.items():
            if c in data:
                data[c] = data[c].fillna(m)
        return data


class step_impute_mode(_SelectorStep):
    """Fill missing nominal values with the training mode."""

    def prep(self, data, info):
        self.columns_ = self.selector.resolve(info)
        self.modes_ = {}
        for c in self.columns_:
            m = data[c].mode(dropna=True)
            self.modes_[c] = m.iloc[0] if len(m) else "missing"

    def bake(self, data):
        data = data.copy()
        for c, m in self.modes_.items():
            if c in data:
                data[c] = data[c].fillna(m)
        return data


class step_log(_SelectorStep):
    """Log-transform selected columns."""

    def __init__(self, selector, base: float = np.e, offset: float = 0.0):
        super().__init__(selector)
        self.base = base
        self.offset = offset

    def prep(self, data, info):
        self.columns_ = self.selector.resolve(info)

    def bake(self, data):
        data = data.copy()
        denom = np.log(self.base)
        for c in self.columns_:
            if c in data:
                data[c] = np.log(data[c] + self.offset) / denom
        return data


class step_other(_SelectorStep):
    """Lump infrequent nominal levels into a single ``"other"`` category."""

    def __init__(self, selector, threshold: float = 0.05, other: str = "other"):
        super().__init__(selector)
        self.threshold = threshold
        self.other = other

    def prep(self, data, info):
        self.columns_ = self.selector.resolve(info)
        self.keep_: dict[str, set] = {}
        for c in self.columns_:
            freq = data[c].value_counts(normalize=True)
            self.keep_[c] = set(freq[freq >= self.threshold].index)

    def bake(self, data):
        data = data.copy()
        for c, keep in self.keep_.items():
            if c in data:
                data[c] = data[c].where(data[c].isin(keep), self.other)
        return data


class step_dummy(_SelectorStep):
    """One-hot encode nominal predictors into numeric indicator columns."""

    def __init__(self, selector, drop_first: bool = True):
        super().__init__(selector)
        self.drop_first = drop_first

    def prep(self, data, info):
        self.columns_ = self.selector.resolve(info)
        self.levels_: dict[str, list] = {}
        for c in self.columns_:
            levs = sorted(map(str, pd.Series(data[c].dropna().unique())))
            if self.drop_first and levs:
                levs = levs[1:]
            self.levels_[c] = levs
            # rewrite schema: drop the nominal column, add numeric dummies
            info.drop(c)
            for lev in levs:
                info.add(f"{c}_{lev}", roles.PREDICTOR, NUMERIC)

    def bake(self, data):
        data = data.copy()
        for c, levs in self.levels_.items():
            if c not in data:
                continue
            s = data[c].astype("string")
            for lev in levs:
                data[f"{c}_{lev}"] = (s == lev).astype(float)
            data = data.drop(columns=[c])
        return data


class step_normalize(_SelectorStep):
    """Center and scale numeric columns to zero mean, unit variance."""

    def prep(self, data, info):
        self.columns_ = self.selector.resolve(info)
        self.mean_ = {}
        self.sd_ = {}
        for c in self.columns_:
            self.mean_[c] = float(data[c].mean())
            sd = float(data[c].std(ddof=0))
            self.sd_[c] = sd if sd > 0 else 1.0  # guard zero-variance columns

    def bake(self, data):
        data = data.copy()
        for c in self.columns_:
            if c in data:
                data[c] = (data[c] - self.mean_[c]) / self.sd_[c]
        return data


class step_zv(_SelectorStep):
    """Remove zero-variance columns learned from the training data."""

    def prep(self, data, info):
        cand = self.selector.resolve(info)
        self.columns_ = [c for c in cand if data[c].nunique(dropna=False) <= 1]
        for c in self.columns_:
            info.drop(c)

    def bake(self, data):
        return data.drop(columns=[c for c in self.columns_ if c in data])
