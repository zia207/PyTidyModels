"""Data splitting and resampling.

Returns inspectable, labelled objects rather than bare index arrays, so a
split or a set of folds is something you can look at, iterate, and reason
about -- the resampling-first foundation the rest of the workflow builds on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ..pipe import PipeMixin


def _strata_labels(data: pd.DataFrame, strata: str | None, bins: int = 4):
    """Turn a (possibly continuous) strata column into class labels."""
    if strata is None:
        return None
    col = data[strata]
    if pd.api.types.is_numeric_dtype(col) and col.nunique() > bins:
        # bin a continuous outcome so stratification is meaningful
        return pd.qcut(col, q=bins, labels=False, duplicates="drop")
    return col.to_numpy()


@dataclass
class Split(PipeMixin):
    """A single train/test partition of a data frame."""

    data: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray

    def training(self) -> pd.DataFrame:
        return self.data.iloc[self.train_idx].reset_index(drop=True)

    def testing(self) -> pd.DataFrame:
        return self.data.iloc[self.test_idx].reset_index(drop=True)

    def __repr__(self) -> str:
        return (f"<Split train={len(self.train_idx)} "
                f"test={len(self.test_idx)}>")


def initial_split(data: pd.DataFrame, prop: float = 0.75,
                  strata: str | None = None, seed: int | None = None) -> Split:
    """Split ``data`` into a training/testing pair.

    ``prop`` is the training proportion. ``strata`` keeps the distribution
    of a column balanced across the two sets (a continuous column is binned
    first).
    """
    n = len(data)
    rng = np.random.default_rng(seed)
    labels = _strata_labels(data, strata)
    train_mask = np.zeros(n, dtype=bool)

    if labels is None:
        train_mask[rng.choice(n, size=int(round(prop * n)), replace=False)] = True
    else:
        labels = np.asarray(labels)
        for lev in pd.unique(labels):
            idx = np.where(labels == lev)[0]
            k = int(round(prop * len(idx)))
            train_mask[rng.choice(idx, size=k, replace=False)] = True

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]
    return Split(data.reset_index(drop=True), train_idx, test_idx)


@dataclass
class Fold:
    """One resample: an analysis set and its held-out assessment set."""

    data: pd.DataFrame
    analysis_idx: np.ndarray
    assessment_idx: np.ndarray
    id: str

    def analysis(self) -> pd.DataFrame:
        return self.data.iloc[self.analysis_idx].reset_index(drop=True)

    def assessment(self) -> pd.DataFrame:
        return self.data.iloc[self.assessment_idx].reset_index(drop=True)

    def __repr__(self) -> str:
        return (f"<Fold {self.id} analysis={len(self.analysis_idx)} "
                f"assessment={len(self.assessment_idx)}>")


class ResampleSet:
    """An iterable, tidy collection of folds."""

    def __init__(self, folds: list[Fold]):
        self._folds = folds

    def __iter__(self):
        return iter(self._folds)

    def __len__(self):
        return len(self._folds)

    def __getitem__(self, i):
        return self._folds[i]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"id": [f.id for f in self._folds],
                             "splits": list(self._folds)})

    def __repr__(self) -> str:
        return f"<ResampleSet folds={len(self._folds)}>"


def vfold_cv(data: pd.DataFrame, v: int = 10, repeats: int = 1,
             strata: str | None = None, seed: int | None = None) -> ResampleSet:
    """V-fold cross-validation, optionally stratified and repeated."""
    data = data.reset_index(drop=True)
    labels = _strata_labels(data, strata)
    folds: list[Fold] = []
    for r in range(repeats):
        rs = None if seed is None else seed + r
        if labels is None:
            splitter = KFold(n_splits=v, shuffle=True, random_state=rs)
            gen = splitter.split(data)
        else:
            splitter = StratifiedKFold(n_splits=v, shuffle=True, random_state=rs)
            gen = splitter.split(data, labels)
        for k, (ana, ass) in enumerate(gen, start=1):
            fid = f"Fold{k:02d}" if repeats == 1 else f"Repeat{r+1}.Fold{k:02d}"
            folds.append(Fold(data, ana, ass, fid))
    return ResampleSet(folds)


# -- free-function accessors (tidymodels-faithful; nice with the pipe) -------

def training(split: Split) -> pd.DataFrame:
    """Functional form of ``split.training()`` (pipe-friendly)."""
    return split.training()


def testing(split: Split) -> pd.DataFrame:
    """Functional form of ``split.testing()`` (pipe-friendly)."""
    return split.testing()


def analysis(fold: Fold) -> pd.DataFrame:
    """Functional form of ``fold.analysis()``."""
    return fold.analysis()


def assessment(fold: Fold) -> pd.DataFrame:
    """Functional form of ``fold.assessment()``."""
    return fold.assessment()


# -- free-function accessors (pipe-friendly) --------------------------------

def training(split: "Split") -> pd.DataFrame:
    """Functional form of ``split.training()`` for use in pipes."""
    return split.training()


def testing(split: "Split") -> pd.DataFrame:
    """Functional form of ``split.testing()`` for use in pipes."""
    return split.testing()


def analysis(fold: "Fold") -> pd.DataFrame:
    """Functional form of ``fold.analysis()`` for use in pipes."""
    return fold.analysis()


def assessment(fold: "Fold") -> pd.DataFrame:
    """Functional form of ``fold.assessment()`` for use in pipes."""
    return fold.assessment()
