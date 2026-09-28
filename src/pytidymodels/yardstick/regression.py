"""Regression metrics. Each returns a tidy one-row frame."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from ._core import metric, tidy_result


@metric(name="rmse", kind="numeric", direction="min")
def rmse(data: pd.DataFrame, truth: str, estimate: str) -> pd.DataFrame:
    val = float(np.sqrt(skm.mean_squared_error(data[truth], data[estimate])))
    return tidy_result("rmse", val)


@metric(name="mae", kind="numeric", direction="min")
def mae(data: pd.DataFrame, truth: str, estimate: str) -> pd.DataFrame:
    val = float(skm.mean_absolute_error(data[truth], data[estimate]))
    return tidy_result("mae", val)


@metric(name="rsq", kind="numeric", direction="max")
def rsq(data: pd.DataFrame, truth: str, estimate: str) -> pd.DataFrame:
    # squared correlation, matching yardstick's definition
    r = np.corrcoef(data[truth], data[estimate])[0, 1]
    return tidy_result("rsq", float(r ** 2))


@metric(name="mape", kind="numeric", direction="min")
def mape(data: pd.DataFrame, truth: str, estimate: str) -> pd.DataFrame:
    val = float(skm.mean_absolute_percentage_error(data[truth], data[estimate]) * 100)
    return tidy_result("mape", val)
