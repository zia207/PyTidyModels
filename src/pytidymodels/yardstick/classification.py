"""Classification metrics. Class-based metrics take a predicted-class column;
probability-based metrics (roc_auc) take a probability column."""
from __future__ import annotations

import pandas as pd
from sklearn import metrics as skm

from ._core import metric, tidy_result


@metric(name="accuracy", kind="class", direction="max")
def accuracy(data: pd.DataFrame, truth: str, estimate: str) -> pd.DataFrame:
    return tidy_result("accuracy",
                       float(skm.accuracy_score(data[truth], data[estimate])))


@metric(name="f_meas", kind="class", direction="max")
def f_meas(data: pd.DataFrame, truth: str, estimate: str) -> pd.DataFrame:
    val = float(skm.f1_score(data[truth], data[estimate],
                             average="binary" if data[truth].nunique() == 2
                             else "macro"))
    return tidy_result("f_meas", val, estimator="binary")


@metric(name="sensitivity", kind="class", direction="max")
def sensitivity(data: pd.DataFrame, truth: str, estimate: str) -> pd.DataFrame:
    val = float(skm.recall_score(data[truth], data[estimate],
                                 average="binary" if data[truth].nunique() == 2
                                 else "macro"))
    return tidy_result("sensitivity", val, estimator="binary")


@metric(name="specificity", kind="class", direction="max")
def specificity(data: pd.DataFrame, truth: str, estimate: str) -> pd.DataFrame:
    labels = sorted(data[truth].unique())
    if len(labels) != 2:
        raise ValueError("specificity is defined for binary outcomes here.")
    neg, pos = labels[0], labels[1]
    tn = int(((data[truth] == neg) & (data[estimate] == neg)).sum())
    fp = int(((data[truth] == neg) & (data[estimate] == pos)).sum())
    val = tn / (tn + fp) if (tn + fp) else float("nan")
    return tidy_result("specificity", val, estimator="binary")


@metric(name="roc_auc", kind="prob", direction="max")
def roc_auc(data: pd.DataFrame, truth: str, prob: str) -> pd.DataFrame:
    """Binary ROC AUC. ``prob`` is the probability column for the positive
    (second, sorted) class."""
    labels = sorted(data[truth].unique())
    pos = labels[-1]
    y_true = (data[truth] == pos).astype(int)
    val = float(skm.roc_auc_score(y_true, data[prob]))
    return tidy_result("roc_auc", val)
