"""Core metric machinery: tidy result shaping, the metric decorator, metric_set.

Every metric carries a ``kind`` -- ``numeric`` or ``class`` (needs a predicted
column) or ``prob`` (needs a probability column) -- so a ``metric_set`` can
mix, say, ``accuracy`` and ``roc_auc`` and route each to the right input.
"""
from __future__ import annotations

import pandas as pd


def tidy_result(name: str, value: float, estimator: str = "standard") -> pd.DataFrame:
    return pd.DataFrame([{"metric": name, "estimator": estimator,
                          "estimate": value}])


_DIRECTIONS: dict[str, str] = {}


def metric(name: str, kind: str, direction: str = "max"):
    """Tag a metric with its name, input kind, and optimization direction.

    ``direction`` is 'max' (higher is better, e.g. accuracy) or 'min' (lower is
    better, e.g. rmse) -- used by tuning to rank candidates.
    """
    def deco(fn):
        fn.metric_name = name
        fn.metric_kind = kind  # "numeric" | "class" | "prob"
        fn.metric_direction = direction
        _DIRECTIONS[name] = direction
        return fn
    return deco


def metric_direction(name: str) -> str:
    """'min' or 'max' for a metric name (defaults to 'max' if unknown)."""
    return _DIRECTIONS.get(name, "max")


class MetricSet:
    """Apply several metrics at once, returning one stacked tidy frame."""

    def __init__(self, metrics):
        self.metrics = list(metrics)

    def __call__(self, data: pd.DataFrame, truth: str,
                 estimate: str | None = None, prob: str | None = None) -> pd.DataFrame:
        rows = []
        for m in self.metrics:
            kind = getattr(m, "metric_kind", "numeric")
            if kind == "prob":
                if prob is None:
                    raise ValueError(
                        f"Metric {m.metric_name!r} needs a probability column "
                        f"(pass prob=...).")
                rows.append(m(data, truth, prob))
            else:
                if estimate is None:
                    raise ValueError(
                        f"Metric {m.metric_name!r} needs an estimate column "
                        f"(pass estimate=...).")
                rows.append(m(data, truth, estimate))
        return pd.concat(rows, ignore_index=True)


def metric_set(*metrics) -> MetricSet:
    return MetricSet(metrics)
