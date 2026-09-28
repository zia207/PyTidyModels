"""Hyperparameter tuning over resamples.

``tune_grid`` fits each candidate (a grid row) across each resample fold,
scores the held-out assessment set, and returns tidy, aggregated results. The
verbs ``collect_metrics`` / ``show_best`` / ``select_best`` read those results;
``finalize_workflow`` bakes the chosen values back into a workflow, and
``last_fit`` does the final train-fit + test-evaluation in one step.

Candidates come from a spec whose arguments are marked ``parsnip.tune()`` (or
simply from a grid whose columns name the spec arguments to override).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..workflows.workflow import Workflow
from ..yardstick import MetricSet, metric_direction, metric_set


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_metric_set(metrics) -> MetricSet:
    if isinstance(metrics, MetricSet):
        return metrics
    if callable(metrics):
        return metric_set(metrics)
    return metric_set(*metrics)


def _py(v):
    """Convert numpy scalars to native Python so sklearn/torch accept them."""
    return v.item() if hasattr(v, "item") else v


def _apply_params(spec, params: dict):
    """Return a copy of ``spec`` with ``params`` substituted into its args."""
    s = copy.deepcopy(spec)
    for k, v in params.items():
        v = _py(v)
        if k in s.args:
            s.args[k] = v
        else:
            s.engine_kwargs[k] = v
    return s


def finalize_workflow(workflow: Workflow, params: dict) -> Workflow:
    """A new (unfitted) workflow with ``params`` fixed into its model spec."""
    new_spec = _apply_params(workflow._model, params)
    return Workflow().add_recipe(workflow._recipe).add_model(new_spec)


def _score(wf: Workflow, assessment: pd.DataFrame, outcome: str,
           metrics: MetricSet) -> pd.DataFrame:
    aug = wf.augment(assessment)
    model = wf.extract_fit_parsnip()
    if model.mode == "classification":
        pos = sorted(model.levels)[-1]
        return metrics(aug, truth=outcome, estimate=".pred_class",
                       prob=f".pred_{pos}")
    return metrics(aug, truth=outcome, estimate=".pred")


# ---------------------------------------------------------------------------
# results container + verbs
# ---------------------------------------------------------------------------

class TuneResults:
    """Aggregated tuning results plus the raw per-fold scores."""

    def __init__(self, summary: pd.DataFrame, raw: pd.DataFrame,
                 param_cols: list[str]):
        self.summary = summary
        self.raw = raw
        self.param_cols = param_cols

    def __repr__(self) -> str:
        n_cand = self.summary.drop_duplicates(self.param_cols).shape[0] \
            if self.param_cols else len(self.summary)
        mets = sorted(self.summary["metric"].unique())
        return f"<TuneResults candidates={n_cand} metrics={mets}>"


def collect_metrics(results: TuneResults) -> pd.DataFrame:
    """The tidy per-candidate, per-metric summary (mean / n / std_err)."""
    return results.summary


def show_best(results: TuneResults, metric: str, n: int = 5) -> pd.DataFrame:
    """Top ``n`` candidates for ``metric``, best first."""
    df = results.summary[results.summary["metric"] == metric].copy()
    ascending = metric_direction(metric) == "min"
    return df.sort_values("mean", ascending=ascending).head(n).reset_index(drop=True)


def select_best(results: TuneResults, metric: str) -> dict:
    """The parameter values of the single best candidate for ``metric``."""
    best = show_best(results, metric, n=1)
    if best.empty:
        raise ValueError(f"No results for metric {metric!r}.")
    row = best.iloc[0]
    return {c: _py(row[c]) for c in results.param_cols}


def select_by_one_std_err(results: TuneResults, metric: str,
                          *order_by: str) -> dict:
    """Simplest candidate within one standard error of the best.

    "Simplest" is defined by ``order_by`` columns (ascending) -- e.g. fewer
    ``trees`` or larger ``min_n``. Falls back to the plain best if no
    ordering is given.
    """
    df = results.summary[results.summary["metric"] == metric].copy()
    if df.empty:
        raise ValueError(f"No results for metric {metric!r}.")
    ascending = metric_direction(metric) == "min"
    df = df.sort_values("mean", ascending=ascending)
    best = df.iloc[0]
    se = 0.0 if pd.isna(best["std_err"]) else best["std_err"]
    if ascending:
        within = df[df["mean"] <= best["mean"] + se]
    else:
        within = df[df["mean"] >= best["mean"] - se]
    if order_by:
        within = within.sort_values(list(order_by))
    row = within.iloc[0]
    return {c: _py(row[c]) for c in results.param_cols}


# ---------------------------------------------------------------------------
# tune_grid
# ---------------------------------------------------------------------------

def tune_grid(workflow: Workflow, resamples, grid: pd.DataFrame,
              metrics) -> TuneResults:
    """Evaluate every ``grid`` candidate across every fold in ``resamples``.

    ``workflow`` bundles the recipe and a model spec; ``grid`` columns name the
    spec arguments to set. Each fold's recipe is prepped on its own analysis
    set, so there's no leakage into the assessment scores.
    """
    metrics = _as_metric_set(metrics)
    outcome = workflow._recipe._outcomes[0]
    param_cols = list(grid.columns)

    rows = []
    for params in grid.to_dict(orient="records"):
        for fold in resamples:
            wf = finalize_workflow(workflow, params).fit(fold.analysis())
            scored = _score(wf, fold.assessment(), outcome, metrics)
            for _, m in scored.iterrows():
                rows.append({**{k: _py(v) for k, v in params.items()},
                             "id": fold.id, "metric": m["metric"],
                             "estimate": m["estimate"]})

    raw = pd.DataFrame(rows)
    grp = raw.groupby(param_cols + ["metric"])["estimate"]
    summary = grp.agg(mean="mean", n="size", std="std").reset_index()
    summary["std_err"] = summary["std"] / np.sqrt(summary["n"])
    summary = summary.drop(columns="std")
    return TuneResults(summary, raw, param_cols)


# ---------------------------------------------------------------------------
# last_fit
# ---------------------------------------------------------------------------

@dataclass
class LastFit:
    workflow: Workflow
    metrics: pd.DataFrame
    predictions: pd.DataFrame


def last_fit(workflow: Workflow, split, metrics) -> LastFit:
    """Fit ``workflow`` on the training split and evaluate on the test split.

    ``workflow`` should already be finalized (no ``tune()`` left). Returns the
    fitted workflow, the test-set metrics, and the test-set predictions.
    """
    metrics = _as_metric_set(metrics)
    outcome = workflow._recipe._outcomes[0]
    wf = finalize_workflow(workflow, {}).fit(split.training())
    test = split.testing()
    scored = _score(wf, test, outcome, metrics)
    preds = wf.augment(test)
    return LastFit(workflow=wf, metrics=scored, predictions=preds)
