"""Survival (censored) metrics: concordance, Brier, and time-dependent AUC.

Two entry points:

* :func:`concordance_survival` -- a tidy, standalone metric: pass a frame with
  true ``time`` / ``event`` and a risk ``estimate`` column, get Harrell's C.
* :func:`survival_metrics` -- given a fitted survival workflow plus train and
  test frames, compute the standard bundle (Harrell + Uno concordance,
  integrated Brier score, mean time-dependent AUC) as one tidy table. It needs
  the training data because censoring-aware metrics estimate the censoring
  distribution from it.

All wrap scikit-survival, so ``pip install "pytidymodels[survival]"`` is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..parsnip.survival_engine import _check_event
from ._core import tidy_result


def _require_sksurv_metrics():
    try:
        from sksurv import metrics, util  # noqa: F401
    except (ImportError, OSError) as e:
        raise ImportError(
            "Survival metrics need scikit-survival. Install with:  "
            "pip install 'pytidymodels[survival]'") from e
    from sksurv import metrics, util
    return metrics, util


def concordance_survival(data: pd.DataFrame, time: str, event: str,
                         estimate: str) -> pd.DataFrame:
    """Harrell's concordance index (higher is better) as a tidy one-row frame.

    ``estimate`` is a risk score where larger = higher hazard / earlier event.
    """
    metrics, _ = _require_sksurv_metrics()
    ev = _check_event(data[event]).astype(bool)
    c = metrics.concordance_index_censored(
        ev, np.asarray(data[time], float), np.asarray(data[estimate], float))[0]
    return tidy_result("concordance", float(c), estimator="survival")


def _valid_times(train_time, test_time, test_event, times):
    if times is not None:
        return np.asarray(times, dtype=float)
    ev_t = test_time[test_event == 1]
    lo = np.quantile(ev_t, 0.10)
    hi = min(np.quantile(ev_t, 0.90),
             float(train_time.max()) * 0.999, float(test_time.max()) * 0.999)
    return np.linspace(lo, hi, 10)


def survival_metrics(workflow, train: pd.DataFrame, test: pd.DataFrame,
                     times=None) -> pd.DataFrame:
    """Standard survival metric bundle for a fitted workflow, as a tidy frame."""
    metrics, util = _require_sksurv_metrics()
    model = workflow.extract_fit_parsnip()
    if getattr(model, "mode", None) != "survival":
        raise ValueError("survival_metrics expects a fitted censored/survival model.")
    adapter = model.estimator
    rec = workflow.extract_recipe()
    tcol, ecol = rec._outcomes[0], rec._outcomes[1]

    tr_t = np.asarray(train[tcol], float); tr_e = _check_event(train[ecol])
    te_t = np.asarray(test[tcol], float);  te_e = _check_event(test[ecol])
    Xte = rec.bake_predictors(test)
    risk = np.asarray(adapter.risk(Xte), dtype=float)

    y_tr = util.Surv.from_arrays(tr_e.astype(bool), tr_t)
    y_te = util.Surv.from_arrays(te_e.astype(bool), te_t)
    tgrid = _valid_times(tr_t, te_t, te_e, times)

    rows = [("concordance",
             float(metrics.concordance_index_censored(te_e.astype(bool), te_t, risk)[0]))]
    try:
        rows.append(("concordance_ipcw",
                     float(metrics.concordance_index_ipcw(y_tr, y_te, risk,
                                                          tau=float(tgrid[-1]))[0])))
    except Exception:
        pass
    try:
        S = adapter.survival_function(Xte, tgrid)
        rows.append(("integrated_brier_score",
                     float(metrics.integrated_brier_score(y_tr, y_te, S, tgrid))))
    except Exception:
        pass
    try:
        _, mean_auc = metrics.cumulative_dynamic_auc(y_tr, y_te, risk, tgrid)
        rows.append(("mean_auc", float(mean_auc)))
    except Exception:
        pass
    return pd.DataFrame([{"metric": m, "estimator": "survival", "estimate": v}
                         for m, v in rows])
