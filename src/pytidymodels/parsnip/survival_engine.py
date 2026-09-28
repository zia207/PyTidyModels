"""Optional survival-analysis engines: lifelines and scikit-survival.

This is the ``censored`` piece of the ecosystem. Survival models use a
two-column outcome ``(time, event)`` where ``event`` is 1 for an observed
event and 0 for right-censoring, and the model mode is
``"censored regression"`` (aliases: ``"survival"``, ``"censored"``).

Two backends sit behind one adapter interface so the rest of the package
(workflows, tidy predictions, metrics) doesn't care which library fits:

* **lifelines** -- ``CoxPHFitter`` (proportional_hazards) and the parametric
  AFT fitters (survival_reg: Weibull / log-normal / log-logistic).
* **scikit-survival** -- ``CoxPHSurvivalAnalysis`` and the tree ensembles
  (``RandomSurvivalForest``, ``GradientBoostingSurvivalAnalysis``,
  ``SurvivalTree``) for rand_forest / boost_tree / decision_tree in
  censored mode.

Every adapter exposes: ``fit(X, time, event)``, ``risk(X)`` (higher = higher
hazard, for concordance), ``survival_function(X, times)`` (an n x len(times)
matrix of S(t|x)), and ``median_time(X)``. Both are optional dependencies,
imported lazily. Install with ``pip install "pytidymodels[survival]"``.
"""
from __future__ import annotations

import numpy as np

SURVIVAL_MODES = {"survival", "censored", "censored regression"}


def is_survival(mode) -> bool:
    return mode in SURVIVAL_MODES


def _require(pkg: str):
    try:
        return __import__(pkg)
    except (ImportError, OSError) as e:
        raise ImportError(
            f"Survival models need {pkg}. Install with:  "
            "pip install 'pytidymodels[survival]'") from e


def _check_event(event: np.ndarray) -> np.ndarray:
    ev = np.asarray(event)
    uniq = set(np.unique(ev).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(
            "Survival 'event' must be coded 0 (censored) / 1 (event); got "
            f"values {sorted(uniq)}. Recode before fitting (e.g. status==2 -> 1).")
    return ev.astype(int)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

class _Adapter:
    train_time: np.ndarray
    train_event: np.ndarray

    def risk(self, X):                       # pragma: no cover - interface
        raise NotImplementedError

    def survival_function(self, X, times):   # pragma: no cover
        raise NotImplementedError

    def median_time(self, X):                # pragma: no cover
        raise NotImplementedError


class LifelinesAdapter(_Adapter):
    """Wraps a lifelines CoxPHFitter or an AFT fitter."""

    def __init__(self, fitter, kind: str):
        self.fitter = fitter
        self.kind = kind                     # "cox" | "aft"

    def fit(self, X, time, event):
        import pandas as pd
        event = _check_event(event)
        df = pd.DataFrame(X).copy()
        df["__T__"] = np.asarray(time, dtype=float)
        df["__E__"] = event
        self.fitter.fit(df, duration_col="__T__", event_col="__E__")
        self.features_ = list(pd.DataFrame(X).columns)
        self.train_time = np.asarray(time, dtype=float)
        self.train_event = event
        self._grid = np.unique(self.train_time[self.train_event == 1])
        return self

    def _frame(self, X):
        import pandas as pd
        return pd.DataFrame(X, columns=self.features_)

    def survival_function(self, X, times):
        sf = self.fitter.predict_survival_function(self._frame(X), times=times)
        return sf.to_numpy().T                # (n_subjects, n_times)

    def risk(self, X):
        if self.kind == "cox":
            return np.asarray(self.fitter.predict_partial_hazard(self._frame(X)))
        # AFT: rank by negative restricted mean survival time (finite, robust)
        S = self.survival_function(X, self._grid)
        rmst = np.trapezoid(S, self._grid, axis=1)
        return -rmst

    def median_time(self, X):
        med = np.asarray(self.fitter.predict_median(self._frame(X)), dtype=float)
        return med


class SksurvAdapter(_Adapter):
    """Wraps a scikit-survival estimator."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, time, event):
        from sksurv.util import Surv
        event = _check_event(event)
        y = Surv.from_arrays(event=event.astype(bool), time=np.asarray(time, float))
        self.model.fit(np.asarray(X, dtype=float), y)
        self.train_time = np.asarray(time, dtype=float)
        self.train_event = event
        self._grid = np.asarray(
            getattr(self.model, "event_times_",
                    np.unique(self.train_time[self.train_event == 1])), dtype=float)
        return self

    def risk(self, X):
        return np.asarray(self.model.predict(np.asarray(X, dtype=float)))

    def survival_function(self, X, times):
        fns = self.model.predict_survival_function(np.asarray(X, dtype=float))
        return np.vstack([[float(fn(t)) for t in times] for fn in fns])

    def median_time(self, X):
        grid = self._grid
        S = self.survival_function(X, grid)
        out = []
        for row in S:
            below = np.where(row <= 0.5)[0]
            out.append(float(grid[below[0]]) if below.size else np.inf)
        return np.array(out)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

def build_survival(spec):
    """Build a survival adapter for a spec whose mode is censored/survival."""
    mt, engine = spec.model_type, spec.engine
    a, ek = spec.args, dict(spec.engine_kwargs)

    if mt == "proportional_hazards":
        engine = engine or "lifelines"
        if engine == "lifelines":
            _require("lifelines")
            from lifelines import CoxPHFitter
            return LifelinesAdapter(
                CoxPHFitter(penalizer=a.get("penalty") or 0.0,
                            l1_ratio=a.get("mixture") or 0.0), "cox")
        if engine == "sksurv":
            _require("sksurv")
            from sksurv.linear_model import CoxPHSurvivalAnalysis
            return SksurvAdapter(
                CoxPHSurvivalAnalysis(alpha=a.get("penalty") or 0.0, **ek))
        raise ValueError(f"proportional_hazards engine {engine!r} not supported "
                         "(use 'lifelines' or 'sksurv').")

    if mt == "survival_reg":
        _require("lifelines")
        from lifelines import (LogLogisticAFTFitter, LogNormalAFTFitter,
                               WeibullAFTFitter)
        dist = (a.get("dist") or "weibull").lower()
        cls = {"weibull": WeibullAFTFitter, "lognormal": LogNormalAFTFitter,
               "loglogistic": LogLogisticAFTFitter}.get(dist)
        if cls is None:
            raise ValueError(f"survival_reg dist {dist!r} not supported "
                             "(weibull | lognormal | loglogistic).")
        return LifelinesAdapter(cls(penalizer=a.get("penalty") or 0.0), "aft")

    if mt in ("rand_forest", "boost_tree", "decision_tree"):
        engine = engine or "sksurv"
        if engine != "sksurv":
            raise ValueError(f"{mt} in censored mode needs engine 'sksurv'.")
        _require("sksurv")
        return SksurvAdapter(_sksurv_tree(mt, a, ek))

    raise ValueError(f"Model type {mt!r} has no survival engine.")


def _sksurv_tree(mt, a, ek):
    def drop_none(d):
        return {k: v for k, v in d.items() if v is not None}
    if mt == "rand_forest":
        from sksurv.ensemble import RandomSurvivalForest
        return RandomSurvivalForest(**drop_none({
            "n_estimators": a.get("trees"), "min_samples_leaf": a.get("min_n"),
            "max_features": a.get("mtry")}), **ek)
    if mt == "boost_tree":
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        return GradientBoostingSurvivalAnalysis(**drop_none({
            "n_estimators": a.get("trees"), "learning_rate": a.get("learn_rate"),
            "max_depth": a.get("tree_depth")}), **ek)
    from sksurv.tree import SurvivalTree
    return SurvivalTree(**drop_none({
        "max_depth": a.get("tree_depth"), "min_samples_split": a.get("min_n")}), **ek)
