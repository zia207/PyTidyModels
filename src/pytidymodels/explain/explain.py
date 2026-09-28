"""Model interpretation via SHAP.

``explain(workflow, data)`` computes SHAP values for a fitted workflow on the
**baked** design matrix (the columns the model actually sees, after the
recipe), and returns an :class:`Explanation` with tidy accessors:
per-observation values, a mean-|SHAP| importance table, and an optional
summary plot.

Tree models use the fast exact ``TreeExplainer``; anything else falls back to
a model-agnostic explainer over a background sample. For classification, values
explain the probability of the positive (last, sorted) class -- matching the
``roc_auc`` convention used elsewhere.

SHAP is an optional dependency: ``pip install "pytidymodels[shap]"``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _require_shap():
    try:
        import shap  # noqa: F401
    except (ImportError, OSError) as e:
        raise ImportError(
            "Interpretation uses SHAP, which isn't installed. Install with:  "
            "pip install 'pytidymodels[shap]'"
        ) from e
    import shap
    return shap


class Explanation:
    """SHAP values for one fitted model, with tidy accessors."""

    def __init__(self, values, base_value, features, X):
        self.values = np.asarray(values)
        self.base_value = float(base_value)
        self.features = list(features)
        self.X = X

    def importance(self) -> pd.DataFrame:
        """Global importance = mean |SHAP| per feature, most important first."""
        imp = np.abs(self.values).mean(axis=0)
        return (pd.DataFrame({"feature": self.features, "importance": imp})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))

    def to_frame(self, long: bool = True) -> pd.DataFrame:
        """SHAP values as a tidy frame (long: row/feature/shap_value)."""
        wide = pd.DataFrame(self.values, columns=self.features)
        if not long:
            return wide
        wide.insert(0, "row", range(len(wide)))
        return wide.melt(id_vars="row", var_name="feature",
                         value_name="shap_value")

    def summary_plot(self, path: Optional[str] = None, max_display: int = 15):
        """A SHAP beeswarm summary; saves to ``path`` (PNG) or returns the fig."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap = _require_shap()
        shap.summary_plot(self.values, self.X, feature_names=self.features,
                          show=False, max_display=max_display)
        fig = plt.gcf()
        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            return path
        return fig

    def __repr__(self) -> str:
        return (f"<Explanation n={self.values.shape[0]} "
                f"features={len(self.features)} base={self.base_value:.4g}>")


def _positive_index(estimator) -> int:
    classes = list(getattr(estimator, "classes_", []))
    if not classes:
        return -1
    return classes.index(sorted(classes)[-1])


def explain(workflow, data, *, background=None, n_background: int = 100,
            method: str = "auto", seed: Optional[int] = None) -> Explanation:
    """Compute SHAP values for a fitted ``workflow`` on ``data``.

    ``method`` is 'auto' (TreeExplainer for tree models, else model-agnostic),
    'tree', 'permutation', or 'kernel'. ``background`` (raw, un-baked) supplies
    the reference distribution for non-tree explainers; a sample of size
    ``n_background`` is used.
    """
    shap = _require_shap()
    rec = workflow.extract_recipe()
    model = workflow.extract_fit_parsnip()
    est = model.estimator
    mode = model.mode

    X = rec.bake_predictors(data)

    # tree-based fast path
    if method in ("auto", "tree"):
        try:
            explainer = shap.TreeExplainer(est)
            sv = explainer(X)
            values = np.asarray(sv.values)
            base = np.asarray(sv.base_values)
            if values.ndim == 3:                 # (n, features, classes)
                values = values[:, :, -1]        # positive class
                base = base[:, -1] if base.ndim == 2 else base
            return Explanation(values, float(np.mean(base)), list(X.columns), X)
        except Exception:
            if method == "tree":
                raise

    # model-agnostic path
    bg_src = background if background is not None else data
    Xbg = rec.bake_predictors(bg_src)
    if len(Xbg) > n_background:
        Xbg = Xbg.sample(n=n_background, random_state=seed)

    if mode == "classification":
        pos = _positive_index(est)
        def f(d):
            return est.predict_proba(d)[:, pos]
    else:
        def f(d):
            return est.predict(d)

    if method == "kernel":
        expl = shap.KernelExplainer(f, np.asarray(Xbg))
        vals = np.asarray(expl.shap_values(np.asarray(X)))
        base = float(np.mean(expl.expected_value))
        return Explanation(vals, base, list(X.columns), X)

    expl = shap.Explainer(f, Xbg)
    sv = expl(X)
    return Explanation(np.asarray(sv.values), float(np.mean(sv.base_values)),
                       list(X.columns), X)


def vip(workflow, data, **kwargs) -> pd.DataFrame:
    """Convenience: the SHAP importance table for ``workflow`` on ``data``."""
    return explain(workflow, data, **kwargs).importance()
