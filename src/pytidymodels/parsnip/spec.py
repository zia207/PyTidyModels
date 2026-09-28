"""Model specifications and fitted models.

A spec declares *what* model you want; ``set_engine`` binds *how* it's
computed and ``set_mode`` says regression vs classification. Specs are
immutable -- every setter returns a new copy -- and fitting always yields
predictions as a tidy, labelled frame (``.pred`` for regression;
``.pred_class`` plus per-level probabilities for classification).

Two ways to specify a model:

* **Curated types** -- ``rand_forest()``, ``boost_tree()``, ``svm_rbf()``,
  ``mlp()``, ... with tidy, engine-neutral argument names.
* **Any sklearn estimator** -- ``sklearn_model("HistGradientBoostingRegressor")``
  or ``sklearn_model(SomeEstimatorClass, ...)`` for full coverage.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd

from ..hardhat.base import FittedModel, ModelSpec
from . import engines


class Tune:
    """Placeholder marking an argument to be resolved by the tuner (Phase 2)."""

    def __repr__(self):
        return "tune()"


def tune() -> Tune:
    return Tune()


class Spec(ModelSpec):
    def __init__(self, model_type: str, args: dict,
                 generic_estimator=None, torch=None, keras=None, h2o=None):
        self.model_type = model_type
        self.args = args
        self.generic_estimator = generic_estimator  # class/name/instance or None
        self.torch = torch                          # dict for torch_model, or None
        self.keras = keras                          # dict for keras_model, or None
        self.h2o = h2o                              # dict for h2o_automl, or None
        self.engine: Optional[str] = None
        self.engine_kwargs: dict = {}
        self.mode: Optional[str] = None

    def set_engine(self, engine: str, **kwargs) -> "Spec":
        new = copy.deepcopy(self)
        new.engine = engine
        new.engine_kwargs = kwargs
        return new

    def set_mode(self, mode: str) -> "Spec":
        new = copy.deepcopy(self)
        new.mode = mode
        return new

    def _check_resolved(self) -> None:
        unresolved = [k for k, v in self.args.items() if isinstance(v, Tune)]
        if unresolved:
            raise ValueError(
                f"Arguments still marked tune() and need finalizing: "
                f"{unresolved}. (Tuning lands in Phase 2.)")

    def fit_xy(self, X: pd.DataFrame, y) -> "FittedSpec":
        self._check_resolved()
        estimator, mode = engines.build(self)
        if mode == "survival":
            # y is a two-column frame [time, event]
            yf = y if hasattr(y, "iloc") else pd.DataFrame(y)
            time = np.asarray(yf.iloc[:, 0], dtype=float)
            event = np.asarray(yf.iloc[:, 1])
            estimator.fit(X, time, event)
            return FittedSpec(estimator, "survival", None, list(X.columns))
        estimator.fit(X, np.asarray(y))
        levels = (list(getattr(estimator, "classes_", []))
                  if mode == "classification" else None)
        return FittedSpec(estimator, mode, levels, list(X.columns))

    def __repr__(self) -> str:
        if self.h2o is not None:
            return f"<Spec h2o:automl mode={self.mode}>"
        if self.keras is not None:
            m = self.keras["module"]
            name = getattr(m, "__name__", type(m).__name__)
            return f"<Spec keras:{name} mode={self.mode}>"
        if self.torch is not None:
            m = self.torch["module"]
            name = getattr(m, "__name__", type(m).__name__)
            return f"<Spec torch:{name} mode={self.mode}>"
        if self.generic_estimator is not None:
            name = getattr(self.generic_estimator, "__name__",
                           str(self.generic_estimator))
            return f"<Spec sklearn:{name} mode={self.mode} kwargs={self.args}>"
        return (f"<Spec {self.model_type} engine={self.engine or 'sklearn'} "
                f"mode={self.mode} args={self.args}>")


class FittedSpec(FittedModel):
    def __init__(self, estimator, mode, levels, feature_names):
        self.estimator = estimator
        self.mode = mode
        self.levels = levels
        self.feature_names = feature_names

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "regression":
            preds = self.estimator.predict(X)
            return pd.DataFrame({".pred": np.asarray(preds)}, index=X.index)

        if self.mode == "survival":
            return pd.DataFrame({
                ".pred_time": np.asarray(self.estimator.median_time(X)),
                ".pred_risk": np.asarray(self.estimator.risk(X)),
            }, index=X.index)

        out = pd.DataFrame(index=X.index)
        out[".pred_class"] = self.estimator.predict(X)
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            for j, lev in enumerate(self.estimator.classes_):
                out[f".pred_{lev}"] = proba[:, j]
        return out

    def predict_survival(self, X: pd.DataFrame, times) -> pd.DataFrame:
        """Survival probability S(t|x) at each requested time (survival mode)."""
        if self.mode != "survival":
            raise ValueError("predict_survival is only for censored/survival models.")
        times = np.asarray(times, dtype=float)
        S = self.estimator.survival_function(X, times)
        cols = {f".pred_surv_{t:g}": S[:, j] for j, t in enumerate(times)}
        return pd.DataFrame(cols, index=X.index)


# ---------------------------------------------------------------------------
# curated model-type constructors
# ---------------------------------------------------------------------------

def rand_forest(mtry=None, trees=None, min_n=None) -> Spec:
    return Spec("rand_forest", {"mtry": mtry, "trees": trees, "min_n": min_n})


def extra_trees(mtry=None, trees=None, min_n=None) -> Spec:
    return Spec("extra_trees", {"mtry": mtry, "trees": trees, "min_n": min_n})


def boost_tree(trees=None, tree_depth=None, learn_rate=None, min_n=None,
               sample_size=None, mtry=None, stop_iter=None,
               loss_reduction=None) -> Spec:
    return Spec("boost_tree", {"trees": trees, "tree_depth": tree_depth,
                               "learn_rate": learn_rate, "min_n": min_n,
                               "sample_size": sample_size, "mtry": mtry,
                               "stop_iter": stop_iter,
                               "loss_reduction": loss_reduction})


def hist_boost_tree(trees=None, tree_depth=None, learn_rate=None, min_n=None,
                    stop_iter=None) -> Spec:
    return Spec("hist_boost_tree", {"trees": trees, "tree_depth": tree_depth,
                                    "learn_rate": learn_rate, "min_n": min_n,
                                    "stop_iter": stop_iter})


def decision_tree(tree_depth=None, min_n=None, cost_complexity=None) -> Spec:
    return Spec("decision_tree", {"tree_depth": tree_depth, "min_n": min_n,
                                  "cost_complexity": cost_complexity})


def bag_tree(trees=None) -> Spec:
    return Spec("bag_tree", {"trees": trees})


def adaboost(trees=None, learn_rate=None) -> Spec:
    return Spec("adaboost", {"trees": trees, "learn_rate": learn_rate})


def nearest_neighbor(neighbors=None, weight_func=None, dist_power=None) -> Spec:
    return Spec("nearest_neighbor", {"neighbors": neighbors,
                                     "weight_func": weight_func,
                                     "dist_power": dist_power})


def svm_linear(cost=None, margin=None) -> Spec:
    return Spec("svm_linear", {"cost": cost, "margin": margin})


def svm_rbf(cost=None, rbf_sigma=None, margin=None) -> Spec:
    return Spec("svm_rbf", {"cost": cost, "rbf_sigma": rbf_sigma,
                            "margin": margin})


def svm_poly(cost=None, degree=None, scale_factor=None, margin=None) -> Spec:
    return Spec("svm_poly", {"cost": cost, "degree": degree,
                             "scale_factor": scale_factor, "margin": margin})


def mlp(hidden_units=None, penalty=None, epochs=None, learn_rate=None,
        activation=None) -> Spec:
    return Spec("mlp", {"hidden_units": hidden_units, "penalty": penalty,
                        "epochs": epochs, "learn_rate": learn_rate,
                        "activation": activation})


def gaussian_process() -> Spec:
    return Spec("gaussian_process", {})


def naive_Bayes(smoothness=None) -> Spec:
    return Spec("naive_Bayes", {"smoothness": smoothness})


def discrim_linear() -> Spec:
    return Spec("discrim_linear", {})


def discrim_quad() -> Spec:
    return Spec("discrim_quad", {})


def linear_reg(penalty=None, mixture=None) -> Spec:
    return Spec("linear_reg", {"penalty": penalty, "mixture": mixture})


def logistic_reg(penalty=None, mixture=None) -> Spec:
    return Spec("logistic_reg", {"penalty": penalty, "mixture": mixture})


def multinom_reg(penalty=None, mixture=None) -> Spec:
    return Spec("multinom_reg", {"penalty": penalty, "mixture": mixture})


def poisson_reg(penalty=None) -> Spec:
    return Spec("poisson_reg", {"penalty": penalty})


# -- survival / censored regression ----------------------------------------

def proportional_hazards(penalty=None, mixture=None) -> Spec:
    """Cox proportional hazards (censored mode). Engines: lifelines, sksurv."""
    return Spec("proportional_hazards", {"penalty": penalty, "mixture": mixture})


def survival_reg(dist="weibull", penalty=None) -> Spec:
    """Parametric accelerated failure time model (censored mode; lifelines).

    ``dist``: 'weibull', 'lognormal', or 'loglogistic'.
    """
    return Spec("survival_reg", {"dist": dist, "penalty": penalty})


# ---------------------------------------------------------------------------
# universal scikit-learn access
# ---------------------------------------------------------------------------

def sklearn_model(estimator, mode: Optional[str] = None, **kwargs) -> Spec:
    """Wrap *any* scikit-learn estimator as a tidy spec.

    ``estimator`` may be a class, an instance, or a name string (as listed by
    :func:`list_sklearn_estimators`). Native scikit-learn kwargs are passed
    through unchanged. ``mode`` is inferred from the estimator when omitted.

    Examples
    --------
    >>> sklearn_model("HistGradientBoostingRegressor", max_iter=200)
    >>> from sklearn.linear_model import HuberRegressor
    >>> sklearn_model(HuberRegressor, epsilon=1.5)
    """
    if isinstance(estimator, str):
        # Resolve now so a mistyped name fails fast (with a suggestion),
        # rather than only when the workflow is fit.
        estimator, _ = engines.resolve_estimator_class(estimator)
    spec = Spec("_sklearn", dict(kwargs), generic_estimator=estimator)
    if mode is not None:
        spec.mode = mode
    return spec


def list_models() -> list[str]:
    """Curated model-type names available as tidy constructors."""
    return engines.list_curated()


def list_sklearn_estimators(kind: str = "all") -> list[str]:
    """All scikit-learn estimators reachable via sklearn_model().

    ``kind`` is 'all', 'classifier', or 'regressor'.
    """
    return engines.list_sklearn_estimators(kind)


# ---------------------------------------------------------------------------
# PyTorch access
# ---------------------------------------------------------------------------

def torch_model(module, mode: Optional[str] = None, *, epochs: int = 100,
                batch_size: int = 32, lr: float = 1e-3,
                weight_decay: float = 0.0, device=None, seed=None,
                verbose: bool = False, **kwargs) -> Spec:
    """Wrap *any* PyTorch ``nn.Module`` as a tidy spec with a training loop.

    ``module`` may be a ready ``nn.Module`` instance, an ``nn.Module``
    subclass, or a factory ``factory(in_features, out_features) -> nn.Module``.
    The estimator handles batching, the optimizer (Adam), the loss (MSE for
    regression, cross-entropy for classification), and tidy predictions, so it
    slots into the same ``Workflow`` and ``yardstick`` flow as everything else.

    ``mode`` cannot be inferred from a bare module, so pass it here or via
    ``.set_mode(...)``. Requires ``pip install "pytidymodels[torch]"``.

    Examples
    --------
    >>> torch_model(make_mlp_factory([64, 32]), mode="regression", epochs=200)
    >>> import torch.nn as nn
    >>> torch_model(nn.Sequential(nn.Linear(10, 1)), mode="regression")
    """
    cfg = dict(epochs=epochs, batch_size=batch_size, lr=lr,
               weight_decay=weight_decay, device=device, seed=seed,
               verbose=verbose, **kwargs)
    spec = Spec("_torch", {}, torch={"module": module, "kwargs": cfg})
    if mode is not None:
        spec.mode = mode
    return spec


def list_torch_models() -> list[str]:
    """torchvision architectures, if torchvision is installed (else empty).

    These are image models -- use them via ``torch_model`` with image data,
    not the tabular recipe/workflow pipeline.
    """
    from .torch_engine import list_torch_models as _ltm
    return _ltm()


# ---------------------------------------------------------------------------
# Keras access
# ---------------------------------------------------------------------------

def keras_model(model, mode: Optional[str] = None, *, epochs: int = 100,
                batch_size: int = 32, lr: float = 1e-3,
                weight_decay: float = 0.0, seed=None, verbose: int = 0,
                **kwargs) -> Spec:
    """Wrap *any* Keras model as a tidy spec with a training loop.

    ``model`` may be a ready ``keras.Model``, a ``keras.Model`` subclass, or a
    factory ``factory(in_features, out_features) -> keras.Model``. An
    already-compiled model keeps its loss/optimizer; an uncompiled one is
    compiled with a mode-appropriate loss. Slots into the same ``Workflow`` and
    ``yardstick`` flow as everything else.

    ``mode`` can't be inferred from a bare model, so pass it here or via
    ``.set_mode(...)``. Requires ``pip install "pytidymodels[keras]"``.

    Examples
    --------
    >>> keras_model(make_keras_mlp_factory([64, 32], mode="regression"),
    ...             mode="regression", epochs=200)
    >>> import keras
    >>> m = keras.Sequential([keras.layers.Dense(1)])
    >>> keras_model(m, mode="regression")
    """
    cfg = dict(epochs=epochs, batch_size=batch_size, lr=lr,
               weight_decay=weight_decay, seed=seed, verbose=verbose, **kwargs)
    spec = Spec("_keras", {}, keras={"module": model, "kwargs": cfg})
    if mode is not None:
        spec.mode = mode
    return spec


def list_keras_applications() -> list[str]:
    """Prebuilt architectures in ``keras.applications`` (else empty).

    Image models -- use via ``keras_model`` with image data, not the tabular
    pipeline.
    """
    from .keras_engine import list_keras_applications as _lka
    return _lka()


# ---------------------------------------------------------------------------
# H2O access
# ---------------------------------------------------------------------------

def h2o_automl(mode: Optional[str] = None, *, max_models=None,
               max_runtime_secs=None, sort_metric=None, seed=None,
               **kwargs) -> Spec:
    """Run H2O AutoML and predict with the leader model.

    ``mode`` is 'regression' or 'classification' (pass here or via
    ``.set_mode``). Extra keyword args pass straight to ``H2OAutoML`` (e.g.
    ``nfolds``, ``exclude_algos``, ``include_algos``, ``balance_classes``).
    The fitted spec exposes the leaderboard via
    ``workflow.extract_fit_engine().leaderboard()``.
    Requires ``pip install "pytidymodels[h2o]"`` and a Java runtime.
    """
    cfg = _drop_none_kwargs(dict(max_models=max_models,
                                 max_runtime_secs=max_runtime_secs,
                                 sort_metric=sort_metric, seed=seed))
    cfg.update(kwargs)
    spec = Spec("_h2o_automl", {}, h2o={"kwargs": cfg})
    if mode is not None:
        spec.mode = mode
    return spec


def _drop_none_kwargs(d):
    return {k: v for k, v in d.items() if v is not None}


def h2o_init(**kwargs):
    """Start/configure the H2O cluster (e.g. ``max_mem_size``, ``nthreads``)."""
    from .h2o_engine import h2o_init as _init
    return _init(**kwargs)


def h2o_shutdown():
    """Shut down the H2O cluster."""
    from .h2o_engine import h2o_shutdown as _shut
    return _shut()


def list_h2o_models() -> list[str]:
    """H2O algorithms reachable through the h2o engine."""
    from .h2o_engine import list_h2o_models as _l
    return _l()
