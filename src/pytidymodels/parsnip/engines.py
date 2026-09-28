"""Engine layer: map tidy model specs onto scikit-learn estimators.

Two layers of coverage:

1. **Curated model types** -- the parsnip-style catalog (``rand_forest``,
   ``boost_tree``, ``svm_rbf``, ``nearest_neighbor``, ``mlp``, ...). Each
   exposes tidy, engine-neutral argument names and translates them to the
   right scikit-learn kwargs. Adding an engine for a type is one table row.

2. **Universal access** -- ``sklearn_model(...)`` wraps *any* estimator that
   scikit-learn exposes via ``all_estimators`` (classifiers and regressors),
   passing native kwargs straight through. This is what makes "all models of
   scikit-learn" literally true, for the long tail the curated catalog
   doesn't give a tidy name.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Optional

from sklearn.base import ClassifierMixin, RegressorMixin


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _imp(path: Optional[str]):
    """Lazily import a dotted class path, e.g. 'sklearn.svm.SVC'."""
    if path is None:
        return None
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def _translate(args: dict, mapping: dict) -> dict:
    """Rename tidy args to sklearn kwargs, dropping any left as None."""
    out = {}
    for tidy, val in args.items():
        if val is None:
            continue
        out[mapping.get(tidy, tidy)] = val
    return out


# ---------------------------------------------------------------------------
# curated model-type catalog
# ---------------------------------------------------------------------------

@dataclass
class _ModelType:
    reg: Optional[str]                     # dotted path to regressor class
    clf: Optional[str]                     # dotted path to classifier class
    mapping: dict = field(default_factory=dict)   # tidy arg -> sklearn kwarg
    reg_fixed: dict = field(default_factory=dict)  # always-on regressor kwargs
    clf_fixed: dict = field(default_factory=dict)  # always-on classifier kwargs
    coerce: Optional[Callable[[dict], dict]] = None  # pre-translate arg fixups

    def build(self, mode: str, args: dict, engine_kwargs: dict):
        if mode == "regression":
            cls, fixed = _imp(self.reg), self.reg_fixed
        elif mode == "classification":
            cls, fixed = _imp(self.clf), self.clf_fixed
        else:
            raise ValueError(f"mode must be 'regression' or 'classification', got {mode!r}")
        if cls is None:
            raise ValueError(f"This model type does not support {mode!r} mode.")
        if self.coerce is not None:
            args = self.coerce(dict(args))
        kwargs = {**fixed, **_translate(args, self.mapping), **engine_kwargs}
        return cls(**kwargs)


def _coerce_hidden_units(args: dict) -> dict:
    hu = args.get("hidden_units")
    if hu is not None and not isinstance(hu, (tuple, list)):
        args["hidden_units"] = (int(hu),)
    return args


# Registry of curated types. Custom-logic types (linear/logistic/etc.) are
# registered as callables further down and take precedence.
_CATALOG: dict[str, _ModelType] = {
    "rand_forest": _ModelType(
        reg="sklearn.ensemble.RandomForestRegressor",
        clf="sklearn.ensemble.RandomForestClassifier",
        mapping={"mtry": "max_features", "trees": "n_estimators",
                 "min_n": "min_samples_leaf"},
    ),
    "extra_trees": _ModelType(
        reg="sklearn.ensemble.ExtraTreesRegressor",
        clf="sklearn.ensemble.ExtraTreesClassifier",
        mapping={"mtry": "max_features", "trees": "n_estimators",
                 "min_n": "min_samples_leaf"},
    ),
    "boost_tree": _ModelType(
        reg="sklearn.ensemble.GradientBoostingRegressor",
        clf="sklearn.ensemble.GradientBoostingClassifier",
        mapping={"trees": "n_estimators", "tree_depth": "max_depth",
                 "learn_rate": "learning_rate", "min_n": "min_samples_leaf",
                 "sample_size": "subsample", "mtry": "max_features",
                 "stop_iter": "n_iter_no_change",
                 "loss_reduction": "min_impurity_decrease"},
    ),
    "hist_boost_tree": _ModelType(
        reg="sklearn.ensemble.HistGradientBoostingRegressor",
        clf="sklearn.ensemble.HistGradientBoostingClassifier",
        mapping={"trees": "max_iter", "tree_depth": "max_depth",
                 "learn_rate": "learning_rate", "min_n": "min_samples_leaf",
                 "stop_iter": "n_iter_no_change"},
    ),
    "decision_tree": _ModelType(
        reg="sklearn.tree.DecisionTreeRegressor",
        clf="sklearn.tree.DecisionTreeClassifier",
        mapping={"tree_depth": "max_depth", "min_n": "min_samples_split",
                 "cost_complexity": "ccp_alpha"},
    ),
    "bag_tree": _ModelType(
        reg="sklearn.ensemble.BaggingRegressor",
        clf="sklearn.ensemble.BaggingClassifier",
        mapping={"trees": "n_estimators"},
    ),
    "adaboost": _ModelType(
        reg="sklearn.ensemble.AdaBoostRegressor",
        clf="sklearn.ensemble.AdaBoostClassifier",
        mapping={"trees": "n_estimators", "learn_rate": "learning_rate"},
    ),
    "nearest_neighbor": _ModelType(
        reg="sklearn.neighbors.KNeighborsRegressor",
        clf="sklearn.neighbors.KNeighborsClassifier",
        mapping={"neighbors": "n_neighbors", "weight_func": "weights",
                 "dist_power": "p"},
    ),
    "mlp": _ModelType(
        reg="sklearn.neural_network.MLPRegressor",
        clf="sklearn.neural_network.MLPClassifier",
        mapping={"hidden_units": "hidden_layer_sizes", "penalty": "alpha",
                 "epochs": "max_iter", "learn_rate": "learning_rate_init",
                 "activation": "activation"},
        coerce=_coerce_hidden_units,
    ),
    "gaussian_process": _ModelType(
        reg="sklearn.gaussian_process.GaussianProcessRegressor",
        clf="sklearn.gaussian_process.GaussianProcessClassifier",
        mapping={},
    ),
    "naive_Bayes": _ModelType(
        reg=None, clf="sklearn.naive_bayes.GaussianNB",
        mapping={"smoothness": "var_smoothing"},
    ),
    "discrim_linear": _ModelType(
        reg=None, clf="sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
        mapping={},
    ),
    "discrim_quad": _ModelType(
        reg=None, clf="sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
        mapping={},
    ),
}


# -- custom-logic model types (class choice depends on the tidy args) --------

_CUSTOM: dict[str, Callable] = {}


def _custom(model_type):
    def deco(fn):
        _CUSTOM[model_type] = fn
        return fn
    return deco


@_custom("linear_reg")
def _linear_reg(mode, args, ek):
    if mode != "regression":
        raise ValueError("linear_reg is a regression model.")
    penalty, mixture = args.get("penalty"), args.get("mixture")
    if penalty is None:
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**ek)
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    if mixture in (None, 0):
        return Ridge(alpha=penalty, **ek)
    if mixture == 1:
        return Lasso(alpha=penalty, **ek)
    return ElasticNet(alpha=penalty, l1_ratio=mixture, **ek)


def _logistic_like(mode, args, ek, expect="classification"):
    from sklearn.linear_model import LogisticRegression
    if mode != expect:
        raise ValueError(f"This model is a {expect} model.")
    penalty, mixture = args.get("penalty"), args.get("mixture")
    kw = dict(ek)
    kw.setdefault("max_iter", 1000)
    if penalty is None:
        return LogisticRegression(**kw)
    kw["C"] = 1.0 / penalty
    kw.setdefault("l1_ratio", 0.0 if mixture is None else float(mixture))
    kw.setdefault("solver", "saga")
    return LogisticRegression(**kw)


@_custom("logistic_reg")
def _logistic_reg(mode, args, ek):
    return _logistic_like(mode, args, ek)


@_custom("multinom_reg")
def _multinom_reg(mode, args, ek):
    # Modern LogisticRegression is inherently multiclass.
    return _logistic_like(mode, args, ek)


@_custom("poisson_reg")
def _poisson_reg(mode, args, ek):
    if mode != "regression":
        raise ValueError("poisson_reg is a regression model.")
    from sklearn.linear_model import PoissonRegressor
    penalty = args.get("penalty")
    kw = dict(ek)
    if penalty is not None:
        kw["alpha"] = penalty
    return PoissonRegressor(**kw)


def _svm(mode, args, ek, kernel):
    """SVM builder. Regression -> SVR; classification -> a calibrated SVC so
    predict_proba is available without the deprecated SVC(probability=True)."""
    from sklearn.svm import SVC, SVR
    cost = args.get("cost")
    gamma = args.get("rbf_sigma") if kernel == "rbf" else args.get("scale_factor")
    degree = args.get("degree")
    margin = args.get("margin")
    common = {}
    if cost is not None:
        common["C"] = cost
    if gamma is not None:
        common["gamma"] = gamma
    if degree is not None and kernel == "poly":
        common["degree"] = degree
    if mode == "regression":
        kw = dict(common)
        if margin is not None:      # epsilon-tube applies to SVR only
            kw["epsilon"] = margin
        return SVR(kernel=kernel, **kw, **ek)
    from sklearn.calibration import CalibratedClassifierCV
    svc = SVC(kernel=kernel, **common, **ek)
    return CalibratedClassifierCV(svc, ensemble=False)


@_custom("svm_linear")
def _svm_linear(mode, args, ek):
    return _svm(mode, args, ek, "linear")


@_custom("svm_rbf")
def _svm_rbf(mode, args, ek):
    return _svm(mode, args, ek, "rbf")


@_custom("svm_poly")
def _svm_poly(mode, args, ek):
    return _svm(mode, args, ek, "poly")


# ---------------------------------------------------------------------------
# universal scikit-learn access
# ---------------------------------------------------------------------------

_SK_INDEX: Optional[dict] = None


def sklearn_index() -> dict:
    """Name -> class for every sklearn classifier and regressor."""
    global _SK_INDEX
    if _SK_INDEX is None:
        from sklearn.utils import all_estimators
        idx = {}
        for name, cls in all_estimators(type_filter=["classifier", "regressor"]):
            idx[name] = cls
        _SK_INDEX = idx
    return _SK_INDEX


def infer_mode(cls) -> Optional[str]:
    if issubclass(cls, ClassifierMixin):
        return "classification"
    if issubclass(cls, RegressorMixin):
        return "regression"
    return None


def resolve_estimator_class(estimator):
    """Accept a class, an instance, or a name string; return (class, kwargs)."""
    if isinstance(estimator, str):
        idx = sklearn_index()
        if estimator not in idx:
            import difflib
            hint = difflib.get_close_matches(estimator, idx, n=3)
            raise ValueError(
                f"Unknown scikit-learn estimator {estimator!r}. "
                + (f"Did you mean: {hint}? " if hint else "")
                + "See parsnip.list_sklearn_estimators().")
        return idx[estimator], {}
    if isinstance(estimator, type):
        return estimator, {}
    # an instance -> use its class and its current params
    return type(estimator), estimator.get_params()


def build_generic(estimator, mode: Optional[str], kwargs: dict):
    cls, base_kwargs = resolve_estimator_class(estimator)
    resolved_mode = mode or infer_mode(cls)
    if resolved_mode is None:
        raise ValueError(
            f"Could not infer regression/classification for {cls.__name__}; "
            "pass mode= explicitly.")
    merged = {**base_kwargs, **kwargs}
    return cls(**merged), resolved_mode


# ---------------------------------------------------------------------------
# unified entry point
# ---------------------------------------------------------------------------

def build(spec) -> tuple[object, str]:
    """Build the sklearn/torch estimator for a Spec; return (estimator, mode)."""
    if getattr(spec, "generic_estimator", None) is not None:
        return build_generic(spec.generic_estimator, spec.mode, spec.args)

    if getattr(spec, "torch", None) is not None:
        mode = spec.mode
        if mode is None:
            raise ValueError(
                "torch_model(...) needs a mode -- call .set_mode('regression'"
                "|'classification') or pass mode=.")
        return build_torch_generic(spec), mode

    if getattr(spec, "keras", None) is not None:
        mode = spec.mode
        if mode is None:
            raise ValueError(
                "keras_model(...) needs a mode -- call .set_mode('regression'"
                "|'classification') or pass mode=.")
        return build_keras_generic(spec), mode

    if getattr(spec, "h2o", None) is not None:
        mode = spec.mode
        if mode is None:
            raise ValueError(
                "h2o_automl(...) needs a mode -- call .set_mode('regression'"
                "|'classification') or pass mode=.")
        from .h2o_engine import build_h2o_automl
        return build_h2o_automl(spec), mode

    model_type, mode = spec.model_type, spec.mode
    if mode is None:
        raise ValueError("Call set_mode('regression'|'classification') first.")

    from .survival_engine import build_survival, is_survival
    if is_survival(mode):
        return build_survival(spec), "survival"

    if spec.engine == "torch":
        if model_type != "mlp":
            raise ValueError(
                f"engine 'torch' is currently available for mlp only, not "
                f"{model_type!r}. Use parsnip.torch_model(...) to wrap an "
                "arbitrary nn.Module.")
        return build_torch_curated(spec), mode

    if spec.engine == "keras":
        if model_type != "mlp":
            raise ValueError(
                f"engine 'keras' is currently available for mlp only, not "
                f"{model_type!r}. Use parsnip.keras_model(...) to wrap an "
                "arbitrary keras.Model.")
        return build_keras_curated(spec), mode

    if spec.engine == "h2o":
        from .h2o_engine import build_h2o
        return build_h2o(spec), mode

    if spec.engine not in (None, "sklearn"):
        raise ValueError(
            f"No engine {spec.engine!r} for curated model types -- 'sklearn' is "
            "wired for all of them, 'torch'/'keras' for mlp, and 'h2o' for "
            "linear_reg/logistic_reg/rand_forest/boost_tree/mlp. For other "
            "backends use parsnip.sklearn_model(...), torch_model(...), "
            "keras_model(...), or h2o_automl(...).")
    ek = spec.engine_kwargs
    if model_type in _CUSTOM:
        return _CUSTOM[model_type](mode, spec.args, ek), mode
    if model_type in _CATALOG:
        return _CATALOG[model_type].build(mode, spec.args, ek), mode
    raise ValueError(
        f"Unknown model type {model_type!r}. Curated types: {list_curated()}. "
        "For anything else use parsnip.sklearn_model(...).")


def build_torch_curated(spec):
    """Build a torch MLP for `mlp(...).set_engine('torch', ...)`."""
    from .torch_engine import TorchEstimator, make_mlp_factory
    a = spec.args
    ek = dict(spec.engine_kwargs)
    factory = make_mlp_factory(
        hidden_units=a.get("hidden_units") or (32,),
        activation=a.get("activation") or "relu",
        dropout=ek.pop("dropout", 0.0),
    )
    return TorchEstimator(
        factory, spec.mode,
        epochs=a.get("epochs") or 100,
        lr=a.get("learn_rate") or 1e-3,
        weight_decay=a.get("penalty") or 0.0,
        **ek,
    )


def build_torch_generic(spec):
    """Build a TorchEstimator wrapping an arbitrary nn.Module (torch_model)."""
    from .torch_engine import TorchEstimator
    t = spec.torch
    return TorchEstimator(t["module"], spec.mode, **t["kwargs"])


def build_keras_curated(spec):
    """Build a keras MLP for `mlp(...).set_engine('keras', ...)`."""
    from .keras_engine import KerasEstimator, make_keras_mlp_factory
    a = spec.args
    ek = dict(spec.engine_kwargs)
    factory = make_keras_mlp_factory(
        hidden_units=a.get("hidden_units") or (32,),
        activation=a.get("activation") or "relu",
        dropout=ek.pop("dropout", 0.0),
        mode=spec.mode,
    )
    return KerasEstimator(
        factory, spec.mode,
        epochs=a.get("epochs") or 100,
        lr=a.get("learn_rate") or 1e-3,
        weight_decay=a.get("penalty") or 0.0,
        **ek,
    )


def build_keras_generic(spec):
    """Build a KerasEstimator wrapping an arbitrary keras.Model (keras_model)."""
    from .keras_engine import KerasEstimator
    k = spec.keras
    return KerasEstimator(k["module"], spec.mode, **k["kwargs"])


def list_curated() -> list[str]:
    return sorted(set(_CATALOG) | set(_CUSTOM))


def list_sklearn_estimators(kind: str = "all") -> list[str]:
    idx = sklearn_index()
    if kind == "all":
        return sorted(idx)
    want = ClassifierMixin if kind == "classifier" else RegressorMixin
    if kind not in ("classifier", "regressor"):
        raise ValueError("kind must be 'all', 'classifier', or 'regressor'.")
    return sorted(n for n, c in idx.items() if issubclass(c, want))
