"""Optional H2O engine for PyTidyModels.

H2O runs models on a Java cluster (a local JVM by default) and uses its own
``H2OFrame``. This engine hides both behind the usual scikit-learn-style
adapter, so H2O models flow through the same recipe -> workflow -> yardstick
pipeline as everything else.

Supported via ``.set_engine("h2o")``:

* ``linear_reg`` / ``logistic_reg``  -> ``H2OGeneralizedLinearEstimator``
* ``rand_forest``                    -> ``H2ORandomForestEstimator``
* ``boost_tree``                     -> ``H2OGradientBoostingEstimator``
* ``mlp``                            -> ``H2ODeepLearningEstimator``

Plus ``parsnip.h2o_automl(...)`` which runs ``H2OAutoML`` and predicts with the
leader model.

The cluster starts lazily on first fit; configure it up front with
``parsnip.h2o_init(max_mem_size="4G", nthreads=-1)`` and stop it with
``parsnip.h2o_shutdown()``. H2O is optional:
``pip install "pytidymodels[h2o]"`` (a Java 8-21 runtime is also required).
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

_INIT_KWARGS: dict = {}
_STARTED = False


def _require_h2o():
    try:
        import h2o  # noqa: F401
    except (ImportError, OSError) as e:
        raise ImportError(
            "The H2O engine needs the h2o package (and a Java runtime). "
            "Install with:  pip install 'pytidymodels[h2o]'") from e
    import h2o
    return h2o


def h2o_init(**kwargs):
    """Start (or restart) the H2O cluster with the given init options."""
    global _INIT_KWARGS, _STARTED
    _INIT_KWARGS = kwargs
    _STARTED = False
    return ensure_cluster()


def ensure_cluster():
    """Start the cluster once; reconnect quietly on later calls."""
    global _STARTED
    h2o = _require_h2o()
    if _STARTED:
        return h2o
    kw = dict(bind_to_localhost=True)
    kw.update(_INIT_KWARGS)
    with contextlib.redirect_stdout(io.StringIO()):   # silence the boot banner
        h2o.init(**kw)
        h2o.no_progress()
    _STARTED = True
    return h2o


def h2o_shutdown():
    """Shut down the H2O cluster if this process started one."""
    global _STARTED
    if _STARTED:
        _require_h2o().cluster().shutdown()
        _STARTED = False


def _to_frame(h2o, X, columns=None):
    df = pd.DataFrame(X, columns=columns).reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return h2o.H2OFrame(df)


class H2OEstimator:
    """scikit-learn-style wrapper around a single H2O estimator."""

    def __init__(self, estimator_cls, params: dict, mode: str):
        self.estimator_cls = estimator_cls
        self.params = params
        self.mode = mode

    def _train(self, hf, target):
        model = self.estimator_cls(**self.params)
        model.train(x=self.features_, y=target, training_frame=hf)
        return model

    def fit(self, X, y):
        h2o = ensure_cluster()
        Xdf = pd.DataFrame(X).reset_index(drop=True)
        self.features_ = list(map(str, Xdf.columns))
        Xdf.columns = self.features_
        target = "__target__"
        y = np.asarray(y)
        df = Xdf.copy()
        df[target] = y
        hf = _to_frame(h2o, df)
        if self.mode == "classification":
            hf[target] = hf[target].asfactor()
            levels = hf[target].levels()[0]              # H2O factor order (strings)
            by_str = {str(v): v for v in pd.unique(y)}
            self._by_str = by_str
            self.classes_ = np.array([by_str.get(l, l) for l in levels], dtype=object)
            self._levels = levels
        self.model = self._train(hf, target)
        return self

    def _predict_df(self, X):
        h2o = ensure_cluster()
        hf = _to_frame(h2o, pd.DataFrame(X, columns=self.features_),
                       columns=self.features_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict(hf).as_data_frame()

    def predict(self, X):
        pdf = self._predict_df(X)
        raw = pdf["predict"]
        if self.mode == "classification":
            return raw.map(lambda s: self._by_str.get(str(s), s)).to_numpy()
        return raw.to_numpy()

    def predict_proba(self, X):
        pdf = self._predict_df(X)
        prob_cols = [c for c in pdf.columns if c != "predict"]  # factor-level order
        return pdf[prob_cols].to_numpy()


class H2OAutoMLEstimator(H2OEstimator):
    """Wraps H2OAutoML; predicts with the leader model."""

    def __init__(self, params: dict, mode: str):
        self.params = params
        self.mode = mode

    def _train(self, hf, target):
        from h2o.automl import H2OAutoML
        aml = H2OAutoML(**self.params)
        aml.train(x=self.features_, y=target, training_frame=hf)
        self.automl_ = aml
        self.model = aml.leader
        return aml.leader

    def leaderboard(self) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.automl_.leaderboard.as_data_frame()


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _drop_none(d):
    return {k: v for k, v in d.items() if v is not None}


def build_h2o(spec):
    """Build an H2OEstimator for `<model>.set_engine('h2o')`."""
    _require_h2o()
    mt, mode = spec.model_type, spec.mode
    a, ek = spec.args, dict(spec.engine_kwargs)

    if mt in ("linear_reg", "logistic_reg"):
        from h2o.estimators import H2OGeneralizedLinearEstimator as GLM
        family = "binomial" if mode == "classification" else "gaussian"
        params = _drop_none({"family": family,
                             "lambda_": a.get("penalty") if a.get("penalty") is not None else 0.0,
                             "alpha": a.get("mixture")})
        params.update(ek)
        return H2OEstimator(GLM, params, mode)

    if mt == "rand_forest":
        from h2o.estimators import H2ORandomForestEstimator as DRF
        params = _drop_none({"ntrees": a.get("trees"), "min_rows": a.get("min_n"),
                             "mtries": a.get("mtry")})
        params.update(ek)
        return H2OEstimator(DRF, params, mode)

    if mt == "boost_tree":
        from h2o.estimators import H2OGradientBoostingEstimator as GBM
        params = _drop_none({"ntrees": a.get("trees"), "max_depth": a.get("tree_depth"),
                             "learn_rate": a.get("learn_rate"), "min_rows": a.get("min_n")})
        params.update(ek)
        return H2OEstimator(GBM, params, mode)

    if mt == "mlp":
        from h2o.estimators import H2ODeepLearningEstimator as DL
        hidden = a.get("hidden_units")
        if hidden is not None and not isinstance(hidden, (list, tuple)):
            hidden = [int(hidden)]
        params = _drop_none({"hidden": list(hidden) if hidden else None,
                             "epochs": a.get("epochs"), "activation": a.get("activation")})
        params.update(ek)
        return H2OEstimator(DL, params, mode)

    raise ValueError(
        f"engine 'h2o' is available for linear_reg, logistic_reg, rand_forest, "
        f"boost_tree, mlp -- not {mt!r}. For AutoML use parsnip.h2o_automl(...).")


def build_h2o_automl(spec):
    _require_h2o()
    return H2OAutoMLEstimator(dict(spec.h2o["kwargs"]), spec.mode)


def list_h2o_models() -> list[str]:
    """H2O algorithms reachable through this engine."""
    return ["linear_reg (GLM)", "logistic_reg (GLM)", "rand_forest (DRF)",
            "boost_tree (GBM)", "mlp (DeepLearning)", "h2o_automl (AutoML)"]
