"""Optional Keras engine for PyTidyModels.

Like PyTorch, Keras is a framework, not an enumerable catalog of models -- its
"models" are ``keras.Model`` graphs (Sequential / Functional / subclassed).
So "adding Keras" means:

* a **keras engine for the ``mlp`` spec** -- ``mlp(...).set_engine("keras")``
  trains a configurable feed-forward net; and
* a **generic wrapper** -- ``parsnip.keras_model(...)`` adapts *any*
  ``keras.Model`` into the tidy ``fit -> predict -> metrics`` flow.

keras is an *optional* dependency and is imported lazily. Keras 3 is
backend-agnostic (TensorFlow, JAX, or PyTorch); if ``KERAS_BACKEND`` isn't
set, this module picks the first backend actually installed, so it "just
works" whichever you have. Install with::

    pip install "pytidymodels[keras]"          # keras + TensorFlow
    # or bring your own backend: pip install keras jax jaxlib

The wrapper mimics the scikit-learn estimator API
(``fit`` / ``predict`` / ``predict_proba`` / ``classes_``) so it reuses the
tidy-prediction machinery unchanged.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, Union

import numpy as np


def _require_keras():
    # Choose a backend before the first keras import if the user hasn't.
    if not os.environ.get("KERAS_BACKEND"):
        for backend, module in (("tensorflow", "tensorflow"),
                                ("jax", "jax"), ("torch", "torch")):
            try:
                __import__(module)
                os.environ["KERAS_BACKEND"] = backend
                break
            except Exception:
                continue
    try:
        import keras  # noqa: F401
    except (ImportError, OSError) as e:
        raise ImportError(
            "This model uses the Keras engine, which needs keras plus a "
            "backend (TensorFlow, JAX, or PyTorch). Install with:  "
            "pip install 'pytidymodels[keras]'"
        ) from e
    import keras
    return keras


def make_keras_mlp_factory(hidden_units: Union[int, Sequence[int]] = (32,),
                           activation: str = "relu", dropout: float = 0.0,
                           mode: str = "regression") -> Callable:
    """Return a ``factory(in_features, out_features) -> keras.Model`` MLP builder."""
    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,)
    hidden_units = tuple(hidden_units)

    def factory(in_features: int, out_features: int):
        keras = _require_keras()
        from keras import layers
        model = keras.Sequential()
        model.add(keras.Input(shape=(in_features,)))
        for h in hidden_units:
            model.add(layers.Dense(h, activation=activation))
            if dropout and dropout > 0:
                model.add(layers.Dropout(dropout))
        if mode == "classification":
            model.add(layers.Dense(out_features, activation="softmax"))
        else:
            model.add(layers.Dense(out_features))
        return model

    return factory


class KerasEstimator:
    """A scikit-learn-style wrapper around a Keras model + training loop.

    ``model`` may be a ready ``keras.Model``, a ``keras.Model`` subclass, or a
    factory callable (``factory(in_features, out_features)``, falling back to a
    zero-argument call). An already-compiled model keeps its loss/optimizer;
    an uncompiled one is compiled with a mode-appropriate loss (MSE for
    regression, sparse categorical cross-entropy for classification).
    """

    def __init__(self, model, mode: str, epochs: int = 100,
                 batch_size: int = 32, lr: float = 1e-3,
                 weight_decay: float = 0.0, seed: Optional[int] = None,
                 verbose: int = 0):
        self.model = model
        self.mode = mode
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = seed
        self.verbose = verbose

    def _resolve_model(self, in_features: int, out_features: int):
        keras = _require_keras()
        m = self.model
        if isinstance(m, keras.Model):
            return m
        if isinstance(m, type) and issubclass(m, keras.Model):
            try:
                return m(in_features, out_features)
            except TypeError:
                return m()
        if callable(m):
            try:
                return m(in_features, out_features)
            except TypeError:
                return m()
        raise TypeError(
            "keras model must be a keras.Model, a keras.Model subclass, or a "
            f"factory callable; got {type(m).__name__}.")

    @staticmethod
    def _is_compiled(model) -> bool:
        try:
            return model.optimizer is not None
        except Exception:
            return False

    def fit(self, X, y):
        keras = _require_keras()
        if self.seed is not None:
            keras.utils.set_random_seed(int(self.seed))

        X_arr = np.asarray(X, dtype="float32")
        y_arr = np.asarray(y)
        if self.mode == "classification":
            self.classes_ = np.unique(y_arr)
            target = np.searchsorted(self.classes_, y_arr).astype("int64")
            out_features = len(self.classes_)
            loss = "sparse_categorical_crossentropy"
        else:
            target = y_arr.astype("float32")
            out_features = 1
            loss = "mse"

        model = self._resolve_model(X_arr.shape[1], out_features)
        if not self._is_compiled(model):
            opt = keras.optimizers.Adam(learning_rate=self.lr,
                                        weight_decay=self.weight_decay or None)
            model.compile(optimizer=opt, loss=loss)
        model.fit(X_arr, target, epochs=self.epochs,
                  batch_size=self.batch_size, verbose=self.verbose)
        self.model_ = model
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X):
        out = np.asarray(self.model_.predict(
            np.asarray(X, dtype="float32"), verbose=0))
        if self.mode == "classification":
            return self.classes_[out.argmax(axis=1)]
        return out.ravel()

    def predict_proba(self, X):
        return np.asarray(self.model_.predict(
            np.asarray(X, dtype="float32"), verbose=0))


def list_keras_applications() -> list[str]:
    """Prebuilt architectures in ``keras.applications``, if keras is installed.

    These are **image** models (they expect image tensors, not the tabular
    design matrix), so they're listed for reference; use them via
    ``keras_model`` with image data, not the tabular pipeline. Empty when
    keras is unavailable.
    """
    try:
        keras = _require_keras()
    except ImportError:
        return []
    apps = keras.applications
    return sorted(n for n in dir(apps)
                  if n[0].isupper() and callable(getattr(apps, n)))
