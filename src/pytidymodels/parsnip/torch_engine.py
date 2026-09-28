"""Optional PyTorch engine for PyTidyModels.

PyTorch is not a catalog of models the way scikit-learn is -- it's a
framework whose "models" are ``nn.Module`` graphs. So "adding PyTorch"
here means two things:

* a **torch engine for the ``mlp`` spec** -- ``mlp(...).set_engine("torch")``
  trains a configurable feed-forward net; and
* a **generic wrapper**, :func:`~pytidymodels.parsnip.sklearn`-style, that
  adapts *any* ``nn.Module`` (yours, torchvision's, timm's, torch.hub's) into
  the same tidy ``fit -> predict -> metrics`` flow -- see
  ``parsnip.torch_model``.

torch is an *optional* dependency. Nothing here is imported at package import
time; torch is only imported when a torch-backed spec is actually fit, and a
clear message is raised if it's missing. Install with::

    pip install "pytidymodels[torch]"

The wrapper deliberately mimics the scikit-learn estimator API
(``fit`` / ``predict`` / ``predict_proba`` / ``classes_``) so it reuses the
existing tidy-prediction machinery unchanged.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
    except (ImportError, OSError) as e:  # OSError: broken/partial install
        raise ImportError(
            "This model uses the PyTorch engine, which needs a working torch "
            "install. Install it with:  pip install 'pytidymodels[torch]'"
        ) from e
    import torch
    return torch


_ACTIVATIONS = {
    "relu": "ReLU",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "gelu": "GELU",
    "elu": "ELU",
    "leaky_relu": "LeakyReLU",
}


def make_mlp_factory(hidden_units: Union[int, Sequence[int]] = (32,),
                     activation: str = "relu", dropout: float = 0.0) -> Callable:
    """Return a ``factory(in_features, out_features) -> nn.Module`` MLP builder."""
    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,)
    hidden_units = tuple(hidden_units)

    def factory(in_features: int, out_features: int):
        torch = _require_torch()
        import torch.nn as nn
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; choose from "
                f"{sorted(_ACTIVATIONS)}.")
        act = getattr(nn, _ACTIVATIONS[activation])
        layers = []
        prev = in_features
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))
        return nn.Sequential(*layers)

    return factory


class TorchEstimator:
    """A scikit-learn-style wrapper around a PyTorch module + training loop.

    ``module`` may be a ready ``nn.Module`` instance, an ``nn.Module``
    subclass, or a factory callable. A factory is called as
    ``factory(in_features, out_features)`` when possible, falling back to a
    zero-argument call -- so both shape-aware and fixed-shape modules work.
    """

    def __init__(self, module, mode: str, epochs: int = 100,
                 batch_size: int = 32, lr: float = 1e-3,
                 weight_decay: float = 0.0, device: Optional[str] = None,
                 seed: Optional[int] = None, verbose: bool = False):
        self.module = module
        self.mode = mode
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.seed = seed
        self.verbose = verbose

    # -- module resolution --------------------------------------------------

    def _resolve_module(self, in_features: int, out_features: int):
        torch = _require_torch()
        import torch.nn as nn
        m = self.module
        if isinstance(m, nn.Module):
            return m                      # a ready-built instance
        if isinstance(m, type) and issubclass(m, nn.Module):
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
            "torch model must be an nn.Module, an nn.Module subclass, or a "
            f"factory callable; got {type(m).__name__}.")

    # -- sklearn-style API --------------------------------------------------

    def fit(self, X, y):
        torch = _require_torch()
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if self.seed is not None:
            torch.manual_seed(self.seed)
        device = torch.device(self.device
                              or ("cuda" if torch.cuda.is_available() else "cpu"))

        X_arr = np.asarray(X, dtype=np.float32)
        Xt = torch.from_numpy(X_arr)
        y_arr = np.asarray(y)

        if self.mode == "classification":
            self.classes_ = np.unique(y_arr)
            y_idx = np.searchsorted(self.classes_, y_arr).astype(np.int64)
            yt = torch.from_numpy(y_idx)
            out_features = len(self.classes_)
            loss_fn = nn.CrossEntropyLoss()
        else:
            yt = torch.from_numpy(y_arr.astype(np.float32)).view(-1, 1)
            out_features = 1
            loss_fn = nn.MSELoss()

        module = self._resolve_module(X_arr.shape[1], out_features).to(device)
        opt = torch.optim.Adam(module.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        ds = TensorDataset(Xt, yt)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        module.train()
        for epoch in range(self.epochs):
            running = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = module(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                running += float(loss.item()) * len(xb)
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"epoch {epoch+1}/{self.epochs}  loss={running/len(ds):.4f}")

        self.module_ = module
        self.device_ = device
        self.n_features_in_ = X_arr.shape[1]
        return self

    def _forward(self, X):
        torch = _require_torch()
        module = self.module_
        module.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device_)
        with torch.no_grad():
            return module(Xt).cpu().numpy()

    def predict(self, X):
        out = self._forward(X)
        if self.mode == "classification":
            idx = out.argmax(axis=1)
            return self.classes_[idx]
        return out.ravel()

    def predict_proba(self, X):
        torch = _require_torch()
        out = self._forward(X)
        t = torch.from_numpy(out)
        return torch.softmax(t, dim=1).numpy()


def list_torch_models() -> list[str]:
    """Architectures available via torchvision, if it's installed.

    These are **image** models (they expect image tensors, not the tabular
    design matrix the recipe/workflow flow produces), so they're listed for
    reference; use them via ``torch_model`` with image data, not the tabular
    pipeline. Returns an empty list when torchvision is absent.
    """
    try:
        import torchvision
    except ImportError:
        return []
    return sorted(torchvision.models.list_models())
