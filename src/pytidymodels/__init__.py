"""PyTidyModels -- a tidy, modular modeling grammar for Python.

A tidymodels-style meta-package built on top of scikit-learn. Import the
whole thing::

    import pytidymodels as tm
    tm.rsample, tm.recipes, tm.parsnip, tm.workflows, tm.yardstick

or take just the pieces you need::

    from pytidymodels import rsample, recipes, parsnip, workflows, yardstick

This is the Phase 0-1 vertical slice: split -> preprocess -> specify ->
bundle -> evaluate. Tuning (dials/tune), post-processing (tailor), tidy
outputs (broom) and workflow sets land in later phases.
"""
from . import (
    datasets,
    dials,
    explain,
    hardhat,
    parsnip,
    recipes,
    rsample,
    tune,
    workflows,
    yardstick,
)
from .pipe import P, Pipe, _, pipe, pipeable
from .pipe import _, pipe

# Plain-English aliases for Python-native users who find the R names opaque.
resampling = rsample
preprocessing = recipes
models = parsnip
metrics = yardstick
tuning = tune
interpret = explain

__version__ = "0.9.0"

__all__ = [
    "hardhat",
    "rsample",
    "recipes",
    "parsnip",
    "workflows",
    "yardstick",
    "dials",
    "tune",
    "explain",
    "datasets",
    "resampling",
    "preprocessing",
    "models",
    "metrics",
    "tuning",
    "interpret",
    "pipe",
    "P",
    "Pipe",
    "_",
    "pipeable",
    "__version__",
]
