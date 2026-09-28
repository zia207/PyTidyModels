"""Tuning parameters and grid construction.

A :class:`Parameter` describes one tunable argument -- its range, whether it's
integer-valued, and whether to sample on a log scale. Named constructors
(``trees()``, ``penalty()``, ...) give sensible default ranges keyed to the
tidy argument names the model specs use, so a grid's columns line up with the
spec arguments they fill.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Parameter:
    name: str
    lower: float = 0.0
    upper: float = 1.0
    integer: bool = False
    log: bool = False                       # sample on a log10 scale
    values: Optional[Sequence] = None       # explicit set (categorical/ordinal)

    def sample(self, u: float):
        """Map a quantile ``u`` in [0, 1] to a value in this parameter's space."""
        if self.values is not None:
            idx = min(int(u * len(self.values)), len(self.values) - 1)
            return self.values[idx]
        lo, hi = self.lower, self.upper
        if self.log:
            lo, hi = np.log10(lo), np.log10(hi)
        x = lo + u * (hi - lo)
        if self.log:
            x = 10.0 ** x
        if self.integer:
            return int(round(x))
        return float(x)

    def grid_points(self, levels: int) -> list:
        """Evenly spaced representative values (deduplicated)."""
        if self.values is not None:
            return list(self.values)
        us = np.linspace(0.0, 1.0, levels)
        seen, out = set(), []
        for u in us:
            v = self.sample(u)
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out


# -- named parameter constructors (defaults match common tidy args) ---------

def param(name, range=(0.0, 1.0), integer=False, log=False, values=None):
    """Generic parameter constructor."""
    return Parameter(name, range[0], range[1], integer=integer, log=log,
                     values=values)


def trees(range=(50, 2000)):
    return Parameter("trees", *range, integer=True)


def mtry(range=(1, 15)):
    return Parameter("mtry", *range, integer=True)


def min_n(range=(2, 40)):
    return Parameter("min_n", *range, integer=True)


def tree_depth(range=(1, 15)):
    return Parameter("tree_depth", *range, integer=True)


def learn_rate(range=(1e-3, 0.3)):
    return Parameter("learn_rate", *range, log=True)


def penalty(range=(1e-4, 1.0)):
    return Parameter("penalty", *range, log=True)


def mixture(range=(0.0, 1.0)):
    return Parameter("mixture", *range)


def cost(range=(1e-2, 10.0)):
    return Parameter("cost", *range, log=True)


def neighbors(range=(1, 30)):
    return Parameter("neighbors", *range, integer=True)


def hidden_units(range=(8, 128)):
    return Parameter("hidden_units", *range, integer=True)


def epochs(range=(50, 500)):
    return Parameter("epochs", *range, integer=True)


def finalize(parameter: Parameter, data: pd.DataFrame) -> Parameter:
    """Resolve a data-dependent range (currently ``mtry`` -> #columns)."""
    if parameter.name == "mtry":
        parameter.upper = int(data.shape[1])
    return parameter


# -- grid builders ----------------------------------------------------------

def grid_regular(*params: Parameter, levels: int = 3) -> pd.DataFrame:
    """Cartesian grid of evenly spaced levels per parameter."""
    axes = {p.name: p.grid_points(levels) for p in params}
    keys = list(axes)
    rows = [dict(zip(keys, combo)) for combo in product(*(axes[k] for k in keys))]
    return pd.DataFrame(rows)


def grid_random(*params: Parameter, size: int = 10,
                seed: Optional[int] = None) -> pd.DataFrame:
    """Uniform random grid over the parameter space."""
    rng = np.random.default_rng(seed)
    rows = [{p.name: p.sample(rng.random()) for p in params} for _ in range(size)]
    return pd.DataFrame(rows)


def grid_latin_hypercube(*params: Parameter, size: int = 10,
                         seed: Optional[int] = None) -> pd.DataFrame:
    """Space-filling Latin-hypercube grid (via scipy QMC)."""
    from scipy.stats.qmc import LatinHypercube
    U = LatinHypercube(d=len(params), seed=seed).random(size)
    rows = [{p.name: p.sample(U[i, j]) for j, p in enumerate(params)}
            for i in range(size)]
    return pd.DataFrame(rows)
