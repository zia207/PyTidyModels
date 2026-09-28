"""Small, dependency-free sample datasets (a modeldata analog).

Synthetic but realistic: mixed numeric + nominal predictors, some missing
values, and a rare categorical level -- enough to exercise imputation,
lumping, dummy encoding, and normalization end to end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_NEIGHBORHOODS = ["North", "South", "East", "West", "Riverside"]
_STYLES = ["Ranch", "Colonial", "Cape", "Split", "Victorian"]


def _base_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.normal(1800, 500, n).clip(500, 5000)
    age = rng.integers(0, 100, n).astype(float)
    income = rng.lognormal(mean=11.0, sigma=0.4, size=n)  # positive, skewed
    # weighted so "Riverside" is rare -> exercises step_other
    nb = rng.choice(_NEIGHBORHOODS, n, p=[0.30, 0.28, 0.22, 0.17, 0.03])
    style = rng.choice(_STYLES, n)

    nb_effect = pd.Series(nb).map(
        {"North": 40, "South": 10, "East": 25, "West": 15, "Riverside": 60}
    ).to_numpy()
    price = (60 * area + 800 * nb_effect - 300 * age
             + 0.15 * income + rng.normal(0, 15000, n))

    df = pd.DataFrame({
        "area": area, "age": age, "income": income,
        "neighborhood": nb, "style": style, "price": price,
    })
    # inject missingness into numeric predictors
    for col in ("area", "income"):
        miss = rng.choice(n, size=int(0.05 * n), replace=False)
        df.loc[miss, col] = np.nan
    return df


def load_ames_small(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Regression sample: predict ``price`` from home features."""
    return _base_frame(n, seed)


def load_credit_small(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Classification sample: a binary ``status`` (good/bad) outcome."""
    df = _base_frame(n, seed)
    z = (0.0006 * df["price"].fillna(df["price"].median())
         - 0.02 * df["age"] - 6.0)
    prob = 1 / (1 + np.exp(-(z - z.mean()) / z.std()))
    rng = np.random.default_rng(seed + 1)
    df["status"] = np.where(rng.random(n) < prob, "good", "bad")
    return df.drop(columns=["price"])
