"""Tests for dials parameters and grid builders."""
import pandas as pd

from pytidymodels import dials


def test_parameter_sample_bounds_and_integer():
    p = dials.trees(range=(50, 200))
    assert p.sample(0.0) == 50 and p.sample(1.0) == 200
    assert isinstance(p.sample(0.5), int)


def test_parameter_log_scale():
    p = dials.penalty(range=(1e-4, 1.0))          # log10 space
    lo, mid, hi = p.sample(0.0), p.sample(0.5), p.sample(1.0)
    assert abs(lo - 1e-4) < 1e-9 and abs(hi - 1.0) < 1e-9
    assert abs(mid - 1e-2) < 1e-6                  # geometric midpoint


def test_grid_regular_is_cartesian():
    g = dials.grid_regular(dials.mtry(range=(1, 3)), dials.min_n(range=(2, 6)),
                           levels=3)
    assert set(g.columns) == {"mtry", "min_n"}
    assert len(g) == 9


def test_grid_random_size_and_columns():
    g = dials.grid_random(dials.mtry(range=(1, 10)), dials.trees(), size=7, seed=1)
    assert len(g) == 7 and set(g.columns) == {"mtry", "trees"}


def test_grid_latin_hypercube_size():
    g = dials.grid_latin_hypercube(dials.learn_rate(), dials.min_n(), size=5, seed=1)
    assert len(g) == 5 and set(g.columns) == {"learn_rate", "min_n"}


def test_finalize_mtry_uses_ncols():
    p = dials.mtry(range=(1, 100))
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    dials.finalize(p, df)
    assert p.upper == 3
