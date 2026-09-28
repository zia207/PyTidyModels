"""Tests for the magrittr-style pipe operator."""
import warnings

import pandas as pd
import pytest

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.pipe import Pipe, _, pipe
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")


# -- core semantics --------------------------------------------------------

def test_lhs_becomes_first_arg():
    assert (5 >> pipe(pow, 2)) == 25          # pow(5, 2)
    assert ("abc" >> pipe(str.upper)) == "ABC"


def test_kwargs_pass_through():
    assert (3.14159 >> pipe(round, ndigits=2)) == 3.14


def test_placeholder_positions_the_value():
    assert (2 >> pipe(pow, _, 10)) == 1024     # pow(2, 10)
    assert (10 >> pipe(pow, 2, _)) == 1024     # pow(2, 10)  -> value at 2nd arg
    assert ("_" >> pipe("a{}c".format)) == "a_c"


def test_placeholder_in_kwargs():
    def f(a, b):
        return (a, b)
    assert (7 >> pipe(f, a=1, b=_)) == (1, 7)


def test_pipe_chaining_left_to_right():
    result = "  hi there  " >> pipe(str.strip) >> pipe(str.upper)
    assert result == "HI THERE"


def test_composed_pipeline_is_reusable():
    flow = pipe(str.strip) >> pipe(str.upper) >> pipe(str.replace, " ", "_")
    assert isinstance(flow, Pipe)
    assert ("  a b  " >> flow) == "A_B"
    assert ("  c d  " >> flow) == "C_D"       # reused, no state carried


def test_pipe_is_callable_directly():
    p = pipe(pow, 2)
    assert p(5) == 25


def test_pipe_requires_callable():
    with pytest.raises(TypeError):
        pipe(42)


def test_dataframe_pipes_without_error():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    n = df >> pipe(len)
    assert n == 3


# -- integration with the modeling flow ------------------------------------

def test_split_training_testing_via_pipe():
    ames = datasets.load_ames_small(n=200, seed=0)
    train = (ames >> pipe(rsample.initial_split, prop=0.75, seed=1)
             >> pipe(rsample.training))
    test = (ames >> pipe(rsample.initial_split, prop=0.75, seed=1)
            >> pipe(rsample.testing))
    assert len(train) + len(test) == len(ames)


def test_scoring_via_pipe_matches_direct_call():
    ames = datasets.load_ames_small(n=300, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    train, test = split.training(), split.testing()
    rec = (recipes.Recipe(train, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))
    rf = parsnip.rand_forest(trees=80).set_engine("sklearn", random_state=0).set_mode("regression")
    wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(train)
    score = yardstick.metric_set(yardstick.rmse, yardstick.rsq)

    piped = (wf >> pipe(workflows.augment, test)
             >> pipe(score, truth="price", estimate=".pred"))
    direct = score(wf.augment(test), truth="price", estimate=".pred")
    pd.testing.assert_frame_equal(piped.reset_index(drop=True),
                                  direct.reset_index(drop=True))


def test_workflow_pipe_method():
    ames = datasets.load_ames_small(n=200, seed=0)
    rec = (recipes.Recipe(ames, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_dummy(all_nominal_predictors()))
    rf = parsnip.rand_forest(trees=50).set_mode("regression")
    wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(ames)
    # .pipe() threads the workflow into an arbitrary function
    n_rows = wf.pipe(lambda w: w.augment(ames)).shape[0]
    assert n_rows == len(ames)
