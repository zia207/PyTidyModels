"""Pipe-operator showcase: the same modeling flow written tidyverse-style.

Python can't define a new ``%>%`` token, so PyTidyModels overloads ``>>``
(as dfply / siuba / plydata do) with magrittr semantics: the left value
becomes the first argument, and ``_`` is the placeholder (magrittr's ``.``).

Run:  python examples/pipe_flow.py
"""
import warnings

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.pipe import _, pipe
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")


def basics():
    print("magrittr-style semantics")
    print("-" * 40)
    print("  5 >> pipe(pow, 2)        =", 5 >> pipe(pow, 2))       # pow(5, 2)
    print("  2 >> pipe(pow, _, 10)    =", 2 >> pipe(pow, _, 10))   # pow(2, 10)
    print("  '  hi ' >> strip>>upper  =", "  hi " >> pipe(str.strip) >> pipe(str.upper))
    flow = pipe(str.strip) >> pipe(str.title)     # reusable composed pipeline
    print("  reusable pipeline        =", "  ada lovelace  " >> flow)


def modeling():
    print("\nmodeling flow, piped")
    print("-" * 40)
    ames = datasets.load_ames_small(n=600, seed=42)

    # data -> split -> training / testing, read left to right
    split = ames >> pipe(rsample.initial_split, prop=0.8, strata="price", seed=1)
    train = split >> pipe(rsample.training)
    test = split >> pipe(rsample.testing)

    # recipe stays a fluent chain (the tidy idiom for preprocessing)
    rec = (recipes.Recipe(train, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_other(all_nominal_predictors(), threshold=0.05)
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))

    rf = parsnip.rand_forest(trees=300).set_mode("regression")
    wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(train)

    # fitted workflow -> augment test -> score, all piped
    score = yardstick.metric_set(yardstick.rmse, yardstick.rsq, yardstick.mae)
    out = (wf
           >> pipe(workflows.augment, test)
           >> pipe(score, truth="price", estimate=".pred"))
    print(out.to_string(index=False))


if __name__ == "__main__":
    basics()
    modeling()
    print("\nDone.")
