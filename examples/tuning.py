"""Hyperparameter tuning in PyTidyModels.

Run:  python examples/tuning.py
"""
import warnings

from pytidymodels import (
    datasets,
    dials,
    parsnip,
    recipes,
    rsample,
    tune,
    workflows,
    yardstick,
)
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")


def main():
    ames = datasets.load_ames_small(n=600, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    train = split.training()

    rec = (recipes.Recipe(train, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))

    # mark arguments to tune with parsnip.tune()
    spec = (parsnip.boost_tree(trees=parsnip.tune(), learn_rate=parsnip.tune(),
                               tree_depth=parsnip.tune())
            .set_mode("regression"))
    wf = workflows.Workflow().add_recipe(rec).add_model(spec)

    # 5-fold CV, space-filling grid over the three parameters
    folds = rsample.vfold_cv(train, v=5, strata="price", seed=2)
    grid = dials.grid_latin_hypercube(
        dials.trees(range=(100, 600)),
        dials.learn_rate(range=(1e-2, 0.3)),
        dials.tree_depth(range=(2, 6)),
        size=12, seed=1,
    )
    print(f"Tuning boost_tree: {len(grid)} candidates x 5 folds\n")

    res = tune.tune_grid(wf, folds, grid,
                         yardstick.metric_set(yardstick.rmse, yardstick.rsq))

    print("Best candidates by CV RMSE:")
    print(tune.show_best(res, "rmse", n=5).to_string(index=False))

    best = tune.select_best(res, "rmse")
    print(f"\nselect_best(rmse) -> {best}")

    # finalize and evaluate on the held-out test set
    final = tune.finalize_workflow(wf, best)
    lf = tune.last_fit(final, split,
                       yardstick.metric_set(yardstick.rmse, yardstick.rsq))
    print("\nlast_fit on the test set:")
    print(lf.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
