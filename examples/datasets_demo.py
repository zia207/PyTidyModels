"""Bundled datasets in PyTidyModels.

Run:  python examples/datasets_demo.py
"""
import warnings

from pytidymodels import (
    datasets,
    parsnip,
    recipes,
    rsample,
    workflows,
    yardstick,
)
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")


def main():
    idx = datasets.list_datasets()
    print("Bundled datasets:\n")
    print(idx[["dataset", "task", "outcome", "n_rows", "n_cols"]].to_string(index=False))

    # model on a real classification dataset (Pima diabetes)
    print("\n" + "=" * 60)
    print("Modeling load_pima()  ·  classification")
    print("=" * 60)
    pima = datasets.load_pima()
    split = rsample.initial_split(pima, prop=0.8, strata="Outcome", seed=1)
    train, test = split.training(), split.testing()
    rec = (recipes.Recipe(train, formula="Outcome ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_normalize(all_numeric_predictors()))
    rf = (parsnip.rand_forest(trees=300).set_engine("sklearn", random_state=0)
          .set_mode("classification"))
    wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(train)
    res = wf.augment(test)
    acc = yardstick.accuracy(res, "Outcome", ".pred_class")["estimate"].iloc[0]
    auc = yardstick.roc_auc(res, "Outcome", ".pred_1")["estimate"].iloc[0]
    print(f"random forest  ·  accuracy={acc:.3f}  roc_auc={auc:.3f}")

    # a real dataset with categoricals (NMES physician visits, a count outcome)
    print("\n" + "=" * 60)
    print("Modeling load_nmes()  ·  count outcome, mixed types")
    print("=" * 60)
    nmes = datasets.load_nmes()
    split = rsample.initial_split(nmes, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    rec = (recipes.Recipe(train, formula="visits ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))
    pois = parsnip.poisson_reg().set_mode("regression")
    wf = workflows.Workflow().add_recipe(rec).add_model(pois).fit(train)
    res = wf.augment(test)
    mae = yardstick.mae(res, "visits", ".pred")["estimate"].iloc[0]
    print(f"poisson regression  ·  MAE={mae:.2f} visits")


if __name__ == "__main__":
    main()
