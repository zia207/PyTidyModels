"""End-to-end PyTidyModels slice: split -> recipe -> spec -> workflow -> metrics.

Run:  python examples/hello_world.py
"""
from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import (
    all_nominal_predictors,
    all_numeric_predictors,
)


def regression_demo():
    print("=" * 64)
    print("REGRESSION  ·  predict home price")
    print("=" * 64)
    ames = datasets.load_ames_small(n=600, seed=42)

    # 1. split (stratified on the outcome)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    train, test = split.training(), split.testing()
    print(split)

    # 2. preprocess
    rec = (recipes.Recipe(train, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_other(all_nominal_predictors(), threshold=0.05)
           .step_log("income")
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))

    # 3. specify (Random Forest via scikit-learn engine)
    rf = (parsnip.rand_forest(trees=300, min_n=5)
          .set_engine("sklearn", random_state=0)
          .set_mode("regression"))

    # 4. bundle + fit
    wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(train)
    print(wf)

    # 5. predict + evaluate on the test set
    results = wf.augment(test)
    metrics = yardstick.metric_set(yardstick.rmse, yardstick.rsq, yardstick.mae)
    print(metrics(results, truth="price", estimate=".pred").to_string(index=False))

    # cross-validated RMSE, to show the resampling loop composes
    folds = rsample.vfold_cv(train, v=5, strata="price", seed=2)
    rmses = []
    for fold in folds:
        f_wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(fold.analysis())
        aug = f_wf.augment(fold.assessment())
        rmses.append(yardstick.rmse(aug, "price", ".pred")["estimate"].iloc[0])
    print(f"\n5-fold CV RMSE: mean={sum(rmses)/len(rmses):,.0f}  "
          f"(folds: {', '.join(f'{r:,.0f}' for r in rmses)})")


def classification_demo():
    print("\n" + "=" * 64)
    print("CLASSIFICATION  ·  predict credit status")
    print("=" * 64)
    credit = datasets.load_credit_small(n=600, seed=7)
    split = rsample.initial_split(credit, prop=0.8, strata="status", seed=1)
    train, test = split.training(), split.testing()

    rec = (recipes.Recipe(train, formula="status ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))

    logit = (parsnip.logistic_reg()
             .set_engine("sklearn")
             .set_mode("classification"))

    wf = workflows.Workflow().add_recipe(rec).add_model(logit).fit(train)
    results = wf.augment(test)

    acc = yardstick.accuracy(results, "status", ".pred_class")
    auc = yardstick.roc_auc(results, "status", ".pred_good")
    print(acc.to_string(index=False))
    print(auc.to_string(index=False))


if __name__ == "__main__":
    regression_demo()
    classification_demo()
    print("\nDone.")
