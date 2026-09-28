"""Model catalog showcase: swap models freely over one shared recipe.

Demonstrates the expanded parsnip catalog and universal scikit-learn access.

Run:  python examples/model_catalog.py
"""
import warnings

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")


def main():
    print(f"Curated model types ({len(parsnip.list_models())}):")
    print("  " + ", ".join(parsnip.list_models()))
    print(f"\nUniversal access: {len(parsnip.list_sklearn_estimators('classifier'))} "
          f"sklearn classifiers + "
          f"{len(parsnip.list_sklearn_estimators('regressor'))} regressors "
          f"via parsnip.sklearn_model(...)\n")

    ames = datasets.load_ames_small(n=600, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    train, test = split.training(), split.testing()

    # ONE recipe, reused for every model -- the point of the abstraction.
    rec = (recipes.Recipe(train, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_other(all_nominal_predictors(), threshold=0.05)
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))

    # A bake-off: curated types (tidy args) + one universal sklearn model.
    contenders = {
        "linear_reg":       parsnip.linear_reg(),
        "rand_forest":      parsnip.rand_forest(trees=300),
        "boost_tree":       parsnip.boost_tree(trees=300, learn_rate=0.05),
        "hist_boost_tree":  parsnip.hist_boost_tree(trees=300),
        "svm_rbf":          parsnip.svm_rbf(cost=10),
        "nearest_neighbor": parsnip.nearest_neighbor(neighbors=7),
        "mlp":              parsnip.mlp(hidden_units=64, epochs=500),
        "KernelRidge(sk)":  parsnip.sklearn_model("KernelRidge", alpha=1.0),
    }

    metrics = yardstick.metric_set(yardstick.rmse, yardstick.rsq, yardstick.mae)
    rows = []
    for name, spec in contenders.items():
        spec = spec.set_mode("regression") if spec.mode is None else spec
        wf = workflows.Workflow().add_recipe(rec).add_model(spec).fit(train)
        res = wf.augment(test)
        m = metrics(res, truth="price", estimate=".pred").set_index("metric")["estimate"]
        rows.append((name, m["rmse"], m["rsq"], m["mae"]))

    rows.sort(key=lambda r: r[1])  # rank by RMSE
    print(f"{'model':18s} {'RMSE':>10s} {'R^2':>7s} {'MAE':>10s}")
    print("-" * 48)
    for name, rmse, rsq, mae in rows:
        print(f"{name:18s} {rmse:10,.0f} {rsq:7.3f} {mae:10,.0f}")
    print(f"\nBest by RMSE: {rows[0][0]}")


if __name__ == "__main__":
    main()
