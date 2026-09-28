"""H2O models in PyTidyModels.

Requires:  pip install "pytidymodels[h2o]"  (plus a Java 8-21 runtime)
Run:       python examples/h2o_models.py

H2O runs a local Java cluster; configure it with parsnip.h2o_init(...) and stop
it with parsnip.h2o_shutdown().
"""
import warnings

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import all_numeric_predictors

warnings.filterwarnings("ignore")

try:
    import h2o  # noqa: F401
    HAS_H2O = True
except (ImportError, OSError):
    HAS_H2O = False


def main():
    if not HAS_H2O:
        print("H2O needs the h2o package and a Java runtime:")
        print("  pip install 'pytidymodels[h2o]'")
        print("\nSpecs still construct without it, e.g.:")
        print("  spec =", parsnip.boost_tree(trees=200).set_engine("h2o").set_mode("regression"))
        return

    try:
        parsnip.h2o_init(max_mem_size="2G", nthreads=-1)
    except Exception as e:
        print(f"Could not start the H2O cluster (Java installed?): {e}")
        return

    boston = datasets.load_boston()
    split = rsample.initial_split(boston, prop=0.8, seed=1)
    train, test = split.training(), split.testing()
    rec = (recipes.Recipe(train, formula="medv ~ .")
           .step_normalize(all_numeric_predictors()))
    metrics = yardstick.metric_set(yardstick.rmse, yardstick.rsq)

    print(f"{'H2O model':28s} {'RMSE':>8s} {'R^2':>7s}")
    print("-" * 45)
    contenders = {
        "GLM (linear_reg)":   parsnip.linear_reg(penalty=0.0),
        "DRF (rand_forest)":  parsnip.rand_forest(trees=200),
        "GBM (boost_tree)":   parsnip.boost_tree(trees=200, tree_depth=4, learn_rate=0.1),
        "DeepLearning (mlp)": parsnip.mlp(hidden_units=[64, 32], epochs=200),
    }
    for name, spec in contenders.items():
        wf = (workflows.Workflow().add_recipe(rec)
              .add_model(spec.set_engine("h2o").set_mode("regression")).fit(train))
        m = metrics(wf.augment(test), truth="medv", estimate=".pred").set_index("metric")["estimate"]
        print(f"{name:28s} {m['rmse']:8.2f} {m['rsq']:7.3f}")

    # AutoML: let H2O search and ensemble
    print("\nH2O AutoML (max_models=5):")
    aml = parsnip.h2o_automl(mode="regression", max_models=5, seed=1, sort_metric="RMSE")
    wf = workflows.Workflow().add_recipe(rec).add_model(aml).fit(train)
    lb = wf.extract_fit_engine().leaderboard()
    print(lb.head(5).to_string(index=False))
    r2 = yardstick.rsq(wf.augment(test), "medv", ".pred")["estimate"].iloc[0]
    print(f"\nAutoML leader test R^2: {r2:.3f}")

    parsnip.h2o_shutdown()


if __name__ == "__main__":
    main()
