"""SHAP interpretation in PyTidyModels.

Requires shap:  pip install "pytidymodels[shap]"
Run:  python examples/shap_explain.py
"""
import warnings

from pytidymodels import datasets, explain, parsnip, recipes, rsample, workflows
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")

try:
    import shap  # noqa: F401
    HAS_SHAP = True
except (ImportError, OSError):
    HAS_SHAP = False


def main():
    if not HAS_SHAP:
        print("SHAP is not installed. Install it to run this example:")
        print("  pip install 'pytidymodels[shap]'")
        return

    ames = datasets.load_ames_small(n=500, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    train, test = split.training(), split.testing()

    rec = (recipes.Recipe(train, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))
    model = parsnip.rand_forest(trees=300).set_engine("sklearn", random_state=0) \
                   .set_mode("regression")
    wf = workflows.Workflow().add_recipe(rec).add_model(model).fit(train)

    # explain the fitted model on the test set (SHAP runs on baked features)
    exp = explain.explain(wf, test, seed=0)
    print(exp)

    print("\nGlobal feature importance (mean |SHAP|):")
    print(exp.importance().to_string(index=False))

    # a per-observation view: the SHAP values for the first test row
    contrib = exp.to_frame(long=True)
    row0 = (contrib[contrib["row"] == 0]
            .reindex(contrib[contrib["row"] == 0]["shap_value"].abs()
                     .sort_values(ascending=False).index)
            .head(5))
    print("\nTop contributions for the first test observation:")
    print(row0.to_string(index=False))

    path = exp.summary_plot(path="shap_summary.png")
    print(f"\nBeeswarm summary plot written to: {path}")


if __name__ == "__main__":
    main()
