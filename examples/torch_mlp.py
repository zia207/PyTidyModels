"""PyTorch models in PyTidyModels.

Requires torch:  pip install "pytidymodels[torch]"
Run:  python examples/torch_mlp.py
"""
import warnings

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


def main():
    if not HAS_TORCH:
        print("PyTorch is not installed in this environment.")
        print("Install it to run this example:  pip install 'pytidymodels[torch]'")
        print("\nThe spec API still constructs fine without torch, e.g.:")
        print("  spec =", parsnip.mlp(hidden_units=(64, 32), epochs=200)
              .set_engine("torch").set_mode("regression"))
        print("  (fitting it is what requires torch.)")
        print(f"\ntorchvision architectures available: {len(parsnip.list_torch_models())}")
        return

    ames = datasets.load_ames_small(n=600, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    train, test = split.training(), split.testing()

    # Same recipe as every other model -- deep net is just another engine.
    rec = (recipes.Recipe(train, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))

    # 1. curated mlp on the torch engine
    mlp = (parsnip.mlp(hidden_units=(64, 32), penalty=1e-4, epochs=300, learn_rate=1e-2)
           .set_engine("torch", seed=0).set_mode("regression"))
    wf = workflows.Workflow().add_recipe(rec).add_model(mlp).fit(train)
    res = wf.augment(test)
    metrics = yardstick.metric_set(yardstick.rmse, yardstick.rsq)
    print("torch MLP (curated):")
    print(metrics(res, truth="price", estimate=".pred").to_string(index=False))

    # 2. any nn.Module via torch_model()
    import torch.nn as nn

    def factory(in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_features),
        )

    custom = parsnip.torch_model(factory, mode="regression", epochs=300, lr=1e-2, seed=0)
    wf2 = workflows.Workflow().add_recipe(rec).add_model(custom).fit(train)
    res2 = wf2.augment(test)
    print("\ncustom nn.Module via torch_model():")
    print(metrics(res2, truth="price", estimate=".pred").to_string(index=False))


if __name__ == "__main__":
    main()
