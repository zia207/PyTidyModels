"""Keras models in PyTidyModels.

Requires keras + a backend:  pip install "pytidymodels[keras]"
(Keras 3 is backend-agnostic; TensorFlow, JAX, or PyTorch all work. If
KERAS_BACKEND isn't set, the engine picks whichever backend is installed.)

Run:  python examples/keras_mlp.py
"""
import warnings

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import all_nominal_predictors, all_numeric_predictors

warnings.filterwarnings("ignore")

try:
    from pytidymodels.parsnip.keras_engine import _require_keras
    _require_keras()
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False


def main():
    if not HAS_KERAS:
        print("Keras (or a backend) is not installed in this environment.")
        print("Install it to run this example:  pip install 'pytidymodels[keras]'")
        print("\nThe spec API still constructs fine without keras, e.g.:")
        print("  spec =", parsnip.mlp(hidden_units=(64, 32), epochs=200)
              .set_engine("keras").set_mode("regression"))
        print("  (fitting it is what requires keras.)")
        return

    ames = datasets.load_ames_small(n=600, seed=42)
    split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
    train, test = split.training(), split.testing()

    # Same recipe as every other model -- keras net is just another engine.
    rec = (recipes.Recipe(train, formula="price ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_impute_mode(all_nominal_predictors())
           .step_dummy(all_nominal_predictors())
           .step_normalize(all_numeric_predictors()))
    metrics = yardstick.metric_set(yardstick.rmse, yardstick.rsq)

    # 1. curated mlp on the keras engine
    mlp = (parsnip.mlp(hidden_units=(64, 32), penalty=1e-4, epochs=200, learn_rate=1e-2)
           .set_engine("keras", seed=0, dropout=0.1).set_mode("regression"))
    wf = workflows.Workflow().add_recipe(rec).add_model(mlp).fit(train)
    res = wf.augment(test)
    print("keras MLP (curated):")
    print(metrics(res, truth="price", estimate=".pred").to_string(index=False))

    # 2. any keras.Model via keras_model()
    import keras
    from keras import layers

    def factory(in_features, out_features):
        return keras.Sequential([
            keras.Input((in_features,)),
            layers.Dense(128, activation="relu"), layers.Dropout(0.1),
            layers.Dense(64, activation="relu"),
            layers.Dense(out_features),
        ])

    custom = parsnip.keras_model(factory, mode="regression", epochs=200, lr=1e-2, seed=0)
    wf2 = workflows.Workflow().add_recipe(rec).add_model(custom).fit(train)
    res2 = wf2.augment(test)
    print("\ncustom keras.Model via keras_model():")
    print(metrics(res2, truth="price", estimate=".pred").to_string(index=False))


if __name__ == "__main__":
    main()
