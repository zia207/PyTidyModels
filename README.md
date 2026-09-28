# PyTidyModels

A **tidymodels-style** modular modeling grammar for Python, built *on top of*
scikit-learn rather than replacing it. Split → preprocess → specify → bundle →
evaluate, with one consistent interface across model families and tidy,
labelled outputs everywhere.

> **Status:** v0.9.0. The core flow — `rsample → recipes → parsnip →
> workflows → yardstick` — runs end to end, with a magrittr-style `>>` pipe.
> `parsnip` ships **20+ curated model types**, **universal scikit-learn access**
> (`sklearn_model`), and **optional PyTorch, Keras, H2O, and survival engines**
> (H2O includes GLM/DRF/GBM/DeepLearning + AutoML; survival via lifelines +
> scikit-survival). Also: **hyperparameter tuning** (`dials` + `tune`), **SHAP
> interpretation** (`explain`), **survival analysis** (Cox / AFT / random
> survival forest / boosted), and **14 bundled real datasets**. Post-processing
> (`tailor`), tidy model summaries (`broom`), Bayesian tuning, and workflow
> sets remain for later phases.

## Install

```bash
pip install -e .            # from a clone
# optional engines / tooling:
pip install -e ".[boost]"   # xgboost, lightgbm
pip install -e ".[torch]"   # PyTorch engine (mlp + torch_model)
pip install -e ".[keras]"   # Keras engine (mlp + keras_model), TF backend
pip install -e ".[shap]"    # SHAP interpretation
pip install -e ".[survival]" # survival analysis (lifelines + scikit-survival)
pip install -e ".[h2o]"     # H2O engine (GLM/DRF/GBM/DeepLearning + AutoML; needs Java)
pip install -e ".[dev]"     # pytest, ruff, mypy
```

Requires Python ≥ 3.10, and numpy / pandas / scikit-learn / scipy.

## Quickstart

```python
from pytidymodels import datasets, rsample, recipes, parsnip, workflows, yardstick
from pytidymodels.recipes import all_numeric_predictors, all_nominal_predictors

ames = datasets.load_ames_small()

# 1. split (stratified on the outcome)
split = rsample.initial_split(ames, prop=0.8, strata="price", seed=1)
train, test = split.training(), split.testing()

# 2. preprocess — a readable, role-aware grammar
rec = (recipes.Recipe(train, formula="price ~ .")
       .step_impute_median(all_numeric_predictors())
       .step_impute_mode(all_nominal_predictors())
       .step_other(all_nominal_predictors(), threshold=0.05)
       .step_dummy(all_nominal_predictors())
       .step_normalize(all_numeric_predictors()))

# 3. specify — engine-agnostic; swap "sklearn" → "lightgbm" without other changes
rf = (parsnip.rand_forest(trees=300, min_n=5)
      .set_engine("sklearn", random_state=0)
      .set_mode("regression"))

# 4. bundle + fit
wf = workflows.Workflow().add_recipe(rec).add_model(rf).fit(train)

# 5. predict (tidy frame) + evaluate
results = wf.augment(test)
metrics = yardstick.metric_set(yardstick.rmse, yardstick.rsq, yardstick.mae)
print(metrics(results, truth="price", estimate=".pred"))
```

Run the demos:

```bash
python examples/hello_world.py     # regression + classification, end to end
python examples/model_catalog.py   # a bake-off across many models, one recipe
python examples/pipe_flow.py       # the same flow written with the >> pipe
python examples/torch_mlp.py       # deep nets via the PyTorch engine (needs torch)
python examples/keras_mlp.py       # deep nets via the Keras engine (needs keras)
python examples/tuning.py          # grid tuning over resamples with dials + tune
python examples/shap_explain.py    # SHAP feature importance + summary plot (needs shap)
python examples/survival.py        # Cox/AFT/RSF/boosted survival models (needs survival extra)
python examples/h2o_models.py      # H2O GLM/DRF/GBM/DeepLearning + AutoML (needs h2o + Java)
python examples/datasets_demo.py   # browse + model on the bundled real datasets
```

## Model catalog

Two ways to specify a model, both fitting the same `Workflow`:

**1. Curated types** — tidy, engine-neutral argument names (translated to the
right scikit-learn kwargs under the hood):

| Family | Types |
|---|---|
| Trees & ensembles | `rand_forest`, `extra_trees`, `boost_tree`, `hist_boost_tree`, `decision_tree`, `bag_tree`, `adaboost` |
| Linear | `linear_reg`, `logistic_reg`, `multinom_reg`, `poisson_reg` |
| Kernel / distance / net | `svm_linear`, `svm_rbf`, `svm_poly`, `nearest_neighbor`, `mlp` |
| Probabilistic / discriminant / GP | `naive_Bayes`, `discrim_linear`, `discrim_quad`, `gaussian_process` |

```python
parsnip.rand_forest(mtry=4, trees=500, min_n=5).set_engine("sklearn").set_mode("regression")
parsnip.boost_tree(trees=300, tree_depth=4, learn_rate=0.05).set_mode("classification")
parsnip.svm_rbf(cost=10, rbf_sigma=0.1).set_mode("classification")
```

Tidy → sklearn argument translation, e.g. `rand_forest`: `mtry→max_features`,
`trees→n_estimators`, `min_n→min_samples_leaf`; `boost_tree`:
`learn_rate→learning_rate`, `tree_depth→max_depth`, `sample_size→subsample`;
`nearest_neighbor`: `neighbors→n_neighbors`; `svm_*`: `cost→C`,
`rbf_sigma→gamma`.

**2. Any scikit-learn estimator** — for the long tail, wrap it directly
(native kwargs, mode inferred):

```python
parsnip.sklearn_model("HistGradientBoostingRegressor", max_iter=300)
parsnip.sklearn_model("RidgeClassifier")                  # by name
from sklearn.linear_model import HuberRegressor
parsnip.sklearn_model(HuberRegressor, epsilon=1.5)        # by class
parsnip.sklearn_model(RandomForestRegressor(n_estimators=800))  # by instance

parsnip.list_models()                       # curated type names
parsnip.list_sklearn_estimators("regressor")  # every sklearn regressor
```

Between the two, **all scikit-learn models** are reachable through the same
tidy `split → recipe → spec → workflow → metrics` flow.

**3. PyTorch** (optional — `pip install "pytidymodels[torch]"`).

PyTorch isn't an enumerable catalog of models like scikit-learn — it's a
framework whose models are `nn.Module` graphs. So "PyTorch support" here means
a torch **engine** for neural nets plus a generic wrapper for any module:

```python
# curated mlp on the torch engine (tidy args translate to the training loop)
parsnip.mlp(hidden_units=(64, 32), penalty=1e-4, epochs=300, learn_rate=1e-2) \
       .set_engine("torch", seed=0, dropout=0.1).set_mode("regression")

# wrap ANY nn.Module -- yours, torchvision's, timm's, torch.hub's
import torch.nn as nn
def factory(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(),
                         nn.Linear(128, out_features))
parsnip.torch_model(factory, mode="regression", epochs=300, lr=1e-2)
parsnip.torch_model(nn.Sequential(nn.Linear(10, 1)), mode="regression")

parsnip.list_torch_models()   # torchvision architectures, if installed
```

The torch estimator handles batching, the optimizer (Adam), the loss (MSE /
cross-entropy by mode), device, and tidy predictions, so a deep net is just
another model in the same `Workflow`. `mode` must be given for `torch_model`
(it can't be inferred from a bare module). torch is fully optional: the
package imports and runs without it, and a torch-backed spec only errors — with
an install hint — if you fit it without torch present.

**4. Keras** (optional — `pip install "pytidymodels[keras]"`).

Same pattern as PyTorch. Keras 3 is backend-agnostic (TensorFlow, JAX, or
PyTorch); if `KERAS_BACKEND` isn't set, the engine picks whichever backend is
installed.

```python
parsnip.mlp(hidden_units=(64, 32), penalty=1e-4, epochs=200) \
       .set_engine("keras", seed=0, dropout=0.1).set_mode("regression")

import keras
def factory(in_features, out_features):
    return keras.Sequential([keras.Input((in_features,)),
                             keras.layers.Dense(64, activation="relu"),
                             keras.layers.Dense(out_features)])
parsnip.keras_model(factory, mode="regression", epochs=200)
parsnip.keras_model(keras.Sequential([keras.layers.Dense(1)]), mode="regression")

parsnip.list_keras_applications()   # keras.applications image models
```

An uncompiled model is compiled with a mode-appropriate loss; an
already-compiled one keeps its own. As with torch, `mode` is required for
`keras_model`, and keras is fully optional.

**5. H2O** (optional — `pip install "pytidymodels[h2o]"`, plus a Java runtime).

H2O trains on a local Java cluster. Configure it once with
`parsnip.h2o_init(...)` (it also starts lazily on first fit) and stop it with
`parsnip.h2o_shutdown()`. Use `.set_engine("h2o")` on the curated types, or run
AutoML:

```python
parsnip.h2o_init(max_mem_size="4G")                       # optional, once

parsnip.rand_forest(trees=200).set_engine("h2o").set_mode("regression")   # DRF
parsnip.boost_tree(trees=200, tree_depth=4, learn_rate=0.1) \
       .set_engine("h2o").set_mode("classification")                       # GBM
parsnip.linear_reg(penalty=0.1, mixture=0.5).set_engine("h2o")             # GLM
parsnip.mlp(hidden_units=[64, 32]).set_engine("h2o")                       # DeepLearning

# AutoML: search + stacked ensembles, predict with the leader
aml = parsnip.h2o_automl(mode="regression", max_models=20, sort_metric="RMSE")
wf = workflows.Workflow().add_recipe(rec).add_model(aml).fit(train)
wf.extract_fit_engine().leaderboard()      # the AutoML leaderboard
wf.predict(test)                           # leader-model predictions
```

`linear_reg`/`logistic_reg` → GLM, `rand_forest` → DRF, `boost_tree` → GBM,
`mlp` → DeepLearning. Predictions are tidy like every other engine; H2O handles
the `H2OFrame` conversion and factor encoding internally.

> **A note on "all models".** Neither PyTorch nor Keras is an enumerable set of
> models the way scikit-learn is — they're frameworks for *building* models. So
> "all PyTorch/Keras models" is delivered as an engine for `mlp` plus a generic
> wrapper (`torch_model` / `keras_model`) that adapts any module you build or
> import. The `list_torch_models()` / `list_keras_applications()` catalogs are
> the vision model zoos (torchvision / keras.applications), which expect image
> inputs and aren't meant for the tabular pipeline.

## What's in the slice

| Module | Role | Key API |
|---|---|---|
| `hardhat` | internal scaffolding | `ColumnInfo`, `Blueprint`, `parse_formula`, base classes |
| `rsample` | splitting & resampling | `initial_split`, `vfold_cv` |
| `recipes` | preprocessing grammar | `Recipe`, `step_*`, `all_*_predictors()` selectors |
| `parsnip` | unified model spec | 20+ curated types + `sklearn_model()`; PyTorch / Keras / H2O (GLM/DRF/GBM/DL + AutoML) / survival engines; `.set_engine`, `.set_mode` |
| `workflows` | bundle recipe + model | `Workflow`, `.fit`, `.predict`, `.augment`, `.extract_fit_engine` |
| `yardstick` | metrics, tidy output | `metric_set`, `rmse`, `rsq`, `accuracy`, `roc_auc`, `concordance_survival`, `survival_metrics`, … |
| `dials` | tuning params + grids | `trees`, `penalty`, … + `grid_regular` / `grid_random` / `grid_latin_hypercube` |
| `tune` | tuning over resamples | `tune_grid`, `show_best`, `select_best`, `finalize_workflow`, `last_fit` |
| `explain` | SHAP interpretation | `explain`, `vip`, `Explanation.importance()` / `.summary_plot()` |
| `datasets` | sample + real data | `list_datasets`, `load_dataset`, `load_boston` / `load_pima` / `load_lung` / … (14 bundled) + synthetic `load_ames_small` / `load_credit_small` |

## Bundled datasets

Fourteen real datasets ship with the package (gzip-compressed, ~1.5 MB total),
spanning the task types the modeling API supports. `list_datasets()` gives the
overview; each `load_*` returns a tidy DataFrame.

```python
from pytidymodels import datasets

datasets.list_datasets()          # task / outcome / shape / source per dataset
df = datasets.load_pima()         # or datasets.load_dataset("pima")
```

| Dataset | Task | Outcome | Rows × Cols |
|---|---|---|---|
| `boston` | regression | `medv` | 506 × 14 |
| `soil` | spatial regression | `SOC` | 467 × 19 |
| `soil_simulated` | regression | `SOC` | 3000 × 9 |
| `county_diabetes` | spatial regression | `Diabetes_per` | 3107 × 15 |
| `air_pollution` | spatial regression (panel) | `RATE` | 46605 × 13 |
| `airline_costs` | regression | `C` | 90 × 6 |
| `flights` | regression | `arr_delay` | 8000 × 19 |
| `taxi` | regression | `total_amount` | 8000 × 12 |
| `pima` | classification | `Outcome` | 768 × 9 |
| `breast_cancer` | classification | `status` | 198 × 33 |
| `nmes` | count | `visits` | 4406 × 13 |
| `health_insurance` | count | `absent` | 1448 × 6 |
| `lung` | survival | `time`, `status` | 228 × 10 |
| `melanoma` | survival | `time`, `status` | 205 × 7 |

The large sources (NYC taxi, flights) are cleaned and subsampled to a teaching
size with a fixed seed; the rest ship whole. Datasets keep their original
columns (missing values intact, e.g. `lung`) so the recipe steps have something
real to do. Classic datasets (`boston`, `lung`, `melanoma`, `nmes`, `pima`,
`flights`, `taxi`) are widely-used teaching data included for that purpose; the
soil, county-diabetes, and air-pollution sets come from the author's own work.
See `list_datasets()` for per-dataset provenance. The bundle is reproducible:
`tools/build_datasets.py` regenerates every `.csv.gz` from the raw sources
(cleaning, feature engineering, and seeded subsampling all in one script).

## Hyperparameter tuning

Mark arguments with `parsnip.tune()`, build a grid with `dials`, and evaluate
every candidate across resamples with `tune.tune_grid`:

```python
from pytidymodels import dials, tune

spec = (parsnip.boost_tree(trees=parsnip.tune(), learn_rate=parsnip.tune(),
                           tree_depth=parsnip.tune())
        .set_mode("regression"))
wf = workflows.Workflow().add_recipe(rec).add_model(spec)

folds = rsample.vfold_cv(train, v=5, strata="price", seed=2)
grid = dials.grid_latin_hypercube(
    dials.trees(range=(100, 600)),
    dials.learn_rate(range=(1e-2, 0.3)),
    dials.tree_depth(range=(2, 6)),
    size=12, seed=1,
)

res  = tune.tune_grid(wf, folds, grid,
                      yardstick.metric_set(yardstick.rmse, yardstick.rsq))
best = tune.select_best(res, "rmse")           # or show_best / select_by_one_std_err
final = tune.finalize_workflow(wf, best)
lf = tune.last_fit(final, split, yardstick.metric_set(yardstick.rmse, yardstick.rsq))
lf.metrics        # test-set performance of the tuned model
```

Grids: `grid_regular` (evenly spaced cartesian), `grid_random`, and
`grid_latin_hypercube` (space-filling). Parameters carry sensible default
ranges and know whether they're integer-valued or log-scaled (`penalty`,
`learn_rate`). Each fold's recipe is prepped on its own analysis set, so
assessment scores are leakage-free. Results are tidy — `collect_metrics`
returns one row per candidate × metric with `mean` / `n` / `std_err`, and
`show_best` / `select_best` respect each metric's direction (minimize `rmse`,
maximize `roc_auc`).

## Interpretation (SHAP)

`explain` computes SHAP values for a fitted workflow on the **baked** design
matrix (the columns the model actually sees). Tree models use the fast exact
`TreeExplainer`; anything else falls back to a model-agnostic explainer.

```python
from pytidymodels import explain

wf = workflows.Workflow().add_recipe(rec).add_model(
        parsnip.rand_forest(trees=300).set_mode("regression")).fit(train)

exp = explain.explain(wf, test, seed=0)
exp.importance()                 # mean |SHAP| per feature, tidy + sorted
exp.to_frame(long=True)          # row / feature / shap_value
exp.summary_plot("shap.png")     # beeswarm summary

explain.vip(wf, test)            # shortcut for the importance table
```

For classification, values explain the positive (last, sorted) class —
matching the `roc_auc` convention. SHAP is optional
(`pip install "pytidymodels[shap]"`).

## Survival analysis (censored)

Survival (time-to-event) models use a **two-column outcome** `(time, event)`,
where `event` is `1` for an observed event and `0` for right-censoring, and the
mode is `"censored regression"` (aliases: `"survival"`, `"censored"`). Two
backends sit behind one interface: **lifelines** (Cox, parametric AFT) and
**scikit-survival** (Cox, random survival forest, gradient boosting).

```python
from pytidymodels import parsnip, recipes, rsample, workflows, yardstick, datasets

lung = datasets.load_lung().copy()
lung["event"] = (lung["status"] == 2).astype(int)     # recode to 0/1

split = rsample.initial_split(lung, prop=0.8, seed=1)
train, test = split.training(), split.testing()

rec = (recipes.Recipe(train, formula="time + event ~ .")   # time + event
       .step_impute_median(recipes.all_numeric_predictors())
       .step_normalize(recipes.all_numeric_predictors()))

cox = parsnip.proportional_hazards(penalty=0.1).set_mode("censored regression")
wf = workflows.Workflow().add_recipe(rec).add_model(cox).fit(train)

wf.predict(test)                       # .pred_time, .pred_risk
wf.predict_survival(test, [180, 365])  # S(t): .pred_surv_180, .pred_surv_365
yardstick.survival_metrics(wf, train, test)   # concordance, IPCW C, IBS, AUC
```

Model types: `proportional_hazards(penalty, mixture)` (Cox — lifelines or
`sksurv`), `survival_reg(dist="weibull"|"lognormal"|"loglogistic")` (parametric
AFT — lifelines), and `rand_forest` / `boost_tree` / `decision_tree` with
`.set_engine("sksurv").set_mode("censored regression")` (random survival forest,
gradient-boosted, survival tree). Predictions are tidy — `.pred_time`,
`.pred_risk`, and `S(t)` columns; metrics cover Harrell's and Uno's concordance,
integrated Brier score, and mean time-dependent AUC (censoring-aware, so
`survival_metrics` needs the training frame). The bundled `lung` and `melanoma`
datasets are ready to use. Survival is optional
(`pip install "pytidymodels[survival]"`).

## The pipe operator (`>>`)

Python has no user-definable `%>%` token — you can only overload existing
operators — so, like dfply / siuba / plydata, PyTidyModels overloads `>>` with
[magrittr](https://magrittr.tidyverse.org/) semantics: the left value becomes
the **first argument**, and `_` is the placeholder (magrittr's `.`).

```python
from pytidymodels.pipe import pipe, _

5 >> pipe(pow, 2)            # pow(5, 2)  -> 25
2 >> pipe(pow, _, 10)        # pow(2, 10) -> 1024   (placeholder positions the value)
"  hi " >> pipe(str.strip) >> pipe(str.upper)      # "HI"

# reusable composed pipeline
clean = pipe(str.strip) >> pipe(str.title)
"  ada lovelace  " >> clean  # "Ada Lovelace"
```

It works with **any** callable — the package's verbs, pandas, sklearn, or a
lambda — so nothing needs decorating. In a modeling flow:

```python
train = ames >> pipe(rsample.initial_split, prop=0.8, strata="price") >> pipe(rsample.training)

out = (wf
       >> pipe(workflows.augment, test)
       >> pipe(score, truth="price", estimate=".pred"))
```

`rsample.training/testing/analysis/assessment` and `workflows.fit/predict/augment`
have free-function forms (as in R) for exactly this. Builders also carry a
pandas-style `.pipe(func, *args)` for threading into arbitrary functions.

**Method chaining stays the primary idiom** (`recipe.step_x().step_y()`,
`workflow.add_recipe(...).add_model(...).fit(...)`): it gives autocomplete and
type-checking that operator magic can't. The pipe is here for the tidyverse
feel and for threading values through non-method functions.



- **Wrap, don't reimplement.** Recipes compile to scikit-learn transformers and
  workflows to a real `sklearn.Pipeline` (`Workflow.to_sklearn_pipeline`), so
  SHAP, calibration, and MLflow logging still work. `extract_fit_engine()` hands
  back the raw estimator.
- **Spec ≠ engine.** A `parsnip` spec declares *what*; `.set_engine()` binds
  *how*. The engine-translation layer maps tidy argument names (`mtry`, `trees`,
  `min_n`, `penalty`, `mixture`) onto each backend's native kwargs.
- **Tidy in, tidy out.** Predictions are always labelled frames (`.pred`;
  `.pred_class` + per-level `.pred_{level}`); metrics return
  `metric / estimator / estimate` rows.
- **Selectors resolve live.** `all_numeric_predictors()` is evaluated against the
  schema *as it exists at each step*, so it correctly picks up dummy columns
  created upstream.
- **Immutable builders.** Every `set_*` / `step_*` returns a new object.

## R ↔ Python name map

`rsample → rsample`, `recipes → recipes`, `parsnip → parsnip`,
`workflows → workflows`, `yardstick → yardstick`, `hardhat → hardhat`.
Plain-English aliases are available too: `tm.resampling`, `tm.preprocessing`,
`tm.models`, `tm.metrics`.

## Tests

```bash
pytest -q
```

## License

MIT.
