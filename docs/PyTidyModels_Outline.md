# PyTidyModels — Design & Build Outline

A Python meta-package that brings the **tidymodels** philosophy — a consistent, modular, composable grammar for the whole modeling lifecycle — to the Python ecosystem, built *on top of* scikit-learn rather than replacing it.

---

## 1. Vision & Scope

**Goal.** Give Python users the tidymodels experience: split → preprocess → specify → bundle → tune → evaluate → post-process → tidy, with one consistent interface across model families and a unified return grammar (tidy DataFrames everywhere).

**The gap being filled.** Python already has scikit-learn's `Pipeline`/`fit`/`predict` and strong estimators, but it lacks:
- A **unified model spec** decoupled from the engine (so you swap `random_forest` from sklearn → xgboost → lightgbm without rewriting code).
- A **readable preprocessing grammar** with role-aware, selector-based steps (vs. hand-wiring `ColumnTransformer`).
- **Tidy, predictable outputs** — every result is a labeled DataFrame, not a bare NumPy array.
- A **resampling-first** evaluation model where cross-validation, metrics, and tuning share one vocabulary.

**Existing partial efforts to acknowledge/absorb:** `pybroom` and `sktidy` (tidy/augment/glance for a few sklearn models), `scikit-lego`, `feature-engine` (rich transformers). PyTidyModels should interoperate with these, not duplicate them.

**Non-goals (v1).** Not a new numerical/estimator library. Not a deep-learning framework. It orchestrates existing engines (sklearn, statsmodels, xgboost, lightgbm, glum) behind a tidy interface.

---

## 2. Design Philosophy — tidy principles translated to Python

| tidymodels principle | Python translation |
|---|---|
| Consistent verbs across packages | Consistent method/function names; predictable signatures |
| Tidy data in, tidy data out | pandas (optionally polars) DataFrames in and out; predictions as labeled frames |
| Composability via `%>%`/`|>` | Method chaining + a light pipe helper; objects are immutable, chaining returns new objects |
| Separation of concerns | One sub-module per lifecycle stage; each usable alone |
| Model spec ≠ engine | A spec object declares *what*; `.set_engine()` binds *how* |
| Reproducibility | Explicit `seed`/`random_state` threading; deterministic splits |
| Sensible defaults, no silent surprises | Loud, informative errors; no hidden imputation or coercion |

**Interop-first stance:** a `workflow` compiles down to a real `sklearn.Pipeline` under the hood, so anything sklearn-compatible (calibration, `cross_val_predict`, SHAP, MLflow logging) still works.

---

## 3. Architecture Overview

Single installable distribution, `pytidymodels`, exposing sub-modules that mirror the tidymodels core packages. Loaded together via a convenience import, or à la carte.

```python
import pytidymodels as tm          # meta-import (attaches submodules)
# or à la carte:
from pytidymodels import rsample, recipes, parsnip, workflows, tune, yardstick
```

**Dependency layering (bottom → top):**

```
hardhat        ← internal scaffolding: molds/blueprints, role/type handling, spec base classes
  │
  ├── rsample        (splitting & resampling)
  ├── recipes        (preprocessing grammar → sklearn-compatible transformer)
  ├── parsnip        (unified model spec; engines wrap sklearn/xgboost/…)
  ├── dials          (parameter objects + grids)
  ├── yardstick      (metrics; tidy output)
  │
  ├── workflows      (bundle recipe + spec + post-processor)  ─ depends on recipes/parsnip
  ├── tune           (tune workflows over resamples + grids)  ─ depends on rsample/dials/yardstick/workflows
  ├── tailor         (post-processing: thresholds, calibration)
  │
  ├── workflowsets   (many workflows at once)                 ─ depends on workflows/tune
  ├── broom          (tidy / augment / glance on fitted objects)
  └── infer          (statistical inference grammar — optional/later)
```

**Naming decision:** keep the tidymodels sub-module names (`rsample`, `parsnip`, …) for muscle-memory transfer from R users, but expose plain-English aliases (`resampling`, `models`, `preprocessing`, `metrics`) for Python-native users who find the vegetable names opaque.

---

## 4. Module-by-Module Breakdown

Each section lists the tidymodels counterpart, the responsibility, the proposed API surface, and the Python engine(s) it leans on.

### 4.1 `hardhat` (internal infrastructure) ← *hardhat*
The developer-facing scaffolding nearly every other module depends on.
- **Blueprints / molds:** capture the schema (column names, dtypes, roles) at fit time; replay at predict time to guarantee train/predict column alignment.
- **Roles:** tag columns as `outcome`, `predictor`, `id`, `case_weight`, `stratify`, `group`.
- **Base classes:** `ModelSpec`, `PreprocStep`, `Metric`, `Resampler` — abstract contracts enforcing the consistent interface.
- **Formula interface (optional):** a `patsy`/`formulaic`-backed `"y ~ x1 + x2"` parser, so R users keep formulas while Python users can pass explicit `x, y`.

### 4.2 `rsample` — Splitting & Resampling ← *rsample*
- `initial_split(df, prop=0.75, strata=None)` → `Split` object; `.training()`, `.testing()`.
- `initial_validation_split()` for train/val/test.
- Resamplers returning a **resample set** (a DataFrame of fold objects, tidy): `vfold_cv(v=10, repeats=1, strata=)`, `bootstraps()`, `mc_cv()`, `loo_cv()`, `group_vfold_cv()`, `rolling_origin()`/`sliding_window()` for time series.
- `analysis()` / `assessment()` accessors per fold (mirrors tidymodels' train/holdout naming).
- Engine: wrap `sklearn.model_selection` splitters, but return labeled, inspectable objects.
- **Extension hook:** `spatialsample`-style spatial CV (block/cluster) for Zia's geospatial-epi work.

### 4.3 `recipes` — Preprocessing Grammar ← *recipes*
The signature feature. A `Recipe` is declared, then `.prep()` (learn from training data), then `.bake()` (apply).
- `Recipe(data, formula=... | roles=...)`.
- **Selectors:** `all_predictors()`, `all_numeric()`, `all_nominal()`, `all_outcomes()`, `has_role()`, `starts_with()`, `matches()`.
- **Steps (chainable):** `.step_normalize()`, `.step_dummy()`, `.step_impute_median()`, `.step_impute_knn()`, `.step_pca()`, `.step_log()`, `.step_other()` (lump rares), `.step_zv()`/`.step_nzv()` (drop zero/near-zero var), `.step_corr()`, `.step_interact()`, `.step_date()`, `.step_holiday()`.
- `.prep()` → fitted recipe; `.bake(new_data)` → transformed DataFrame.
- **Critical design:** a `Recipe` also exposes `.to_sklearn_transformer()` so it drops straight into any sklearn `Pipeline` — this is what makes the whole system interoperable.
- **Extension hooks:** `textrecipes` (text features), `themis` (SMOTE/imbalance via `imbalanced-learn`), `embed` (target/likelihood encoding).

### 4.4 `parsnip` — Unified Model Specification ← *parsnip*
Declare the model abstractly, bind an engine, set the mode.
- Model generators: `linear_reg()`, `logistic_reg()`, `multinom_reg()`, `rand_forest()`, `boost_tree()`, `decision_tree()`, `nearest_neighbor()`, `svm_rbf()`, `svm_linear()`, `mlp()`, `mars()`, `bag_tree()`.
- `.set_engine("sklearn" | "xgboost" | "lightgbm" | "statsmodels" | "glum")`.
- `.set_mode("regression" | "classification" | "censored regression")`.
- Tunable placeholders: pass `tune()` as an argument value to defer to the tuner, e.g. `rand_forest(mtry=tune(), trees=1000)`.
- `.fit(formula_or_xy, data)` → fitted spec with a **standardized `.predict()`** that always returns a tidy DataFrame: regression → `.pred`; classification → `.pred_class`, `.pred_{level}` probabilities.
- **Engine translation layer:** maps tidymodels-style argument names (`mtry`, `trees`, `min_n`, `penalty`, `mixture`) to each engine's native kwargs — the single most valuable abstraction in the package.

### 4.5 `dials` — Tuning Parameters & Grids ← *dials*
- Parameter objects with ranges/scales: `mtry()`, `trees()`, `min_n()`, `penalty(log10)`, `mixture()`, `learn_rate(log10)`, `tree_depth()`.
- `finalize()` to resolve data-dependent ranges (e.g. `mtry` upper bound = #predictors).
- Grid builders: `grid_regular(n=)`, `grid_random(size=)`, `grid_latin_hypercube()`, `grid_max_entropy()` → tidy grid DataFrames.
- Engine: wrap `scipy.stats` distributions and `scikit-optimize`/`optuna` for space-filling designs.

### 4.6 `yardstick` — Metrics ← *yardstick*
- Every metric is a callable with a consistent signature and **tidy output**: `accuracy(df, truth, estimate)` → one-row DataFrame `.metric / .estimator / .estimate`.
- **Metric sets:** `metric_set(rmse, rsq, mae)` → apply many at once, grouped-aware.
- Classification: `accuracy`, `roc_auc`, `pr_auc`, `f_meas`, `sensitivity`, `specificity`, `mcc`, `brier_class`; curves `roc_curve()`, `pr_curve()`, `gain_curve()` returning plot-ready frames.
- Regression: `rmse`, `rsq`, `mae`, `mape`, `ccc`, `huber_loss`.
- Group semantics: respects a grouped DataFrame so per-resample/per-class metrics fall out naturally.
- Engine: thin, correctness-checked wrappers over `sklearn.metrics` with tidy shaping.

### 4.7 `workflows` — Bundle Everything ← *workflows*
- `Workflow().add_recipe(rec).add_model(spec).add_tailor(post)`.
- Alternatively `.add_formula()` / `.add_variables()` instead of a recipe.
- `.fit(data)` compiles to an `sklearn.Pipeline` internally, fits end-to-end, returns a fitted workflow.
- `.predict(new_data)`, `.extract_recipe()`, `.extract_fit_parsnip()`, `.extract_fit_engine()` (escape hatch to the raw sklearn/xgboost object).
- This is the primary object users pass to `tune` and `workflowsets`.

### 4.8 `tune` — Hyperparameter Optimization ← *tune*
- `tune_grid(workflow, resamples, grid, metrics)` → tidy tuning results (one row per param × fold × metric).
- `tune_bayes()` (Bayesian opt via optuna/skopt), plus a `finetune`-style `tune_race_anova()` for early stopping of bad configs.
- Result verbs: `collect_metrics()`, `collect_predictions()`, `show_best()`, `select_best()`, `select_by_one_std_err()`.
- `finalize_workflow(wf, best_params)` → `last_fit(split)` to fit on full train + evaluate on test in one call.
- Engines: `rsample` folds + `dials` grids + `joblib`/`ray` for parallel backends.

### 4.9 `tailor` — Post-Processing ← *tailor* (newer core package)
- Adjust predictions *after* the model: `.adjust_probability_threshold()`, `.adjust_numeric_range()`, probability **calibration** (`adjust_probability_calibration("isotonic"|"platt")`).
- Learns from a calibration/validation split so it's leakage-safe; attaches to a `workflow`.
- Engine: `sklearn.calibration`, custom threshold logic.

### 4.10 `workflowsets` — Many Workflows at Once ← *workflowsets*
- `workflow_set(preprocessors={...}, models={...})` → cross-product of recipes × specs as a tidy frame.
- `.workflow_map("tune_grid", resamples=, grid=)` fits/tunes all combinations.
- `rank_results()`, `autoplot()`-style comparison across the whole set — the "bake-off" tool.

### 4.11 `broom` — Tidy Model Outputs ← *broom* (absorb `pybroom`/`sktidy`)
- `tidy(fit)` → per-term estimates (coef, std err, statistic, p-value) as a DataFrame.
- `glance(fit)` → one-row model summary (r², AIC, deviance, n).
- `augment(fit, data)` → original data + `.fitted`, `.resid`, `.pred`, `.pred_prob`.
- Registrable dispatch so new engines can add tidiers.

### 4.12 `infer` — Statistical Inference (later phase) ← *infer*
- Grammar: `specify() → hypothesize() → generate() → calculate()` for permutation/bootstrap tests and CIs, returning tidy frames. Optional v2+; valuable for Zia's epi/biostat audience.

---

## 5. Core Design Decisions (resolve early)

1. **DataFrame backbone:** pandas as default "tibble"; design the internal contract against a narrow interface so **polars** and Arrow can be added without rewrites. Consider `narwhals` for backend-agnostic DataFrame ops.
2. **Immutability & chaining:** every builder method returns a *new* object (copy-on-write); no in-place mutation. Provides a pipe helper `tm.pipe(df, f, g, ...)` for users who dislike deep method chains.
3. **sklearn as the substrate, not a competitor:** recipes → transformers, workflows → Pipelines. Never fork estimator math; wrap it.
4. **Formula optional:** support both `"y ~ ."` (via `formulaic`) and explicit `x, y`. Don't force either camp.
5. **Consistent prediction contract:** predictions are *always* labeled DataFrames with reserved `.pred*` column names — the invariant everything downstream relies on.
6. **Loud errors:** validate roles/dtypes at `prep`/`fit`; never silently impute or drop. Rich, actionable messages (à la `rlang`/`cli`).
7. **Reproducibility:** a single `seed=` threads through splits, grids, and stochastic engines.
8. **Extensibility via registries:** engines, recipe steps, metrics, and tidiers all register through decorators so third parties extend without patching core.

---

## 6. Repository Structure

```
pytidymodels/
├── pyproject.toml
├── README.md
├── src/pytidymodels/
│   ├── __init__.py            # meta-import, version, aliases
│   ├── hardhat/               # blueprints, roles, base classes, formula parser
│   ├── rsample/
│   ├── recipes/               # recipe.py, steps/, selectors.py
│   ├── parsnip/               # specs/, engines/ (sklearn.py, xgboost.py, …)
│   ├── dials/
│   ├── yardstick/             # classification.py, regression.py, curves.py
│   ├── workflows/
│   ├── tune/                  # grid.py, bayes.py, race.py, collect.py
│   ├── tailor/
│   ├── workflowsets/
│   ├── broom/
│   ├── infer/                 # later
│   └── datasets/              # modeldata analog (ames, penguins, credit)
├── tests/                     # mirrors src/ tree
├── docs/                      # mkdocs-material + notebooks
└── examples/                  # end-to-end case studies
```

---

## 7. Tech Stack & Tooling

- **Build/packaging:** `pyproject.toml` with **hatchling** (or `uv`); publish to PyPI; `src/` layout.
- **Core deps:** numpy, pandas, scikit-learn, scipy. **Optional extras:** `[boost]` xgboost+lightgbm, `[stats]` statsmodels+formulaic, `[tune]` optuna, `[polars]` polars+narwhals, `[imbalance]` imbalanced-learn.
- **Lint/format/type:** `ruff` + `mypy` (typed public API, ship `py.typed`).
- **Testing:** `pytest`, `hypothesis` (property tests on transforms), coverage gate.
- **Docs:** `mkdocs-material` + `mkdocstrings` for API, executable notebook tutorials mirroring the tidymodels "Learn" articles.
- **CI/CD:** GitHub Actions — matrix over Python 3.10–3.13 and OSes; auto-publish on tag; `pre-commit`.

---

## 8. Development Roadmap (phased)

**Phase 0 — Foundations.** `hardhat` (blueprints, roles, base classes), DataFrame contract, project scaffolding, CI, docs skeleton.

**Phase 1 — Minimum Viable Workflow (the "hello world" must run end-to-end):**
`rsample.initial_split` + `vfold_cv` → `recipes` (normalize, dummy, impute) → `parsnip` (linear_reg, logistic_reg, rand_forest on sklearn) → `workflows` → `yardstick` (core metrics). This proves the architecture.

**Phase 2 — Tuning.** `dials` params/grids + `tune.tune_grid` + `collect/select` verbs + `finalize`/`last_fit`. Add xgboost/lightgbm engines to `parsnip`.

**Phase 3 — Comparison & post-processing.** `workflowsets`, `tailor` (thresholds + calibration), `tune_bayes`, racing.

**Phase 4 — Tidy outputs & breadth.** `broom` (tidy/glance/augment), statsmodels engine, more recipe steps, `datasets`.

**Phase 5 — Ecosystem.** `infer`, spatial resampling, text/imbalance recipe extensions, survival (`censored` analog), parallel backends (ray), MLflow integration.

---

## 9. Target End-to-End API (design proof)

The whole design is validated if *this* reads cleanly and runs:

```python
import pytidymodels as tm
from pytidymodels import rsample, recipes, parsnip, workflows, tune, dials, yardstick

# 1. split
split = rsample.initial_split(ames, prop=0.8, strata="sale_price")
train, test = split.training(), split.testing()
folds = rsample.vfold_cv(train, v=10, strata="sale_price")

# 2. preprocess
rec = (recipes.Recipe(train, formula="sale_price ~ .")
       .step_log("sale_price", base=10)
       .step_other(recipes.all_nominal_predictors(), threshold=0.01)
       .step_dummy(recipes.all_nominal_predictors())
       .step_normalize(recipes.all_numeric_predictors()))

# 3. specify (tunable)
rf = (parsnip.rand_forest(mtry=tune(), min_n=tune(), trees=1000)
      .set_engine("lightgbm")
      .set_mode("regression"))

# 4. bundle
wf = workflows.Workflow().add_recipe(rec).add_model(rf)

# 5. tune
grid = dials.grid_latin_hypercube(dials.mtry(), dials.min_n(), size=20)
res = tune.tune_grid(wf, resamples=folds, grid=grid,
                     metrics=yardstick.metric_set(yardstick.rmse, yardstick.rsq))

best = tune.select_best(res, metric="rmse")
final = tune.finalize_workflow(wf, best)

# 6. last fit + evaluate
fit = tune.last_fit(final, split)
tune.collect_metrics(fit)          # tidy DataFrame: rmse, rsq on the test set
```

---

## 10. Key Risks & Mitigations

- **Reinventing sklearn.** → Wrap, never reimplement; compile to `Pipeline`.
- **Two-language cognitive load (R names in Python).** → Ship English aliases; document the R↔Python mapping table prominently.
- **DataFrame overhead / dtype friction.** → narwhals abstraction; benchmark against raw sklearn; allow array fast-paths.
- **Engine-argument drift** (each engine renames params). → Centralized translation registry with tests per engine.
- **Scope creep.** → Ship Phases 0–2 as a genuinely useful release before extensions; resist boiling the ocean.
- **Maintenance burden of many engines.** → Registry + optional extras so engines are pluggable and independently tested.

---

## 11. R ↔ Python Mapping (quick reference)

| tidymodels (R) | PyTidyModels (Python) | Underlying Python engine |
|---|---|---|
| rsample | `rsample` / `resampling` | `sklearn.model_selection` |
| recipes | `recipes` / `preprocessing` | custom + `feature-engine` |
| parsnip | `parsnip` / `models` | sklearn, xgboost, lightgbm, statsmodels |
| dials | `dials` / `params` | scipy, optuna |
| tune | `tune` / `tuning` | optuna, joblib/ray |
| yardstick | `yardstick` / `metrics` | `sklearn.metrics` |
| workflows | `workflows` | `sklearn.Pipeline` |
| workflowsets | `workflowsets` | — |
| tailor | `tailor` | `sklearn.calibration` |
| broom | `broom` | statsmodels, custom |
| infer | `infer` | scipy, custom |
| hardhat | `hardhat` (internal) | formulaic/patsy |
