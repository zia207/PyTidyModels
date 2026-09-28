"""A complete survival-analysis workflow in PyTidyModels.

Uses the bundled NCCTG `lung` dataset: split -> recipe -> fit Cox (lifelines)
and Random Survival Forest (scikit-survival) -> tidy predictions -> survival
curves -> concordance / integrated-Brier / time-AUC metrics.

Requires:  pip install "pytidymodels[survival]"
Run:       python examples/survival.py
"""
import warnings

from pytidymodels import datasets, parsnip, recipes, rsample, workflows, yardstick
from pytidymodels.recipes import all_numeric_predictors

warnings.filterwarnings("ignore")

try:
    import lifelines  # noqa: F401
    import sksurv  # noqa: F401
    HAS_SURVIVAL = True
except (ImportError, OSError):
    HAS_SURVIVAL = False


def main():
    if not HAS_SURVIVAL:
        print("Survival models need lifelines + scikit-survival:")
        print("  pip install 'pytidymodels[survival]'")
        return

    # lung status is coded 1=censored, 2=dead -> recode event to 0/1
    lung = datasets.load_lung().copy()
    lung["event"] = (lung["status"] == 2).astype(int)
    lung = lung[["time", "event", "age", "sex", "ph.ecog", "ph.karno", "wt.loss"]]

    split = rsample.initial_split(lung, prop=0.8, seed=1)
    train, test = split.training(), split.testing()

    # two-column outcome: time + event
    rec = (recipes.Recipe(train, formula="time + event ~ .")
           .step_impute_median(all_numeric_predictors())
           .step_normalize(all_numeric_predictors()))

    models = {
        "Cox PH (lifelines)":
            parsnip.proportional_hazards(penalty=0.1).set_mode("censored regression"),
        "Weibull AFT (lifelines)":
            parsnip.survival_reg(dist="weibull").set_mode("censored regression"),
        "Random Survival Forest (sksurv)":
            parsnip.rand_forest(trees=200, min_n=8)
                   .set_engine("sksurv").set_mode("censored regression"),
        "Gradient Boosted (sksurv)":
            parsnip.boost_tree(trees=200, learn_rate=0.05, tree_depth=2)
                   .set_engine("sksurv").set_mode("censored regression"),
    }

    print(f"{'model':32s} {'C-index':>8s} {'C-ipcw':>8s} {'IBS':>7s} {'AUC':>7s}")
    print("-" * 66)
    fitted = {}
    for name, spec in models.items():
        wf = workflows.Workflow().add_recipe(rec).add_model(spec).fit(train)
        fitted[name] = wf
        m = yardstick.survival_metrics(wf, train, test).set_index("metric")["estimate"]
        print(f"{name:32s} {m.get('concordance', float('nan')):8.3f} "
              f"{m.get('concordance_ipcw', float('nan')):8.3f} "
              f"{m.get('integrated_brier_score', float('nan')):7.3f} "
              f"{m.get('mean_auc', float('nan')):7.3f}")

    # tidy per-subject predictions from the Cox model
    cox = fitted["Cox PH (lifelines)"]
    print("\nCox predictions (first 5 test subjects):")
    print(cox.predict(test).head().to_string(index=False))

    # survival probability S(t) at selected days
    print("\nSurvival probability S(t) at 180 / 365 / 730 days (first 5):")
    print(cox.predict_survival(test, [180, 365, 730]).head().to_string(index=False))


if __name__ == "__main__":
    main()
