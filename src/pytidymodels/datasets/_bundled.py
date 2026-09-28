"""Bundled real-world datasets shipped with PyTidyModels.

A small, curated collection spanning regression, classification, count,
survival, and spatial tasks -- the analog of tidymodels' ``modeldata``. Each
``load_*`` returns a tidy pandas DataFrame; :func:`list_datasets` gives a
one-row-per-dataset overview (task, outcome, shape, source).

Data files ship gzip-compressed under ``datasets/data/`` and are read via
``importlib.resources`` so they resolve whether the package is installed
normally or from a zip. Large sources (NYC taxi, flights) are cleaned and
subsampled to a teaching size; the rest ship whole. See ``list_datasets()``
and each loader's docstring for provenance.
"""
from __future__ import annotations

from importlib.resources import files

import pandas as pd

# name -> (task, outcome, n_rows, n_cols, description, source)
_REGISTRY: dict[str, dict] = {
    "boston": dict(
        task="regression", outcome="medv", n_rows=506, n_cols=14,
        description="Boston census-tract home values and neighborhood covariates.",
        source="Harrison & Rubinfeld (1978); MASS::Boston."),
    "melanoma": dict(
        task="survival", outcome="time, status", n_rows=205, n_cols=7,
        description="Malignant melanoma survival after surgery (status: 1 died-melanoma, 2 alive, 3 died-other).",
        source="Andersen et al.; MASS/boot::Melanoma."),
    "nmes": dict(
        task="count", outcome="visits", n_rows=4406, n_cols=13,
        description="Physician office visits and health/demographic covariates (count outcome).",
        source="US National Medical Expenditure Survey 1987/88; AER::NMES1988."),
    "airline_costs": dict(
        task="regression", outcome="C", n_rows=90, n_cols=6,
        description="Airline cost panel: 6 firms x 15 years (cost, output, fuel price, load factor).",
        source="Greene, econometric airline cost panel."),
    "breast_cancer": dict(
        task="classification", outcome="status", n_rows=198, n_cols=33,
        description="Wisconsin prognostic breast cancer: recurrence (R) vs non-recurrence (N) from nuclear features.",
        source="Wisconsin Prognostic Breast Cancer (WPBC)."),
    "soil": dict(
        task="spatial regression", outcome="SOC", n_rows=467, n_cols=19,
        description="Soil organic carbon field samples with lon/lat and environmental covariates.",
        source="Ahmed / Uppata Analytics (field data)."),
    "soil_simulated": dict(
        task="regression", outcome="SOC", n_rows=3000, n_cols=9,
        description="Simulated soil organic carbon with terrain/climate covariates (larger, clean).",
        source="Ahmed / Uppata Analytics (simulated)."),
    "health_insurance": dict(
        task="count", outcome="absent", n_rows=1448, n_cols=6,
        description="Employee absenteeism days with product, age, household, position, gender.",
        source="Employee health-insurance / absenteeism sample."),
    "lung": dict(
        task="survival", outcome="time, status", n_rows=228, n_cols=10,
        description="NCCTG lung cancer survival (status: 1 censored, 2 dead); has missing covariates.",
        source="Loprinzi et al.; survival::lung."),
    "pima": dict(
        task="classification", outcome="Outcome", n_rows=768, n_cols=9,
        description="Pima Indians diabetes: diagnostic measures vs diabetes onset (0/1).",
        source="UCI / mlbench::PimaIndiansDiabetes."),
    "county_diabetes": dict(
        task="spatial regression", outcome="Diabetes_per", n_rows=3107, n_cols=15,
        description="US county diabetes prevalence 2016-2020 with obesity, inactivity, access covariates and X/Y.",
        source="US county health data 2016-2020 (compiled)."),
    "air_pollution": dict(
        task="spatial regression", outcome="RATE", n_rows=46605, n_cols=13,
        description="US county-year mortality rate vs air pollution (PM2.5, NO2, SO2) and covariates; panel with X/Y and Year.",
        source="US county-year air pollution & mortality panel (compiled)."),
    "flights": dict(
        task="regression", outcome="arr_delay", n_rows=8000, n_cols=19,
        description="NYC 2013 flights (sampled): delays, carrier, route, distance, air time.",
        source="nycflights13 (sampled 8000 rows, seed 42)."),
    "taxi": dict(
        task="regression", outcome="total_amount", n_rows=8000, n_cols=12,
        description="NYC yellow taxi trips Jan 2023 (cleaned sample): distance, duration, hour, fare components.",
        source="NYC TLC trip records (valid rows, sampled 8000, seed 42)."),
}


def _read(name: str) -> pd.DataFrame:
    resource = files("pytidymodels.datasets").joinpath("data", f"{name}.csv.gz")
    with resource.open("rb") as fh:
        return pd.read_csv(fh, compression="gzip")


def load_dataset(name: str) -> pd.DataFrame:
    """Load any bundled dataset by name (see :func:`list_datasets`)."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}.")
    return _read(name)


def list_datasets() -> pd.DataFrame:
    """Overview of every bundled dataset: task, outcome, shape, source."""
    rows = [{"dataset": n, "task": m["task"], "outcome": m["outcome"],
             "n_rows": m["n_rows"], "n_cols": m["n_cols"],
             "description": m["description"], "source": m["source"]}
            for n, m in _REGISTRY.items()]
    return pd.DataFrame(rows)


# -- explicit named loaders (for discoverability / autocomplete) ------------

def load_boston() -> pd.DataFrame:
    """Boston home values (regression, outcome ``medv``)."""
    return _read("boston")


def load_melanoma() -> pd.DataFrame:
    """Melanoma survival (survival: ``time`` + ``status``)."""
    return _read("melanoma")


def load_nmes() -> pd.DataFrame:
    """Physician visits (count regression, outcome ``visits``)."""
    return _read("nmes")


def load_airline_costs() -> pd.DataFrame:
    """Airline cost panel (regression, outcome ``C``)."""
    return _read("airline_costs")


def load_breast_cancer() -> pd.DataFrame:
    """Breast-cancer recurrence (classification, outcome ``status``)."""
    return _read("breast_cancer")


def load_soil() -> pd.DataFrame:
    """Soil organic carbon field data (spatial regression, outcome ``SOC``)."""
    return _read("soil")


def load_soil_simulated() -> pd.DataFrame:
    """Simulated soil organic carbon (regression, outcome ``SOC``)."""
    return _read("soil_simulated")


def load_health_insurance() -> pd.DataFrame:
    """Employee absenteeism (count regression, outcome ``absent``)."""
    return _read("health_insurance")


def load_lung() -> pd.DataFrame:
    """NCCTG lung cancer survival (survival: ``time`` + ``status``)."""
    return _read("lung")


def load_pima() -> pd.DataFrame:
    """Pima Indians diabetes (classification, outcome ``Outcome``)."""
    return _read("pima")


def load_county_diabetes() -> pd.DataFrame:
    """County diabetes prevalence (spatial regression, outcome ``Diabetes_per``)."""
    return _read("county_diabetes")


def load_air_pollution() -> pd.DataFrame:
    """County-year mortality vs air pollution (spatial regression, outcome ``RATE``)."""
    return _read("air_pollution")


def load_flights() -> pd.DataFrame:
    """NYC 2013 flights, sampled (regression, outcome ``arr_delay``)."""
    return _read("flights")


def load_taxi() -> pd.DataFrame:
    """NYC yellow taxi trips, cleaned sample (regression, outcome ``total_amount``)."""
    return _read("taxi")
