"""Bundled sample datasets for examples, tests, and teaching.

Two synthetic generators (``load_ames_small``, ``load_credit_small``) plus a
curated collection of real datasets spanning regression, classification,
count, survival, and spatial tasks. See :func:`list_datasets`.
"""
from ._bundled import (
    list_datasets,
    load_air_pollution,
    load_airline_costs,
    load_boston,
    load_breast_cancer,
    load_county_diabetes,
    load_dataset,
    load_flights,
    load_health_insurance,
    load_lung,
    load_melanoma,
    load_nmes,
    load_pima,
    load_soil,
    load_soil_simulated,
    load_taxi,
)
from ._generate import load_ames_small, load_credit_small

__all__ = [
    # synthetic
    "load_ames_small",
    "load_credit_small",
    # registry / generic
    "list_datasets",
    "load_dataset",
    # real, bundled
    "load_boston",
    "load_melanoma",
    "load_nmes",
    "load_airline_costs",
    "load_breast_cancer",
    "load_soil",
    "load_soil_simulated",
    "load_health_insurance",
    "load_lung",
    "load_pima",
    "load_county_diabetes",
    "load_air_pollution",
    "load_flights",
    "load_taxi",
]
