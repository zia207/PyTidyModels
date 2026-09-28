"""Schema inference: roles, variable types, and a minimal formula parser.

This is the shared vocabulary the ``recipes`` grammar and ``parsnip``
specs both rely on. Types are deliberately coarse -- ``"numeric"`` vs
``"nominal"`` -- which is all the selectors need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from pandas.api import types as pdt

from . import roles as _roles

NUMERIC = "numeric"
NOMINAL = "nominal"


def var_type(series: pd.Series) -> str:
    """Coarse variable type used by recipe selectors."""
    if pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series):
        return NUMERIC
    return NOMINAL


@dataclass
class ColumnInfo:
    """Per-column ``(role, type)`` table threaded through recipe steps.

    Steps read it to resolve selectors and rewrite it as they change the
    data (e.g. one-hot encoding drops a nominal column and adds numeric
    ones), so a later selector sees the columns as they exist *at that
    point* in the pipeline -- exactly the tidymodels behaviour.
    """

    roles: dict[str, str] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, data: pd.DataFrame, outcome: Optional[str] = None,
                   outcomes: Optional[list[str]] = None) -> "ColumnInfo":
        outs = set(outcomes or ([] if outcome is None else [outcome]))
        info = cls()
        for col in data.columns:
            info.roles[col] = _roles.OUTCOME if col in outs else _roles.PREDICTOR
            info.types[col] = var_type(data[col])
        return info

    def columns(self) -> list[str]:
        return list(self.roles.keys())

    def with_type(self, col: str) -> str:
        return self.types.get(col, NOMINAL)

    def copy(self) -> "ColumnInfo":
        return ColumnInfo(dict(self.roles), dict(self.types))

    def drop(self, col: str) -> None:
        self.roles.pop(col, None)
        self.types.pop(col, None)

    def add(self, col: str, role: str, vtype: str) -> None:
        self.roles[col] = role
        self.types[col] = vtype

    def predictors(self) -> list[str]:
        return [c for c, r in self.roles.items() if r == _roles.PREDICTOR]

    def outcomes(self) -> list[str]:
        return [c for c, r in self.roles.items() if r == _roles.OUTCOME]


def parse_formula(formula: str, data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Parse a minimal ``"y ~ x1 + x2"`` / ``"y ~ ."`` formula.

    Kept dependency-free on purpose; a ``formulaic``/``patsy`` backend can
    slot in later behind this same signature. Returns ``(outcomes,
    predictors)``.
    """
    if "~" not in formula:
        raise ValueError(f"Formula must contain '~': got {formula!r}")
    lhs, rhs = (side.strip() for side in formula.split("~", 1))
    outcomes = [t.strip() for t in lhs.split("+") if t.strip()]
    rhs_terms = [t.strip() for t in rhs.split("+") if t.strip()]
    if rhs_terms == ["."]:
        predictors = [c for c in data.columns if c not in outcomes]
    else:
        predictors = rhs_terms
    missing = [c for c in outcomes + predictors if c not in data.columns]
    if missing:
        raise ValueError(f"Formula references unknown columns: {missing}")
    return outcomes, predictors
