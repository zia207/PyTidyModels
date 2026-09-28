"""Selectors: declarative column choices resolved against live schema.

A selector is a small object that, given the ``ColumnInfo`` as it exists
at a given step, returns the matching column names. Because they resolve
*at each step*, ``all_numeric_predictors()`` after a one-hot step correctly
picks up the new dummy columns -- the selector describes intent, not a
frozen list.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..hardhat import roles
from ..hardhat.schema import NOMINAL, NUMERIC, ColumnInfo


class Selector:
    def resolve(self, info: ColumnInfo) -> list[str]:  # pragma: no cover
        raise NotImplementedError


@dataclass
class _RoleType(Selector):
    role: str | None = None
    vtype: str | None = None

    def resolve(self, info: ColumnInfo) -> list[str]:
        out = []
        for col in info.columns():
            if self.role is not None and info.roles.get(col) != self.role:
                continue
            if self.vtype is not None and info.types.get(col) != self.vtype:
                continue
            out.append(col)
        return out


@dataclass
class _Pattern(Selector):
    kind: str      # "starts", "ends", "contains", "matches"
    pattern: str

    def resolve(self, info: ColumnInfo) -> list[str]:
        cols = info.columns()
        p = self.pattern
        if self.kind == "starts":
            return [c for c in cols if c.startswith(p)]
        if self.kind == "ends":
            return [c for c in cols if c.endswith(p)]
        if self.kind == "contains":
            return [c for c in cols if p in c]
        rx = re.compile(p)
        return [c for c in cols if rx.search(c)]


@dataclass
class _Named(Selector):
    names: tuple[str, ...]

    def resolve(self, info: ColumnInfo) -> list[str]:
        return [c for c in self.names if c in info.columns()]


# -- public selector constructors -------------------------------------------

def all_predictors() -> Selector:
    return _RoleType(role=roles.PREDICTOR)


def all_outcomes() -> Selector:
    return _RoleType(role=roles.OUTCOME)


def all_numeric() -> Selector:
    return _RoleType(vtype=NUMERIC)


def all_nominal() -> Selector:
    return _RoleType(vtype=NOMINAL)


def all_numeric_predictors() -> Selector:
    return _RoleType(role=roles.PREDICTOR, vtype=NUMERIC)


def all_nominal_predictors() -> Selector:
    return _RoleType(role=roles.PREDICTOR, vtype=NOMINAL)


def starts_with(prefix: str) -> Selector:
    return _Pattern("starts", prefix)


def ends_with(suffix: str) -> Selector:
    return _Pattern("ends", suffix)


def contains(substr: str) -> Selector:
    return _Pattern("contains", substr)


def matches(regex: str) -> Selector:
    return _Pattern("matches", regex)


def to_selector(x) -> Selector:
    """Coerce a selector, a column name, or a list of names to a Selector."""
    if isinstance(x, Selector):
        return x
    if isinstance(x, str):
        return _Named((x,))
    return _Named(tuple(x))
