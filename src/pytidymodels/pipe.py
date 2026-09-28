"""A magrittr-style pipe for PyTidyModels.

Python has a fixed operator set and no way to define ``%>%``, so a literal
custom pipe operator isn't possible. This module gets as close as Python
allows, three ways:

1. **Method chaining** -- already how recipes and workflows read
   (``Recipe(...).step_x().step_y()`` *is* ``recipe %>% step_x %>% step_y``).

2. **The ``>>`` operator** with :class:`pipe`, for threading a value through
   plain functions the way magrittr threads verbs::

       from pytidymodels import pipe, _
       from pytidymodels import rsample

       train = (ames
                >> pipe(rsample.initial_split, prop=0.8, strata="price")
                >> pipe(rsample.training))

   The value flows in as the first argument, or wherever you place the ``_``
   placeholder (magrittr's ``.``)::

       2 >> pipe(pow, _, 10)      # pow(2, 10)

   ``>>`` is used rather than ``|`` because pandas already defines ``|``
   (elementwise or) on DataFrames and would swallow the pipe; ``>>`` falls
   through to us.

3. **Composition**: piping two ``pipe`` objects builds a reusable flow::

       clean = pipe(str.strip) >> pipe(str.upper)
       "  hi  " >> clean        # "HI"

Every core object also has a ``.pipe(func, ...)`` method (like pandas), so you
can thread an object into any function without leaving the method-chain style.
"""
from __future__ import annotations

from typing import Any, Callable


class _Placeholder:
    """magrittr's ``.`` -- marks where the piped value is substituted."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_"


#: The placeholder singleton (import as ``_``).
_ = _Placeholder()


def _has_placeholder(args, kwargs) -> bool:
    return any(a is _ for a in args) or any(v is _ for v in kwargs.values())


def _apply(func, args, kwargs, value):
    if _has_placeholder(args, kwargs):
        new_args = [value if a is _ else a for a in args]
        new_kwargs = {k: (value if v is _ else v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
    return func(value, *args, **kwargs)   # default: value is the first argument


class Pipe:
    """A pipe stage (or a composed chain of stages).

    Construct one per call -- ``pipe(func, *args, **kwargs)`` -- then apply it
    with ``value >> stage``. Composing two stages with ``>>`` yields a new,
    reusable :class:`Pipe`. Also directly callable: ``pipe(func)(value)``.
    """

    __slots__ = ("_stages",)

    def __init__(self, func: Callable | None = None, *args: Any,
                 _stages=None, **kwargs: Any):
        if _stages is not None:
            self._stages = _stages
            return
        if not callable(func):
            raise TypeError(
                f"pipe() expects a callable, got {type(func).__name__}. "
                "Write value >> pipe(func, ...), not pipe(value).")
        self._stages = [(func, args, kwargs)]

    def _run(self, value):
        for func, args, kwargs in self._stages:
            value = _apply(func, args, kwargs, value)
        return value

    def __rrshift__(self, value):        # value >> pipe(...)  -> apply
        return self._run(value)

    def __rshift__(self, other):         # pipe(...) >> pipe(...) -> compose
        if isinstance(other, Pipe):
            return Pipe(_stages=self._stages + other._stages)
        return NotImplemented

    def __call__(self, value):           # pipe(func)(value) -> apply
        return self._run(value)

    def __repr__(self) -> str:
        names = ", ".join(getattr(f, "__name__", repr(f))
                          for f, _a, _k in self._stages)
        return f"pipe({names})"


#: Primary spelling: ``pipe`` is the wrapper class itself.
pipe = Pipe
#: Short alias, for those who like it.
P = Pipe


def pipeable(func: Callable) -> Callable:
    """Decorator making ``func`` a bare pipe verb (returns a :class:`Pipe`).

    Lets a helper sit in a chain without wrapping::

        @pipeable
        def scale(x, by=2): return x * by

        10 >> scale(by=3)      # 30

    This changes the function's direct-call semantics (it now returns a
    ``Pipe``), so use it for pipe-only helpers, not core verbs whose normal
    ``func(data, ...)`` call you still want.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        return Pipe(func, *args, **kwargs)

    return wrapper


class PipeMixin:
    """Adds a pandas-style ``.pipe(func, *args, **kwargs)`` to an object."""

    def pipe(self, func, *args, **kwargs):
        """Return ``func(self, *args, **kwargs)`` -- thread self into ``func``."""
        return func(self, *args, **kwargs)
