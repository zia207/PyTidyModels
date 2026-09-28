"""Blueprint: the schema captured at fit time and replayed at predict time.

Its whole job is to guarantee that the design matrix a model sees at
``predict`` has exactly the columns, in the order, it saw at ``fit`` --
missing dummy levels filled with 0, unseen extras dropped -- so a model
never silently misaligns on new data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Blueprint:
    predictors: list[str]
    outcome: str | None = None

    def align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` reindexed to the fitted predictor columns."""
        return X.reindex(columns=self.predictors, fill_value=0)
