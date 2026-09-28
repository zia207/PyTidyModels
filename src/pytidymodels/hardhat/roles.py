"""Column role constants.

Roles let every downstream module reason about columns by *purpose*
rather than position: what to predict, what to predict with, what to
carry along untouched.
"""

OUTCOME = "outcome"
PREDICTOR = "predictor"
ID = "id"
CASE_WEIGHT = "case_weight"
GROUP = "group"

ALL_ROLES = (OUTCOME, PREDICTOR, ID, CASE_WEIGHT, GROUP)
