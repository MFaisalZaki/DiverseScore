"""Everything needed to score and select plan sets, in one import.

``from plandiversity.shortcuts import *`` also re-exports the unified-planning
shortcuts, so a script needs only this one star import to build or parse a
task, read its plans, score them, and pick a diverse subset.
"""

import unified_planning.shortcuts as _up_shortcuts
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *

from plandiversity.metrics import (
    InapplicablePlanError,
    Metric,
    Stability,
    States,
    Uniqueness,
)
from plandiversity.models import MaxMean, MaxMin, MaxSum, Model
from plandiversity.solvers import ExactSolver, GreedySolver, Selection, Solver

#: This package's own names. Everything public in
#: :mod:`unified_planning.shortcuts` is appended below, so the star import
#: keeps behaving as a superset of the framework's own.
__all__ = [
    # models
    "MaxMean",
    "MaxMin",
    "MaxSum",
    "Model",
    # metrics
    "Metric",
    "Stability",
    "States",
    "Uniqueness",
    # solvers
    "ExactSolver",
    "GreedySolver",
    "Selection",
    "Solver",
    # support
    "InapplicablePlanError",
    "PDDLReader",
]

__all__ += [
    name
    for name in dir(_up_shortcuts)
    if not name.startswith("_") and name not in __all__
]
