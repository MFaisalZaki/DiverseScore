from unified_planning.plans import SequentialPlan, ActionInstance
from unified_planning.shortcuts import *

class Model:
    """Base class for all models."""

    def __init__(self, task, metric, name):
        """Initialize model by generating the grounded actions for the planning task."""
        self.name = name
        self.metric = metric
        self.task = task

    def __call__(self, planset):
        """Compute the diversity model for planset."""
        raise NotImplementedError
    