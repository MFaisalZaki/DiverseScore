"""Aggregations of pairwise distances into one score. See :class:`.base.Model`."""

from .base import Model
from .max_mean import MaxMean
from .max_min import MaxMin
from .max_sum import MaxSum

__all__ = ["MaxMean", "MaxMin", "MaxSum", "Model"]
