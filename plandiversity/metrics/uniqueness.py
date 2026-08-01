"""Uniqueness: whether two plans are built from exactly the same actions."""

import numpy as np

from .base import Metric, group_codes


class Uniqueness(Metric):
    """Exact-match distance between the sets of actions of two plans.

    ``0.0`` when the two plans use exactly the same set of ground actions and
    ``1.0`` otherwise. This is the coarsest reading of diversity in the
    literature: a plan set is fully diverse as long as no two plans are
    reorderings of one another, no matter how much they overlap.
    :class:`~plandiversity.metrics.stability.Stability` grades the same
    comparison continuously.

    References
    ----------
    .. [1] M. Roberts, A. Howe, and I. Ray, "Evaluating diversity in classical
           planning," in Proceedings of the International Conference on
           Automated Planning and Scheduling, vol. 24, pp. 253-261, 2014.
    """

    name = "Uniqueness"

    def _feature(self, plan):
        return frozenset(str(action) for action in plan.actions)

    def _distance(self, feature_a, feature_b) -> float:
        return 0.0 if feature_a == feature_b else 1.0

    def pairwise(self, plans=None) -> np.ndarray:
        # One integer comparison per pair rather than one set comparison, which
        # matters for plan sets whose action sets are large.
        codes = group_codes(self._features_of(plans))
        return (codes[:, None] != codes[None, :]).astype(np.float64)
