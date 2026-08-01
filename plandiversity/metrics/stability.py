"""Stability: how much of two plans' action sets they fail to share."""

from .base import Metric, jaccard_distance


class Stability(Metric):
    """Jaccard distance between the sets of actions of two plans.

    ``1 - |A(a) & A(b)| / |A(a) | A(b)|``, where ``A(p)`` is the set of ground
    actions appearing in ``p``. Two plans built from the same actions score
    ``0.0`` however they are ordered, and two plans sharing no action score
    ``1.0``. Repeated actions are collapsed: the metric compares sets, not
    multisets or sequences.

    References
    ----------
    .. [1] T. A. Nguyen, M. Do, A. E. Gerevini, I. Serina, B. Srivastava, and
           S. Kambhampati, "Generating diverse plans to handle unknown and
           partially known user preferences," Artificial Intelligence,
           vol. 190, pp. 1-31, 2012.
    """

    name = "Stability"

    def _feature(self, plan):
        return frozenset(str(action) for action in plan.actions)

    def _distance(self, feature_a, feature_b) -> float:
        return jaccard_distance(feature_a, feature_b)
