"""MaxMin: the distance between the two closest plans."""

import numpy as np

from .base import PRECISION, Model


class MaxMin(Model):
    """The smallest distance between any pair of plans in the set.

    The strictest of the three models: a single pair of near-identical plans
    drives the whole score to zero however diverse the rest of the set is. That
    makes it the one to use when a plan set is meant to offer genuinely
    distinct alternatives rather than a good average spread.

    Distances are already in ``[0, 1]``, so ``normalize`` has no effect here;
    it is accepted so that the three models remain interchangeable.

    References
    ----------
    .. [1] F. Sandoya, A. Martinez-Gavara, R. Aceves, A. Duarte, and R. Marti,
           "Diversity and equity models," in Handbook of Heuristics, 2018.
    """

    name = "MaxMin"

    def _score(self, distances: np.ndarray, size: int) -> float:
        return distances.min()

    def score_additions(self, distances, additions, size) -> np.ndarray:
        # A minimum absorbs new pairs without revisiting the old ones.
        return np.round(
            np.minimum(distances.min(), additions.min(axis=1)), PRECISION
        )
