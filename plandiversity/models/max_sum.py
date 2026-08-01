"""MaxSum: total pairwise distance across the plan set."""

import numpy as np

from .base import PRECISION, Model, pair_count


class MaxSum(Model):
    """The sum of the distances between every pair of plans in the set.

    The classic dispersion objective. It grows quadratically with the size of
    the plan set, so raw MaxSum scores are only comparable between sets of the
    same size; ``normalize=True`` divides by the pair count and yields the mean
    pairwise distance instead.

    Because it totals every pair, a set can score well while containing a
    cluster of near-identical plans, as long as the rest are far apart. Use
    :class:`~plandiversity.models.max_min.MaxMin` when that matters.

    References
    ----------
    .. [1] F. Sandoya, A. Martinez-Gavara, R. Aceves, A. Duarte, and R. Marti,
           "Diversity and equity models," in Handbook of Heuristics, 2018.
    """

    name = "MaxSum"

    def _score(self, distances: np.ndarray, size: int) -> float:
        total = distances.sum()
        return total / distances.size if self.normalize else total

    def score_additions(self, distances, additions, size) -> np.ndarray:
        # A sum absorbs new pairs without revisiting the old ones.
        totals = distances.sum() + additions.sum(axis=1)
        if self.normalize:
            totals = totals / pair_count(size)
        return np.round(totals, PRECISION)
