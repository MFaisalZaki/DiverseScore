"""MaxMean: total pairwise distance per plan."""

import numpy as np

from .base import PRECISION, Model, pair_count


class MaxMean(Model):
    """The sum of all pairwise distances divided by the number of plans.

    The max-mean dispersion objective: how much distance each plan contributes
    on average. Unlike :class:`~plandiversity.models.max_sum.MaxSum` it does not
    reward a set simply for being larger, which makes it the fairer choice when
    comparing plan sets of different sizes.

    Note the denominator is the number of *plans*, not the number of pairs, so
    the score still grows as ``(n - 1) / 2`` for a maximally diverse set. Pass
    ``normalize=True`` for the mean pairwise distance in ``[0, 1]``.

    References
    ----------
    .. [1] F. Sandoya, A. Martinez-Gavara, R. Aceves, A. Duarte, and R. Marti,
           "Diversity and equity models," in Handbook of Heuristics, 2018.
    """

    name = "MaxMean"

    def _score(self, distances: np.ndarray, size: int) -> float:
        if self.normalize:
            return distances.sum() / distances.size
        return distances.sum() / size

    def score_additions(self, distances, additions, size) -> np.ndarray:
        totals = distances.sum() + additions.sum(axis=1)
        divisor = pair_count(size) if self.normalize else size
        return np.round(totals / divisor, PRECISION)
