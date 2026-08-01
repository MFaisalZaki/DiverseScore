"""The :class:`Model` base class: turning pairwise distances into one score."""

from abc import ABC, abstractmethod

import numpy as np

from plandiversity.metrics.base import Metric

#: Decimal places every model rounds its score to.
PRECISION = 5


def pair_count(size: int) -> int:
    """The number of unordered pairs in a set of ``size`` plans."""
    return size * (size - 1) // 2


def upper_triangle(distances: np.ndarray) -> np.ndarray:
    """The strict upper triangle of a distance matrix, flattened."""
    rows, columns = np.triu_indices(distances.shape[0], k=1)
    return distances[rows, columns]


class Model(ABC):
    """Aggregates the pairwise distances of a plan set into a single score.

    A model takes one or more metrics, averages their distance matrices, and
    reduces the result to a number -- summing it, averaging it, or taking its
    minimum. Higher always means more diverse, because every metric reports a
    distance (see :class:`~plandiversity.metrics.base.Metric`).

    Parameters
    ----------
    metrics:
        The metrics to score with; a bare metric is accepted as shorthand for a
        list of one. With several, each pair's distance is the mean of what the
        metrics report for it, so they must be comparably scaled -- all of the
        built-in metrics are, in ``[0, 1]``.
    normalize:
        Scale the score into ``[0, 1]`` by dividing by the largest value it
        could take if every pair of plans were maximally distant. This makes
        scores comparable across plan sets of different sizes, at the cost of
        no longer rewarding a larger set for being larger.
    """

    name = "Model"

    def __init__(self, metrics, normalize=False):
        if isinstance(metrics, Metric):
            metrics = [metrics]
        self.metrics = list(metrics)
        if not self.metrics:
            raise ValueError("a model needs at least one metric")
        self.normalize = normalize

    def distance_matrix(self, planset) -> np.ndarray:
        """The symmetric ``(n, n)`` matrix of mean distances across the metrics."""
        planset = list(planset)
        n = len(planset)
        total = np.zeros((n, n), dtype=np.float64)
        for metric in self.metrics:
            distances = np.asarray(metric.pairwise(planset), dtype=np.float64)
            if distances.shape != (n, n):
                raise ValueError(
                    f"{metric} returned a {distances.shape} matrix for {n} plans; "
                    f"Metric.pairwise must return an (n, n) matrix"
                )
            total += distances
        return total / len(self.metrics)

    def pairwise_distances(self, planset) -> np.ndarray:
        """The distances of the ``n * (n - 1) / 2`` unordered pairs, as a flat array.

        Every model's score is a function of just these numbers.
        """
        return upper_triangle(self.distance_matrix(planset))

    def __call__(self, planset) -> float:
        """Score ``planset``. Fewer than two plans have no pairs, and score ``0.0``."""
        planset = list(planset)
        return self.score_pairs(self.pairwise_distances(planset), len(planset))

    def score_matrix(self, distances: np.ndarray) -> float:
        """Score an already-computed ``(n, n)`` distance matrix.

        The entry point for scoring many subsets of one plan set: the matrix is
        built once and sliced, rather than recomputed per subset.
        """
        return self.score_pairs(upper_triangle(distances), distances.shape[0])

    def score_pairs(self, distances: np.ndarray, size: int) -> float:
        """Score a set of ``size`` plans from its flat array of pair distances."""
        if distances.size == 0:
            return 0.0
        return round(float(self._score(distances, size)), PRECISION)

    def score_additions(
        self, distances: np.ndarray, additions: np.ndarray, size: int
    ) -> np.ndarray:
        """Score each of several one-plan extensions of the same set.

        ``distances`` holds the pair distances of a set of ``size - 1`` plans
        and ``additions`` is a ``(candidates, size - 1)`` array whose rows are
        the distances from each candidate plan to those already chosen. Returns
        one score per candidate: what the set would score with that plan added.

        The greedy solver calls this once per plan it adds, so a model that can
        fold a new plan into its score without re-reducing the whole set should
        override it -- that is the difference between the solver costing
        ``O(n k^2)`` and ``O(n k^3)``. An override must agree with this
        definition exactly; ``tests/test_solvers.py`` checks that it does.
        """
        return np.array(
            [
                self.score_pairs(np.concatenate([distances, addition]), size)
                for addition in additions
            ],
            dtype=np.float64,
        )

    def optimistic_score(
        self, distances: np.ndarray, size: int, target: int, ceiling: float = 1.0
    ) -> float:
        """The best score any ``target``-sized superset of this partial set could reach.

        Assumes every pair not yet fixed achieves ``ceiling``, the largest
        distance still available. Used by
        :class:`~plandiversity.solvers.exact.ExactSolver` to prune: a branch
        whose optimistic score already loses to the incumbent cannot win.

        Admissible for any model whose ``_score`` is monotone non-decreasing in
        each pair distance, which the three built-in models are.
        """
        remaining = pair_count(target) - pair_count(size)
        if remaining < 0:
            raise ValueError(f"a set of {size} plans is already larger than {target}")
        padded = np.concatenate([distances, np.full(remaining, float(ceiling))])
        return self.score_pairs(padded, target)

    @abstractmethod
    def _score(self, distances: np.ndarray, size: int) -> float:
        """Reduce the non-empty pairwise ``distances`` of a set of ``size`` plans."""

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        metrics = ", ".join(str(metric) for metric in self.metrics)
        return f"{type(self).__name__}([{metrics}], normalize={self.normalize})"
