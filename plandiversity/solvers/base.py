"""The :class:`Solver` base class: choosing k plans out of a larger set.

Scoring a plan set and *building* one are different problems. Diverse planners
in the ForbidIterative mould generate far more plans than asked for -- each
iteration reformulates the task to forbid the plans found so far -- and then
pick a subset of the requested size in post-processing. That second step is
what the solvers here do.

Picking the best k of n plans is the max-dispersion problem, which is NP-hard,
so there are two solvers: :class:`~plandiversity.solvers.greedy.GreedySolver`
builds a set quickly with no guarantee, and
:class:`~plandiversity.solvers.exact.ExactSolver` searches for the true optimum.

Katz and Sohrabi's taxonomy (AAAI 2020, section 4) names the problems these
correspond to:

``sat-k``
    Given k, find any k plans. ``GreedySolver`` -- what ForbidIterative itself
    uses, "a simple greedy algorithm, with a negligible computational
    overhead".
``optD-k``
    Given k, find the k plans of maximum diversity. ``ExactSolver``.
``bD-k``
    Given k and a bound b, find k plans whose diversity is at least b.
    ``ExactSolver(..., bound=b)``, which stops at the first subset that clears
    the bound instead of proving optimality.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from plandiversity.models.base import Model


class Selection(NamedTuple):
    """The plans a solver chose, and what they score."""

    #: The chosen plans, in the order the solver added them.
    plans: list
    #: Their positions in the plan set the solver was given.
    indices: tuple
    #: The model's score for the chosen plans.
    score: float
    #: Whether the score is known to be the best achievable for this size.
    optimal: bool

    def __len__(self) -> int:
        return len(self.plans)


class Solver(ABC):
    """Chooses the subset of a plan set that a model scores best.

    Parameters
    ----------
    model:
        The model whose score is being maximised. Its metrics are what make one
        subset better than another.

    Notes
    -----
    Every solver computes the distance matrix once and then works on indices
    into it, so a metric's cost is paid once per plan pair however many subsets
    are examined.
    """

    name = "Solver"

    def __init__(self, model: Model):
        self.model = model

    def select(self, planset, k: int, costs=None) -> Selection:
        """Choose ``k`` plans out of ``planset``.

        Parameters
        ----------
        planset:
            The candidate plans.
        k:
            How many to choose. Asking for at least as many plans as there are
            returns all of them -- a plan set smaller than requested is still a
            valid solution (Katz and Sohrabi, definition 1), not an error.
        costs:
            Optional plan costs, one per plan. Solvers break ties towards
            cheaper plans, which is how ForbidIterative keeps the selected set
            from drifting towards expensive plans it had no reason to prefer.
            Without them the plan set's own order breaks ties.

        Returns
        -------
        Selection
        """
        planset = list(planset)
        if k < 0:
            raise ValueError(f"cannot select {k} plans")
        order = self._preference_order(len(planset), costs)

        if k >= len(planset):
            # Every plan is selected, so there is nothing to choose between.
            return self._selection(planset, tuple(range(len(planset))), optimal=True)
        if k < 2:
            # No pairs, so every subset of this size scores the same.
            return self._selection(planset, tuple(order[:k]), optimal=True)

        distances = self.model.distance_matrix(planset)
        indices, optimal = self._choose(distances, k, order)
        return self._selection(planset, tuple(indices), optimal=optimal)

    @abstractmethod
    def _choose(self, distances: np.ndarray, k: int, order: np.ndarray):
        """Return ``(indices, optimal)`` for ``k`` plans, ``k`` in ``[2, n)``.

        ``order`` lists every plan index most-preferred first; solvers walk it
        in that order so that ties resolve towards cheaper plans.
        """

    def _selection(self, planset, indices, optimal) -> Selection:
        chosen = [planset[index] for index in indices]
        return Selection(
            plans=chosen,
            indices=indices,
            score=self.model(chosen),
            optimal=optimal,
        )

    @staticmethod
    def _preference_order(n: int, costs) -> np.ndarray:
        """Plan indices, cheapest first; input order when there are no costs."""
        if costs is None:
            return np.arange(n)
        costs = np.asarray(costs, dtype=np.float64)
        if costs.shape != (n,):
            raise ValueError(f"expected {n} costs, got {costs.shape[0]}")
        # Stable, so plans of equal cost keep the order they were given in.
        return np.argsort(costs, kind="stable")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model!r})"
