"""The exact plan selector: the best k plans, or k plans meeting a bound."""

from math import comb

import numpy as np

from .base import Solver


class ExactSolver(Solver):
    """Search every subset of the requested size, pruning the hopeless ones.

    Solves ``optD-k`` -- given k, find the k plans of maximum diversity -- and,
    with a ``bound``, ``bD-k``: find k plans whose diversity is at least b.
    Katz and Sohrabi solve the bounded problem with a mixed-integer program
    (under MaxMin it is a k-clique on the graph of pairs at least b apart), and
    report exact techniques for the *optimal* problem to be "prohibitively
    slow" -- which is why ForbidIterative ships the greedy selector instead.

    The same caveat applies here. Choosing k of n plans means ``C(n, k)``
    candidate subsets, so this is for small plan sets, for measuring what the
    greedy solver gave up, and for the bounded problem, where the search stops
    at the first subset clearing the bound rather than proving optimality.

    Pruning comes from :meth:`~plandiversity.models.base.Model.optimistic_score`:
    a partial set is abandoned once the best score any completion of it could
    reach fails to beat both the best set already found and the bound. Under
    :class:`~plandiversity.models.max_min.MaxMin` that estimate is exact --
    adding a plan can only lower a minimum -- so the search there is far
    cheaper than the subset count suggests.

    Parameters
    ----------
    model:
        The model whose score is being maximised.
    bound:
        Stop at the first set of k plans scoring at least this much, instead of
        searching for the best. The resulting :class:`.base.Selection` is
        marked ``optimal=False``: it meets the bound but was never compared
        against the alternatives. A bound no set can reach makes the problem
        unsolvable, and :meth:`~.base.Solver.select` raises ``ValueError`` --
        as in classical planning, where a super-optimal bound has no solution.
    max_subsets:
        Refuse outright rather than begin a search over more than this many
        subsets. ``None`` removes the guard.

    References
    ----------
    .. [1] M. Katz and S. Sohrabi, "Reshaping diverse planning," in Proceedings
           of the Thirty-Fourth AAAI Conference on Artificial Intelligence,
           pp. 9892-9899, 2020.
    """

    name = "Exact"

    def __init__(self, model, bound=None, max_subsets=2_000_000):
        super().__init__(model)
        self.bound = bound
        self.max_subsets = max_subsets

    def _choose(self, distances: np.ndarray, k: int, order: np.ndarray):
        n = len(order)
        self._guard_search_size(n, k)

        # Distances reindexed into preference order, so the depth-first walk
        # meets cheaper plans first and ties resolve towards them.
        preferred = distances[np.ix_(order, order)]
        ceiling = float(preferred.max())

        best_score = -np.inf
        best_set = None
        satisfied = False

        def descend(chosen, pairs, start):
            nonlocal best_score, best_set, satisfied

            if len(chosen) == k:
                score = self.model.score_pairs(pairs, k)
                if score > best_score:
                    best_score, best_set = score, list(chosen)
                    satisfied = self.bound is not None and score >= self.bound
                return

            # Fewer plans remain than are needed to reach k.
            if n - start < k - len(chosen):
                return

            # No completion of this partial set can beat the incumbent, nor
            # reach the bound if there is one.
            if len(chosen) > 1:
                reachable = self.model.optimistic_score(pairs, len(chosen), k, ceiling)
                if reachable <= best_score:
                    return
                if self.bound is not None and reachable < self.bound:
                    return

            for candidate in range(start, n):
                chosen.append(candidate)
                descend(
                    chosen,
                    np.concatenate([pairs, preferred[candidate, chosen[:-1]]]),
                    candidate + 1,
                )
                chosen.pop()
                if satisfied:
                    return

        descend([], np.empty(0, dtype=np.float64), 0)

        if self.bound is not None and not satisfied:
            # The search was exhaustive -- nothing was pruned that could have
            # reached the bound -- so no such set exists. Returning the best
            # set found instead would hand back one that misses the bound the
            # caller asked for.
            raise ValueError(
                f"no set of {k} plans reaches a diversity of {self.bound} under "
                f"{self.model}; the bound is above what this plan set can offer"
            )
        # Back from preference order into the caller's plan set indices.
        return [int(order[position]) for position in best_set], not satisfied

    def _guard_search_size(self, n: int, k: int) -> None:
        if self.max_subsets is None:
            return
        subsets = comb(n, k)
        if subsets > self.max_subsets:
            raise ValueError(
                f"choosing {k} of {n} plans means searching {subsets:,} subsets, "
                f"over the max_subsets limit of {self.max_subsets:,}. Use "
                f"GreedySolver, or raise the limit if the wait is acceptable."
            )
