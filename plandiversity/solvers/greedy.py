"""The greedy plan selector ForbidIterative uses."""

import numpy as np

from .base import Solver


class GreedySolver(Solver):
    """Grow a plan set one plan at a time, always taking the best next plan.

    This is the selection step of Katz and Sohrabi (2020), section 5.2, quoted
    in full because the algorithm is exactly its description:

        We first order the found plans by their cost. Then, going from the
        cheapest plans to the more expensive ones, we find a pair of plans with
        the largest diversity score. Starting with the found pair of plans, we
        iteratively construct the set by greedily choosing the next plan to add
        to the set, maximizing the diversity of the resulting set at that
        iteration step. We stop once the set reaches the requested size k.

    The paper is candid that "the quality of the solution obtained by such an
    algorithm may be considerably improved" -- it is a heuristic, and
    :class:`~plandiversity.solvers.exact.ExactSolver` will sometimes beat it.
    What it buys is a polynomial cost: ``O(n^2)`` to find the opening pair, and
    ``O(n k)`` distance lookups for each plan added after that.

    What "maximizing the diversity of the resulting set" means is left to the
    model, so the greedy step follows whichever objective is being measured.
    Under :class:`~plandiversity.models.max_sum.MaxSum` it adds the plan with
    the largest total distance to those already chosen; under
    :class:`~plandiversity.models.max_min.MaxMin` it adds the plan whose nearest
    already-chosen neighbour is furthest away.

    References
    ----------
    .. [1] M. Katz and S. Sohrabi, "Reshaping diverse planning," in Proceedings
           of the Thirty-Fourth AAAI Conference on Artificial Intelligence,
           pp. 9892-9899, 2020.
    """

    name = "Greedy"

    def _choose(self, distances: np.ndarray, k: int, order: np.ndarray):
        chosen = list(self._best_pair(distances, order))
        # The pair distances of the set built so far. Carried along rather than
        # recomputed, so adding a plan costs one row of the matrix.
        pairs = np.array([distances[chosen[0], chosen[1]]], dtype=np.float64)

        taken = np.zeros(len(distances), dtype=bool)
        taken[chosen] = True

        while len(chosen) < k:
            candidates = order[~taken[order]]
            # Row per candidate: its distance to each already-chosen plan.
            additions = distances[np.ix_(candidates, chosen)]
            scores = self.model.score_additions(pairs, additions, len(chosen) + 1)
            # argmax takes the first maximum and candidates is in preference
            # order, so ties go to the cheaper plan.
            best = int(np.argmax(scores))

            pairs = np.concatenate([pairs, additions[best]])
            chosen.append(int(candidates[best]))
            taken[candidates[best]] = True

        return chosen, False

    @staticmethod
    def _best_pair(distances: np.ndarray, order: np.ndarray) -> tuple:
        """The two furthest-apart plans, preferring cheaper ones on a tie.

        Every model ranks two-plan sets by their single pair distance -- the
        sum, mean and minimum of one number are all that number -- so this is
        the best opening pair whichever model is in use.
        """
        # Reordered into preference order, so that the first maximum found is
        # the one built from the most-preferred plans.
        preferred = distances[np.ix_(order, order)]
        # -1 rather than 0 below the diagonal: a plan set whose plans are all
        # identical has nothing but zero distances, and masking with 0 would
        # let the diagonal win and pair a plan with itself.
        masked = np.full(preferred.shape, -1.0)
        upper = np.triu_indices(len(order), k=1)
        masked[upper] = preferred[upper]
        row, column = np.unravel_index(np.argmax(masked), masked.shape)
        return int(order[row]), int(order[column])
