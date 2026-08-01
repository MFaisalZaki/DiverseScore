"""Solver behaviour, checked against brute force over a hand-built geometry.

Solvers are combinatorial optimisers over a distance matrix; which planning
metric produced that matrix is irrelevant to them. So most of this file drives
them with :class:`MatrixMetric`, whose distances are a table written out in the
test, and whose "plans" are the integers indexing it. That makes the geometry
of each case explicit and the expected answer checkable by hand.

``tests/test_end_to_end.py`` covers the solvers over real plans.
"""

from itertools import combinations

import numpy as np
import pytest

from plandiversity.metrics.base import Metric
from plandiversity.models.base import Model
from plandiversity.shortcuts import (
    ExactSolver,
    GreedySolver,
    MaxMean,
    MaxMin,
    MaxSum,
    Solver,
    Stability,
)


class MatrixMetric(Metric):
    """Reads its distances from a fixed table; a "plan" is a row index."""

    name = "Matrix"

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        super().__init__(task=None, plans=None)

    def _feature(self, plan):
        return plan

    def _distance(self, feature_a, feature_b) -> float:
        return float(self.matrix[feature_a][feature_b])


def matrix_from_pairs(n, pairs, default=0.0):
    """A symmetric matrix, zero on the diagonal, from ``{(i, j): distance}``."""
    matrix = np.full((n, n), default, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    for (i, j), distance in pairs.items():
        matrix[i][j] = matrix[j][i] = distance
    return matrix


#: Two plans far apart, and three forming a well-separated cluster with nothing
#: linking the two groups. The trap for a greedy solver: the best *pair* (0, 1)
#: is in none of the best triples, so opening with it strands the search.
GREEDY_TRAP = matrix_from_pairs(
    5, {(0, 1): 1.0, (2, 3): 0.9, (2, 4): 0.9, (3, 4): 0.9}
)

#: Every pair equally distant, so every subset of a given size ties.
ALL_TIED = matrix_from_pairs(3, {}, default=0.5)

#: Six plans, every pair a different non-zero distance. Used wherever a zero or
#: a coincidence would hide a bug -- a set whose distances sum to zero cannot
#: tell "add the new pairs to the running total" from "replace it".
DENSE = matrix_from_pairs(
    6,
    {
        (0, 1): 0.90, (0, 2): 0.20, (0, 3): 0.50, (0, 4): 0.70, (0, 5): 0.30,
        (1, 2): 0.60, (1, 3): 0.10, (1, 4): 0.80, (1, 5): 0.40,
        (2, 3): 0.75, (2, 4): 0.35, (2, 5): 0.55,
        (3, 4): 0.45, (3, 5): 0.65,
        (4, 5): 0.15,
    },
)
DENSE_PLANS = list(range(6))

PLANS = list(range(5))
MODELS = [MaxSum, MaxMean, MaxMin]
SOLVERS = [GreedySolver, ExactSolver]


def model_over(matrix, model_class=MaxSum, **kwargs):
    return model_class([MatrixMetric(matrix)], **kwargs)


def brute_force(model, plans, k):
    """The best-scoring subset of size k, by looking at every one of them."""
    return max(
        (model([plans[i] for i in subset]) for subset in combinations(range(len(plans)), k)),
        default=0.0,
    )


class TestGreedy:
    """The ForbidIterative selection step: best pair, then best addition."""

    def test_it_opens_with_the_two_furthest_apart_plans(self):
        selection = GreedySolver(model_over(GREEDY_TRAP)).select(PLANS, k=2)
        assert set(selection.indices) == {0, 1}
        assert selection.score == pytest.approx(1.0)

    def test_it_then_adds_the_plan_that_most_improves_the_set(self):
        # Distances to the opening pair (0, 1): plan 2 is closer to neither, but
        # plan 4 is 0.6 from plan 1 and so adds the most.
        matrix = matrix_from_pairs(
            5, {(0, 1): 1.0, (1, 2): 0.2, (1, 3): 0.4, (1, 4): 0.6}
        )
        selection = GreedySolver(model_over(matrix)).select(PLANS, k=3)
        assert selection.indices[:2] == (0, 1)
        assert selection.indices[2] == 4

    def test_the_greedy_step_follows_the_model(self):
        """MaxSum wants the largest total distance to the chosen set; MaxMin
        wants the largest distance to the *nearest* chosen plan."""
        matrix = matrix_from_pairs(
            4,
            {
                (0, 1): 1.0,
                # Plan 2 totals 0.9 but comes within 0.1 of plan 1.
                (0, 2): 0.8, (1, 2): 0.1,
                # Plan 3 totals 0.8 but stays 0.4 from both.
                (0, 3): 0.4, (1, 3): 0.4,
            },
        )
        plans = list(range(4))
        assert GreedySolver(model_over(matrix, MaxSum)).select(plans, k=3).indices == (
            0, 1, 2,
        )
        assert GreedySolver(model_over(matrix, MaxMin)).select(plans, k=3).indices == (
            0, 1, 3,
        )

    def test_it_returns_exactly_k_plans(self):
        for k in range(2, 6):
            assert len(GreedySolver(model_over(GREEDY_TRAP)).select(PLANS, k)) == k

    def test_it_never_repeats_a_plan(self):
        selection = GreedySolver(model_over(GREEDY_TRAP)).select(PLANS, k=4)
        assert len(set(selection.indices)) == 4

    def test_it_is_marked_as_not_proven_optimal(self):
        assert GreedySolver(model_over(GREEDY_TRAP)).select(PLANS, k=3).optimal is False

    def test_all_identical_plans_do_not_pair_a_plan_with_itself(self):
        """Every distance is zero, so nothing distinguishes the candidates."""
        selection = GreedySolver(model_over(np.zeros((4, 4)))).select(
            list(range(4)), k=3
        )
        assert len(set(selection.indices)) == 3
        assert selection.score == 0.0


class TestCostTieBreaking:
    """ForbidIterative orders plans by cost and works from the cheapest up."""

    def test_ties_go_to_the_cheaper_plans(self):
        plans = list(range(3))
        solver = GreedySolver(model_over(ALL_TIED))
        assert solver.select(plans, k=2).indices == (0, 1)
        # Plans 1 and 2 are now the cheapest, so they win the same tie.
        assert solver.select(plans, k=2, costs=[3, 1, 2]).indices == (1, 2)

    def test_equal_costs_keep_the_given_order(self):
        plans = list(range(3))
        solver = GreedySolver(model_over(ALL_TIED))
        assert solver.select(plans, k=2, costs=[1, 1, 1]).indices == (0, 1)

    def test_costs_do_not_override_diversity(self):
        """Cost only breaks ties; it never makes a worse set win."""
        selection = GreedySolver(model_over(GREEDY_TRAP)).select(
            PLANS, k=2, costs=[9, 9, 1, 1, 1]
        )
        assert set(selection.indices) == {0, 1}

    def test_the_wrong_number_of_costs_is_rejected(self):
        with pytest.raises(ValueError, match="expected 5 costs"):
            GreedySolver(model_over(GREEDY_TRAP)).select(PLANS, k=3, costs=[1, 2])

    @pytest.mark.parametrize("solver_class", [GreedySolver, ExactSolver])
    def test_both_solvers_accept_costs(self, solver_class):
        selection = solver_class(model_over(ALL_TIED)).select(
            list(range(3)), k=2, costs=[3, 1, 2]
        )
        assert selection.indices == (1, 2)


class TestExact:
    """The optimum, verified against every subset."""

    @pytest.mark.parametrize("model_class", MODELS)
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_it_matches_brute_force(self, model_class, k):
        model = model_over(GREEDY_TRAP, model_class)
        selection = ExactSolver(model).select(PLANS, k)
        assert selection.score == pytest.approx(brute_force(model, PLANS, k))
        assert selection.optimal is True

    @pytest.mark.parametrize("model_class", MODELS)
    @pytest.mark.parametrize("seed", range(8))
    def test_it_matches_brute_force_on_random_geometries(self, model_class, seed):
        """Pruning must never cut off the branch holding the optimum."""
        rng = np.random.default_rng(seed)
        n = 7
        upper = rng.random((n, n))
        matrix = np.triu(upper, 1)
        matrix = matrix + matrix.T
        model = model_over(matrix, model_class)
        plans = list(range(n))
        for k in (2, 3, 4):
            selection = ExactSolver(model).select(plans, k)
            assert selection.score == pytest.approx(brute_force(model, plans, k)), (
                f"{model_class.__name__} k={k} seed={seed}"
            )

    def test_it_finds_the_cluster_the_greedy_solver_misses(self):
        """The whole reason to have an exact solver."""
        model = model_over(GREEDY_TRAP, MaxSum)
        greedy = GreedySolver(model).select(PLANS, k=3)
        exact = ExactSolver(model).select(PLANS, k=3)
        assert set(exact.indices) == {2, 3, 4}
        assert exact.score == pytest.approx(2.7)
        assert greedy.score == pytest.approx(1.0)

    @pytest.mark.parametrize("model_class", MODELS)
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_the_greedy_solver_never_beats_it(self, model_class, k):
        model = model_over(GREEDY_TRAP, model_class)
        assert (
            GreedySolver(model).select(PLANS, k).score
            <= ExactSolver(model).select(PLANS, k).score
        )

    def test_a_search_that_is_too_large_is_refused_rather_than_started(self):
        with pytest.raises(ValueError, match="over the max_subsets limit"):
            ExactSolver(model_over(GREEDY_TRAP), max_subsets=5).select(PLANS, k=3)

    def test_the_guard_can_be_lifted(self):
        solver = ExactSolver(model_over(GREEDY_TRAP), max_subsets=None)
        assert len(solver.select(PLANS, k=3)) == 3


class TestBounded:
    """bD-k: k plans whose diversity clears a bound, or nothing."""

    def test_it_returns_a_set_meeting_the_bound(self):
        model = model_over(GREEDY_TRAP, MaxMin)
        selection = ExactSolver(model, bound=0.5).select(PLANS, k=3)
        assert selection.score >= 0.5
        assert set(selection.indices) == {2, 3, 4}

    def test_a_bounded_answer_is_not_claimed_to_be_optimal(self):
        selection = ExactSolver(model_over(GREEDY_TRAP, MaxMin), bound=0.5).select(
            PLANS, k=3
        )
        assert selection.optimal is False

    def test_an_unreachable_bound_has_no_solution(self):
        """As in classical planning, a super-optimal bound is unsolvable."""
        with pytest.raises(ValueError, match="no set of 3 plans reaches"):
            ExactSolver(model_over(GREEDY_TRAP, MaxMin), bound=0.95).select(PLANS, k=3)

    def test_a_bound_the_optimum_exactly_meets_is_satisfied(self):
        model = model_over(GREEDY_TRAP, MaxMin)
        assert ExactSolver(model, bound=0.9).select(PLANS, k=3).score == pytest.approx(
            0.9
        )

    @pytest.mark.parametrize("model_class", MODELS)
    def test_a_bound_of_zero_is_met_by_anything(self, model_class):
        selection = ExactSolver(model_over(GREEDY_TRAP, model_class), bound=0.0).select(
            PLANS, k=3
        )
        assert len(selection) == 3


class TestSolverContract:
    """Shared behaviour, checked against both solvers."""

    @pytest.mark.parametrize("solver_class", SOLVERS)
    @pytest.mark.parametrize("model_class", MODELS)
    def test_the_reported_score_is_the_score_of_the_chosen_plans(
        self, solver_class, model_class
    ):
        """The number returned must be what the model says about that subset,
        not something the solver computed along the way."""
        model = model_over(GREEDY_TRAP, model_class)
        selection = solver_class(model).select(PLANS, k=3)
        assert selection.score == model(selection.plans)

    @pytest.mark.parametrize("solver_class", SOLVERS)
    def test_the_chosen_plans_match_the_reported_indices(self, solver_class):
        selection = solver_class(model_over(GREEDY_TRAP)).select(PLANS, k=3)
        assert selection.plans == [PLANS[index] for index in selection.indices]

    @pytest.mark.parametrize("solver_class", SOLVERS)
    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
    def test_it_returns_the_requested_number_of_plans(self, solver_class, k):
        assert len(solver_class(model_over(GREEDY_TRAP)).select(PLANS, k)) == k

    @pytest.mark.parametrize("solver_class", SOLVERS)
    def test_asking_for_more_plans_than_exist_returns_all_of_them(self, solver_class):
        """A set smaller than requested is still a valid solution."""
        selection = solver_class(model_over(GREEDY_TRAP)).select(PLANS, k=99)
        assert selection.indices == (0, 1, 2, 3, 4)
        assert selection.optimal is True

    @pytest.mark.parametrize("solver_class", SOLVERS)
    def test_taking_every_plan_is_trivially_optimal(self, solver_class):
        """With k equal to the plan count there is nothing to choose between,
        so even the greedy solver's answer is the best one available."""
        selection = solver_class(model_over(GREEDY_TRAP)).select(PLANS, k=len(PLANS))
        assert selection.indices == (0, 1, 2, 3, 4)
        assert selection.optimal is True

    @pytest.mark.parametrize("solver_class", SOLVERS)
    @pytest.mark.parametrize("k", [0, 1])
    def test_fewer_than_two_plans_have_no_pairs_to_score(self, solver_class, k):
        selection = solver_class(model_over(GREEDY_TRAP)).select(PLANS, k)
        assert selection.score == 0.0
        assert selection.optimal is True

    @pytest.mark.parametrize("solver_class", SOLVERS)
    def test_a_negative_size_is_rejected(self, solver_class):
        with pytest.raises(ValueError, match="cannot select"):
            solver_class(model_over(GREEDY_TRAP)).select(PLANS, k=-1)

    @pytest.mark.parametrize("solver_class", SOLVERS)
    def test_an_empty_plan_set(self, solver_class):
        selection = solver_class(model_over(GREEDY_TRAP)).select([], k=3)
        assert selection.plans == []
        assert selection.score == 0.0

    @pytest.mark.parametrize("solver_class", SOLVERS)
    def test_repeated_calls_give_the_same_answer(self, solver_class):
        solver = solver_class(model_over(GREEDY_TRAP))
        first = solver.select(PLANS, k=3)
        assert solver.select(PLANS, k=3).indices == first.indices
        assert solver.select(PLANS, k=3).indices == first.indices

    @pytest.mark.parametrize("solver_class", SOLVERS)
    def test_a_planset_may_be_any_iterable(self, solver_class):
        solver = solver_class(model_over(GREEDY_TRAP))
        assert solver.select(iter(PLANS), k=3).indices == solver.select(PLANS, k=3).indices

    def test_solver_is_abstract(self):
        with pytest.raises(TypeError):
            Solver(model_over(GREEDY_TRAP))

    def test_repr_names_the_solver_and_its_model(self, task):
        solver = GreedySolver(MaxSum([Stability(task)]))
        assert repr(solver) == "GreedySolver(MaxSum([Stability], normalize=False))"

    def test_str_is_the_solver_name(self):
        assert str(GreedySolver(model_over(GREEDY_TRAP))) == "Greedy"
        assert str(ExactSolver(model_over(GREEDY_TRAP))) == "Exact"


class TestModelSubsetScoring:
    """The hooks solvers score subsets through must agree with the model."""

    @pytest.mark.parametrize("model_class", MODELS)
    @pytest.mark.parametrize("normalize", [False, True])
    def test_score_matrix_agrees_with_scoring_the_plans(self, model_class, normalize):
        model = model_over(GREEDY_TRAP, model_class, normalize=normalize)
        assert model.score_matrix(model.distance_matrix(PLANS)) == model(PLANS)

    @pytest.mark.parametrize("model_class", MODELS)
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("chosen", [[0, 1], [0, 1, 2], [0, 1, 2, 3]])
    def test_score_additions_agrees_with_scoring_each_extension(
        self, model_class, normalize, chosen
    ):
        """Every model overrides score_additions with a fast path that folds a
        new plan into the score without re-reducing the set. This is what keeps
        those overrides honest -- the same contract as Metric.pairwise.

        Run over DENSE and over several sizes on purpose: an override that
        dropped the running total would still look right on a set whose pair
        distances are all zero, and one that confused the plan count with the
        pair count would still look right at size three, where they are equal.
        """
        model = model_over(DENSE, model_class, normalize=normalize)
        matrix = model.distance_matrix(DENSE_PLANS)
        size = len(chosen) + 1
        pairs = np.array([matrix[i][j] for i, j in combinations(chosen, 2)])
        candidates = [plan for plan in DENSE_PLANS if plan not in chosen]
        additions = matrix[np.ix_(candidates, chosen)]

        fast = model.score_additions(pairs, additions, size)
        # Model.score_additions unbound: the generic definition the overrides
        # replace, called on a model that has replaced it.
        generic = Model.score_additions(model, pairs, additions, size)
        assert np.allclose(fast, generic)
        # ...and both agree with simply scoring the resulting plan sets.
        assert np.allclose(fast, [model(chosen + [c]) for c in candidates])

    @pytest.mark.parametrize("model_class", MODELS)
    @pytest.mark.parametrize("k", [3, 4])
    def test_optimistic_score_is_never_pessimistic(self, model_class, k):
        """Pruning is only sound if this over-estimates every completion."""
        model = model_over(DENSE, model_class)
        matrix = model.distance_matrix(DENSE_PLANS)
        ceiling = float(matrix.max())
        for start in combinations(DENSE_PLANS, 2):
            pairs = np.array([matrix[start[0]][start[1]]])
            estimate = model.optimistic_score(pairs, 2, k, ceiling)
            completions = [
                model(list(subset))
                for subset in combinations(DENSE_PLANS, k)
                if set(start) <= set(subset)
            ]
            assert estimate >= max(completions) - 1e-9

    def test_optimistic_score_rejects_an_oversized_set(self):
        model = model_over(GREEDY_TRAP)
        with pytest.raises(ValueError, match="already larger"):
            model.optimistic_score(np.zeros(10), size=5, target=3)
