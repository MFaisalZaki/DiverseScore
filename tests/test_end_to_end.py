"""The README's workflow, end to end: PDDL files in, a diversity score out."""

import pytest

from plandiversity.shortcuts import (
    ExactSolver,
    GreedySolver,
    MaxMean,
    MaxMin,
    MaxSum,
    PDDLReader,
    Stability,
    States,
    Uniqueness,
)

DOMAIN_PDDL = """
(define (domain transport)
  (:requirements :strips :typing)
  (:types location truck)
  (:predicates (at ?t - truck ?l - location)
               (delivered ?l - location))
  (:action move
     :parameters (?t - truck ?f - location ?to - location)
     :precondition (at ?t ?f)
     :effect (and (not (at ?t ?f)) (at ?t ?to)))
  (:action drop
     :parameters (?t - truck ?l - location)
     :precondition (at ?t ?l)
     :effect (delivered ?l)))
"""

PROBLEM_PDDL = """
(define (problem transport-1)
  (:domain transport)
  (:objects l0 l1 l2 - location
            tr1 tr2 - truck)
  (:init (at tr1 l0) (at tr2 l0))
  (:goal (and (delivered l1) (delivered l2))))
"""

PLANS = [
    # tr1 delivers l1, then l2.
    "(move tr1 l0 l1)\n(drop tr1 l1)\n(move tr1 l1 l2)\n(drop tr1 l2)\n",
    # tr1 delivers l2, then l1.
    "(move tr1 l0 l2)\n(drop tr1 l2)\n(move tr1 l2 l1)\n(drop tr1 l1)\n",
    # A truck each.
    "(move tr1 l0 l1)\n(drop tr1 l1)\n(move tr2 l0 l2)\n(drop tr2 l2)\n",
]


@pytest.fixture
def pddl_task(tmp_path):
    domain = tmp_path / "domain.pddl"
    problem = tmp_path / "problem.pddl"
    domain.write_text(DOMAIN_PDDL)
    problem.write_text(PROBLEM_PDDL)
    return PDDLReader().parse_problem(str(domain), str(problem))


#: A larger candidate pool, of the shape a diverse planner actually produces:
#: several near-duplicates plus a few genuinely different plans. Plans 0 and 3
#: use the same actions in a different order, so no metric here separates them.
CANDIDATE_PLANS = PLANS + [
    # A reordering of plan 0 -- same actions, delivered in the same order.
    "(move tr1 l0 l1)\n(drop tr1 l1)\n(move tr1 l1 l2)\n(drop tr1 l2)\n",
    # tr2 does all the work.
    "(move tr2 l0 l1)\n(drop tr2 l1)\n(move tr2 l1 l2)\n(drop tr2 l2)\n",
    # A truck each, the other way round.
    "(move tr2 l0 l1)\n(drop tr2 l1)\n(move tr1 l0 l2)\n(drop tr1 l2)\n",
    # tr1 with a pointless detour through l2 first.
    (
        "(move tr1 l0 l2)\n(move tr1 l2 l0)\n(move tr1 l0 l1)\n(drop tr1 l1)\n"
        "(move tr1 l1 l2)\n(drop tr1 l2)\n"
    ),
]


@pytest.fixture
def pddl_planset(pddl_task):
    return [PDDLReader().parse_plan_string(pddl_task, plan) for plan in PLANS]


@pytest.fixture
def candidate_planset(pddl_task):
    return [
        PDDLReader().parse_plan_string(pddl_task, plan) for plan in CANDIDATE_PLANS
    ]


class TestReadmeWorkflow:
    def test_parsing_yields_the_expected_plans(self, pddl_planset):
        assert len(pddl_planset) == 3
        assert all(len(plan.actions) == 4 for plan in pddl_planset)

    def test_a_single_metric_scores_the_set(self, pddl_task, pddl_planset):
        score = MaxSum([Stability(pddl_task)])(pddl_planset)
        # The same three plans as the hand-built fixtures: 2/3 + 2/3 + 6/7.
        assert score == pytest.approx(2 / 3 + 2 / 3 + 6 / 7, abs=1e-5)

    def test_several_metrics_combine_into_one_score(self, pddl_task, pddl_planset):
        score = MaxMean(
            [
                Stability(pddl_task, pddl_planset),
                States(pddl_task, pddl_planset),
                Uniqueness(pddl_task, pddl_planset),
            ]
        )(pddl_planset)
        assert 0.0 < score <= 1.0

    @pytest.mark.parametrize("model_class", [MaxSum, MaxMean, MaxMin])
    @pytest.mark.parametrize(
        "metric_class", [Stability, States, Uniqueness]
    )
    def test_every_model_and_metric_pairing_runs(
        self, pddl_task, pddl_planset, model_class, metric_class
    ):
        score = model_class([metric_class(pddl_task, pddl_planset)])(pddl_planset)
        assert score >= 0.0

    def test_the_grounded_and_hand_built_tasks_agree(
        self, pddl_task, pddl_planset, task, planset
    ):
        """The PDDL task in this file is the conftest task written out, so the
        two must score identically."""
        for metric_class in (Stability, States, Uniqueness):
            from_pddl = MaxSum([metric_class(pddl_task, pddl_planset)])(pddl_planset)
            from_python = MaxSum([metric_class(task, planset)])(planset)
            assert from_pddl == pytest.approx(from_python), metric_class.name


class TestSelectionWorkflow:
    """Generate many plans, keep the k most diverse -- the ForbidIterative shape."""

    @pytest.mark.parametrize("solver_class", [GreedySolver, ExactSolver])
    @pytest.mark.parametrize("model_class", [MaxSum, MaxMean, MaxMin])
    def test_selecting_from_a_candidate_pool(
        self, pddl_task, candidate_planset, solver_class, model_class
    ):
        model = model_class([Stability(pddl_task, candidate_planset)])
        selection = solver_class(model).select(candidate_planset, k=3)
        assert len(selection) == 3
        assert selection.score == model(selection.plans)

    @pytest.mark.parametrize("solver_class", [GreedySolver, ExactSolver])
    def test_the_duplicate_plan_is_left_behind(
        self, pddl_task, candidate_planset, solver_class
    ):
        """Plans 0 and 3 use the same actions. Under MaxMin, keeping both pins
        the score to zero, so no selector that is working should keep both."""
        model = MaxMin([Stability(pddl_task, candidate_planset)])
        selection = solver_class(model).select(candidate_planset, k=3)
        assert not {0, 3} <= set(selection.indices)
        assert selection.score > 0.0

    def test_a_selected_subset_beats_taking_the_first_k(
        self, pddl_task, candidate_planset
    ):
        """The point of selecting at all."""
        model = MaxSum([Stability(pddl_task, candidate_planset)])
        selected = GreedySolver(model).select(candidate_planset, k=4)
        assert selected.score > model(candidate_planset[:4])

    def test_selecting_with_several_metrics(self, pddl_task, candidate_planset):
        model = MaxSum(
            [
                Stability(pddl_task, candidate_planset),
                States(pddl_task, candidate_planset),
                Uniqueness(pddl_task, candidate_planset),
            ]
        )
        selection = GreedySolver(model).select(candidate_planset, k=3)
        assert len(selection) == 3
        assert selection.score > 0.0

    def test_cheaper_plans_win_ties(self, pddl_task, candidate_planset):
        """Costs here are plan lengths: the detour plan is the expensive one."""
        costs = [len(plan.actions) for plan in candidate_planset]
        model = MaxSum([Uniqueness(pddl_task, candidate_planset)])
        # Uniqueness scores every distinct pair 1.0, so every subset without
        # the duplicate pair ties and only cost separates them.
        selection = GreedySolver(model).select(candidate_planset, k=3, costs=costs)
        detour = len(candidate_planset) - 1
        assert detour not in selection.indices

    def test_the_greedy_and_exact_solvers_agree_on_an_easy_pool(
        self, pddl_task, candidate_planset
    ):
        """With k=2 the best pair is the answer, so both must find it."""
        model = MaxSum([Stability(pddl_task, candidate_planset)])
        greedy = GreedySolver(model).select(candidate_planset, k=2)
        exact = ExactSolver(model).select(candidate_planset, k=2)
        assert greedy.score == exact.score

    @pytest.mark.parametrize("model_class", [MaxSum, MaxMean, MaxMin])
    def test_the_exact_solver_is_never_beaten_on_real_plans(
        self, pddl_task, candidate_planset, model_class
    ):
        model = model_class([Stability(pddl_task, candidate_planset)])
        for k in (2, 3, 4):
            greedy = GreedySolver(model).select(candidate_planset, k)
            exact = ExactSolver(model).select(candidate_planset, k)
            assert greedy.score <= exact.score
