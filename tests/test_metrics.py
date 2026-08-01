"""Per-metric behaviour, against values worked out by hand on the tiny task.

The task and its plans are described in ``conftest.py``. Its nine grounded
boolean fluents are ``at(t, l)`` for two trucks and three locations, plus
``delivered(l)`` for three locations; ``fuel`` is numeric and takes no part.
"""

import numpy as np
import pytest

from plandiversity.shortcuts import (
    InapplicablePlanError,
    Stability,
    States,
    Uniqueness,
)


class TestStability:
    """Jaccard distance over action sets."""

    def test_reordering_the_same_actions_is_zero_distance(
        self, task, domain, make_plan, plan_l1_then_l2
    ):
        reversed_plan = make_plan(*reversed(list_of_steps(domain, plan_l1_then_l2)))
        assert Stability(task)(plan_l1_then_l2, reversed_plan) == 0.0

    def test_plans_sharing_half_their_actions(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        # Both plans have four actions and share drop(tr1, l1) and drop(tr1, l2):
        # 1 - 2/6.
        assert Stability(task)(plan_l1_then_l2, plan_l2_then_l1) == pytest.approx(2 / 3)

    def test_plans_sharing_one_action(self, task, plan_l2_then_l1, plan_two_trucks):
        # Only drop(tr1, l1) is common: 1 - 1/7.
        assert Stability(task)(plan_l2_then_l1, plan_two_trucks) == pytest.approx(6 / 7)

    def test_plans_sharing_no_action_are_maximally_distant(
        self, task, domain, make_plan
    ):
        move, tr1, tr2 = domain["move"], domain["tr1"], domain["tr2"]
        l0, l1, l2 = domain["l0"], domain["l1"], domain["l2"]
        assert Stability(task)(
            make_plan((move, (tr1, l0, l1))), make_plan((move, (tr2, l0, l2)))
        ) == 1.0

    def test_two_empty_plans_are_identical_not_undefined(self, task, empty_plan):
        assert Stability(task)(empty_plan, empty_plan) == 0.0

    def test_an_empty_plan_is_maximally_distant_from_a_non_empty_one(
        self, task, empty_plan, plan_l1_then_l2
    ):
        assert Stability(task)(empty_plan, plan_l1_then_l2) == 1.0

    def test_repeated_actions_are_collapsed(self, task, domain, make_plan):
        move, tr1, l0, l1 = domain["move"], domain["tr1"], domain["l0"], domain["l1"]
        once = make_plan((move, (tr1, l0, l1)))
        twice = make_plan((move, (tr1, l0, l1)), (move, (tr1, l0, l1)))
        assert Stability(task)(once, twice) == 0.0


class TestUniqueness:
    """Exact-match distance over action sets."""

    def test_different_action_sets_are_maximally_distant(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        assert Uniqueness(task)(plan_l1_then_l2, plan_l2_then_l1) == 1.0

    def test_a_reordering_is_not_unique(self, task, domain, make_plan, plan_l1_then_l2):
        reordered = make_plan(*reversed(list_of_steps(domain, plan_l1_then_l2)))
        assert Uniqueness(task)(plan_l1_then_l2, reordered) == 0.0

    def test_it_is_the_all_or_nothing_reading_of_stability(
        self, task, planset, plan_l1_then_l2_long
    ):
        """Uniqueness is 1.0 wherever Stability is non-zero, and 0.0 where it is 0."""
        plans = planset + [plan_l1_then_l2_long]
        stability = Stability(task, plans).pairwise(plans)
        uniqueness = Uniqueness(task, plans).pairwise(plans)
        assert np.array_equal(uniqueness, (stability > 0).astype(float))

    def test_two_empty_plans_are_identical(self, task, empty_plan):
        assert Uniqueness(task)(empty_plan, empty_plan) == 0.0

    def test_groups_of_equal_plans_are_matched_across_the_whole_matrix(
        self, task, plan_l1_then_l2, plan_l2_then_l1, domain, make_plan
    ):
        twin = make_plan(*list_of_steps(domain, plan_l1_then_l2))
        plans = [plan_l1_then_l2, plan_l2_then_l1, twin]
        distances = Uniqueness(task, plans).pairwise(plans)
        # Plans 0 and 2 are distinct objects with the same actions.
        assert distances[0][2] == 0.0
        assert distances[0][1] == 1.0
        assert distances[1][2] == 1.0


class TestStates:
    """Jaccard over the propositions true at each step, averaged over the longer plan."""

    def test_hand_computed_distance_for_equal_length_plans(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        # Step-by-step Jaccard similarities are 1, 1/3, 1/5, 1/5 and 3/5;
        # both plans have five states, so 1 - (7/3) / 5.
        assert States(task)(plan_l1_then_l2, plan_l2_then_l1) == pytest.approx(
            1 - (1 + 1 / 3 + 1 / 5 + 1 / 5 + 3 / 5) / 5
        )

    def test_hand_computed_distance_for_plans_that_diverge_late(
        self, task, plan_l1_then_l2, plan_two_trucks
    ):
        # The first three states agree; then 1/5 and 1/3.
        assert States(task)(plan_l1_then_l2, plan_two_trucks) == pytest.approx(
            1 - (1 + 1 + 1 + 1 / 5 + 1 / 3) / 5
        )

    def test_a_prefix_is_not_free(self, task, domain, make_plan, plan_l1_then_l2):
        """A plan and its own strict prefix must differ.

        Their first three states are identical, so every state the two have in
        common matches; the distance comes entirely from the two steps the
        prefix never took. Averaging over the shorter plan instead of the
        longer would score this pair 0.0 and make every prefix of a plan look
        like the plan itself.
        """
        move, drop = domain["move"], domain["drop"]
        tr1, l0, l1 = domain["tr1"], domain["l0"], domain["l1"]
        prefix = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)))
        # Three matching states out of the longer plan's five: 1 - 3/5.
        assert States(task)(plan_l1_then_l2, prefix) == pytest.approx(0.4)

    def test_unequal_length_plans_stay_within_the_unit_interval(
        self, task, plan_l1_then_l2, plan_l1_then_l2_long
    ):
        distance = States(task)(plan_l1_then_l2, plan_l1_then_l2_long)
        assert 0.0 < distance <= 1.0

    def test_an_empty_plan_against_itself_is_zero(self, task, empty_plan):
        assert States(task)(empty_plan, empty_plan) == 0.0

    def test_an_empty_plan_shares_only_the_initial_state(
        self, task, empty_plan, plan_l1_then_l2
    ):
        # One matching state (the initial one) out of the other plan's five.
        assert States(task)(empty_plan, plan_l1_then_l2) == pytest.approx(1 - 1 / 5)

    def test_a_state_is_the_whole_assignment_not_the_action_effect(
        self, task, domain, make_plan
    ):
        """Two plans that move different trucks differ in the fluents they leave
        untouched as well as the ones they change.

        A UPState only stores what its action assigned and defers the rest to
        its ancestors, so a metric that reads a state's own values compares
        effects rather than states -- and would score this pair 1.0.
        """
        move = domain["move"]
        tr1, tr2, l0, l1 = domain["tr1"], domain["tr2"], domain["l0"], domain["l1"]
        moved_tr1 = make_plan((move, (tr1, l0, l1)))
        moved_tr2 = make_plan((move, (tr2, l0, l1)))
        # After the move the states are {at(tr1,l1), at(tr2,l0)} and
        # {at(tr1,l0), at(tr2,l1)}: disjoint, so step similarity 0, against an
        # identical initial state. 1 - (1 + 0) / 2.
        assert States(task)(moved_tr1, moved_tr2) == pytest.approx(0.5)

    def test_numeric_fluents_are_ignored(self, task, domain, make_plan):
        """fuel differs between these plans; only booleans are compared."""
        move, drop = domain["move"], domain["drop"]
        tr1, l0, l1 = domain["tr1"], domain["l0"], domain["l1"]
        one_move = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)))
        # Same boolean trajectory, different fuel, via a no-op-ish self-move.
        assert States(task)(one_move, one_move) == 0.0

    def test_an_inapplicable_plan_is_rejected(self, task, inapplicable_plan):
        with pytest.raises(InapplicablePlanError, match="not applicable"):
            States(task)(inapplicable_plan, inapplicable_plan)

    def test_an_inapplicable_plan_is_rejected_at_construction(
        self, task, inapplicable_plan
    ):
        with pytest.raises(InapplicablePlanError):
            States(task, [inapplicable_plan])

    def test_empty_state_sequences_do_not_divide_by_zero(self, task):
        """Simulation always yields at least the initial state, so this guards
        the ``_distance`` hook rather than any path through the metric."""
        assert States(task)._distance([], []) == 0.0


def list_of_steps(domain, plan):
    """Recover ``(action, (objects, ...))`` pairs from a plan, for rebuilding it."""
    return [
        (
            domain[instance.action.name],
            tuple(domain[str(parameter)] for parameter in instance.actual_parameters),
        )
        for instance in plan.actions
    ]
