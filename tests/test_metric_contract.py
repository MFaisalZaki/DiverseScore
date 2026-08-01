"""Invariants every metric must satisfy, checked against all of them at once.

These are the tests that would have caught the defects this package was
refactored out of: metrics that returned a similarity where a distance was
expected, and a ``pairwise`` that implemented a different comparison from the
``__call__`` beside it.
"""

import numpy as np
import pytest

from plandiversity.shortcuts import Metric, Stability, States, Uniqueness


@pytest.fixture
def metric_factories():
    """Every metric, as a ``name -> (task, plans) -> Metric`` mapping."""
    return {
        "Stability": Stability,
        "States": States,
        "Uniqueness": Uniqueness,
    }


ALL_METRICS = ["Stability", "States", "Uniqueness"]


@pytest.fixture
def build(metric_factories):
    def _build(name, task, plans=None):
        return metric_factories[name](task, plans)

    return _build


@pytest.mark.parametrize("name", ALL_METRICS)
class TestDistanceContract:
    """Each metric reports a distance in [0, 1], not a similarity."""

    def test_identical_plans_are_zero_distance(self, name, build, task, planset):
        metric = build(name, task, planset)
        for plan in planset:
            assert metric(plan, plan) == 0.0

    def test_distances_are_within_the_unit_interval(self, name, build, task, planset):
        metric = build(name, task, planset)
        distances = metric.pairwise(planset)
        assert distances.min() >= 0.0
        assert distances.max() <= 1.0

    def test_distance_is_symmetric(self, name, build, task, planset):
        metric = build(name, task, planset)
        a, b, _ = planset
        assert metric(a, b) == metric(b, a)

    def test_at_least_one_pair_of_the_planset_is_distinguished(
        self, name, build, task, planset
    ):
        """A metric that returned a constant would pass every other test here."""
        assert build(name, task, planset).pairwise(planset).max() > 0.0


@pytest.mark.parametrize("name", ALL_METRICS)
class TestPairwiseMatchesCall:
    """``pairwise(plans)[i][j]`` must equal ``metric(plans[i], plans[j])``.

    Several metrics override ``pairwise`` with a vectorised implementation.
    This is what keeps those overrides honest.
    """

    def test_every_entry_matches_the_scalar_call(self, name, build, task, planset):
        metric = build(name, task, planset)
        distances = metric.pairwise(planset)
        for i, plan_a in enumerate(planset):
            for j, plan_b in enumerate(planset):
                assert distances[i][j] == pytest.approx(metric(plan_a, plan_b)), (
                    f"{name} disagrees with itself at ({i}, {j})"
                )

    def test_matrix_is_square_symmetric_and_zero_on_the_diagonal(
        self, name, build, task, planset
    ):
        distances = build(name, task, planset).pairwise(planset)
        assert distances.shape == (len(planset), len(planset))
        assert np.allclose(distances, distances.T)
        assert np.allclose(np.diag(distances), 0.0)

    def test_matrix_is_float_valued(self, name, build, task, planset):
        """Models sum these matrices; a boolean one would silently saturate."""
        assert build(name, task, planset).pairwise(planset).dtype == np.float64


@pytest.mark.parametrize("name", ALL_METRICS)
class TestCaching:
    """Caching plans at construction is an optimisation, never a behaviour."""

    def test_warm_and_cold_metrics_agree(self, name, build, task, planset):
        warm = build(name, task, planset)
        cold = build(name, task, None)
        assert np.allclose(warm.pairwise(planset), cold.pairwise(planset))

    def test_plans_absent_from_the_cache_are_scored_correctly(
        self, name, build, task, planset, plan_l1_then_l2_long
    ):
        """The cache is keyed by object identity; an uncached plan must not
        collide with a cached one."""
        warm = build(name, task, planset)
        cold = build(name, task, None)
        for plan in planset:
            assert warm(plan, plan_l1_then_l2_long) == pytest.approx(
                cold(plan, plan_l1_then_l2_long)
            )

    def test_repeated_calls_are_stable(self, name, build, task, planset):
        metric = build(name, task, planset)
        first = metric.pairwise(planset)
        assert np.allclose(first, metric.pairwise(planset))
        assert np.allclose(first, metric.pairwise(planset))

    def test_pairwise_defaults_to_the_plans_given_at_construction(
        self, name, build, task, planset
    ):
        metric = build(name, task, planset)
        assert np.allclose(metric.pairwise(), metric.pairwise(planset))


@pytest.mark.parametrize("name", ALL_METRICS)
class TestDegenerateInput:
    def test_empty_planset(self, name, build, task):
        assert build(name, task, []).pairwise([]).shape == (0, 0)

    def test_single_plan(self, name, build, task, plan_l1_then_l2):
        distances = build(name, task, [plan_l1_then_l2]).pairwise([plan_l1_then_l2])
        assert distances.shape == (1, 1)
        assert distances[0][0] == 0.0

    def test_two_copies_of_one_plan_are_zero_distance(
        self, name, build, task, plan_l1_then_l2
    ):
        pair = [plan_l1_then_l2, plan_l1_then_l2]
        assert np.allclose(build(name, task, pair).pairwise(pair), 0.0)


class TestMetricBase:
    def test_metric_is_abstract(self, task):
        with pytest.raises(TypeError):
            Metric(task)

    def test_subclass_must_implement_both_hooks(self, task):
        class Incomplete(Metric):
            def _feature(self, plan):
                return None

        with pytest.raises(TypeError):
            Incomplete(task)

    def test_str_is_the_metric_name(self, task):
        assert str(Stability(task)) == "Stability"
        assert str(States(task)) == "States"

    def test_repr_names_the_class(self, task):
        assert repr(Stability(task)) == "Stability()"

    def test_a_custom_metric_needs_only_the_two_hooks(self, task, planset):
        """The extension point the README documents."""

        class PlanLength(Metric):
            name = "PlanLength"

            def _feature(self, plan):
                return len(plan.actions)

            def _distance(self, a, b):
                return abs(a - b) / max(a, b, 1)

        metric = PlanLength(task, planset)
        # Every plan in the set has four actions.
        assert np.allclose(metric.pairwise(planset), 0.0)
        assert str(metric) == "PlanLength"
