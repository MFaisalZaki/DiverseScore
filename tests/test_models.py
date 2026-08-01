"""Model behaviour, against values worked out by hand on the tiny task.

Scored with :class:`Stability` over the three-plan ``planset``, whose pairwise
distances are ``2/3`` (plans 0-1), ``2/3`` (0-2) and ``6/7`` (1-2). See
``test_metrics.py`` for where those come from.
"""

import numpy as np
import pytest

from plandiversity.shortcuts import (
    MaxMean,
    MaxMin,
    MaxSum,
    Model,
    Stability,
    Uniqueness,
)

#: The pairwise Stability distances of the ``planset`` fixture.
DISTANCES = [2 / 3, 2 / 3, 6 / 7]


@pytest.fixture
def stability(task, planset):
    return Stability(task, planset)


class TestMaxSum:
    def test_it_totals_every_pairwise_distance(self, stability, planset):
        assert MaxSum([stability])(planset) == pytest.approx(sum(DISTANCES), abs=1e-5)

    def test_normalize_gives_the_mean_pairwise_distance(self, stability, planset):
        assert MaxSum([stability], normalize=True)(planset) == pytest.approx(
            sum(DISTANCES) / 3, abs=1e-5
        )


class TestMaxMean:
    def test_it_is_the_total_per_plan(self, stability, planset):
        assert MaxMean([stability])(planset) == pytest.approx(
            sum(DISTANCES) / 3, abs=1e-5
        )

    def test_normalize_gives_the_mean_pairwise_distance(self, stability, planset):
        assert MaxMean([stability], normalize=True)(planset) == pytest.approx(
            sum(DISTANCES) / len(DISTANCES), abs=1e-5
        )


class TestMaxMin:
    def test_it_is_the_closest_pair(self, stability, planset):
        assert MaxMin([stability])(planset) == pytest.approx(min(DISTANCES), abs=1e-5)

    def test_it_is_not_identically_zero(self, stability, planset):
        """The score used to collapse to 0.0 for every input, because the model
        padded its list of pairwise distances with a trailing zero."""
        assert MaxMin([stability])(planset) > 0.0

    def test_one_duplicated_plan_drives_the_score_to_zero(
        self, task, planset, plan_l1_then_l2
    ):
        """MaxMin's whole point: it is unmoved by how diverse the rest is."""
        with_duplicate = planset + [plan_l1_then_l2]
        metric = Stability(task, with_duplicate)
        assert MaxMin([metric])(with_duplicate) == 0.0
        assert MaxSum([metric])(with_duplicate) > MaxSum([metric])(planset)

    def test_normalize_has_no_effect(self, stability, planset):
        assert MaxMin([stability], normalize=True)(planset) == MaxMin([stability])(
            planset
        )


class TestModelOrdering:
    """Every model must rank a more diverse set above a less diverse one."""

    @pytest.mark.parametrize("model_class", [MaxSum, MaxMean, MaxMin])
    def test_identical_plans_score_zero(self, task, plan_l1_then_l2, model_class):
        plans = [plan_l1_then_l2, plan_l1_then_l2, plan_l1_then_l2]
        assert model_class([Stability(task, plans)])(plans) == 0.0

    @pytest.mark.parametrize("model_class", [MaxSum, MaxMean, MaxMin])
    def test_a_diverse_set_outscores_a_repetitive_one(
        self, task, planset, plan_l1_then_l2, model_class
    ):
        repetitive = [plan_l1_then_l2, plan_l1_then_l2, plan_l1_then_l2]
        assert model_class([Stability(task, planset)])(planset) > model_class(
            [Stability(task, repetitive)]
        )(repetitive)


class TestMultipleMetrics:
    def test_each_pair_is_the_mean_across_the_metrics(self, task, planset):
        model = MaxSum([Stability(task, planset), Uniqueness(task, planset)])
        # Uniqueness is 1.0 for all three pairs of this set.
        expected = sum((distance + 1.0) / 2 for distance in DISTANCES)
        assert model(planset) == pytest.approx(expected, abs=1e-5)

    def test_repeating_a_metric_does_not_change_the_score(self, task, planset):
        metric = Stability(task, planset)
        assert MaxSum([metric, metric])(planset) == MaxSum([metric])(planset)

    def test_the_matrix_is_the_mean_of_the_metrics_matrices(self, task, planset):
        stability = Stability(task, planset)
        uniqueness = Uniqueness(task, planset)
        combined = MaxSum([stability, uniqueness]).distance_matrix(planset)
        expected = (stability.pairwise(planset) + uniqueness.pairwise(planset)) / 2
        assert np.allclose(combined, expected)


class TestModelBase:
    def test_a_bare_metric_is_accepted_as_shorthand(self, task, planset):
        metric = Stability(task, planset)
        assert MaxSum(metric)(planset) == MaxSum([metric])(planset)

    def test_no_metrics_is_rejected(self):
        with pytest.raises(ValueError, match="at least one metric"):
            MaxSum([])

    def test_model_is_abstract(self, task):
        with pytest.raises(TypeError):
            Model([Stability(task)])

    def test_a_metric_returning_the_wrong_shape_is_rejected(self, task, planset):
        class Broken(Stability):
            def pairwise(self, plans=None):
                return np.zeros((2, 2))

        with pytest.raises(ValueError, match="must return an"):
            MaxSum([Broken(task)])(planset)

    @pytest.mark.parametrize("model_class", [MaxSum, MaxMean, MaxMin])
    def test_scores_are_stable_across_repeated_calls(
        self, task, planset, model_class
    ):
        """The score used to drift, because the model accumulated its pair
        count into instance state instead of deriving it per call."""
        model = model_class([Stability(task, planset)])
        first = model(planset)
        assert model(planset) == first
        assert model(planset) == first

    @pytest.mark.parametrize("model_class", [MaxSum, MaxMean, MaxMin])
    def test_a_planset_is_scored_independently_of_the_ones_before_it(
        self, task, planset, plan_l1_then_l2, model_class
    ):
        model = model_class([Stability(task)])
        expected = model_class([Stability(task)])(planset)
        model([plan_l1_then_l2, plan_l1_then_l2])
        assert model(planset) == expected

    @pytest.mark.parametrize("model_class", [MaxSum, MaxMean, MaxMin])
    @pytest.mark.parametrize("size", [0, 1])
    def test_fewer_than_two_plans_have_no_pairs(
        self, task, plan_l1_then_l2, model_class, size
    ):
        plans = [plan_l1_then_l2] * size
        assert model_class([Stability(task)])(plans) == 0.0

    def test_a_planset_may_be_any_iterable(self, task, planset):
        model = MaxSum([Stability(task)])
        assert model(iter(planset)) == model(planset)

    def test_pairwise_distances_has_one_entry_per_pair(self, task, planset):
        distances = MaxSum([Stability(task)]).pairwise_distances(planset)
        assert distances.shape == (3,)
        assert sorted(distances) == pytest.approx(sorted(DISTANCES))

    def test_repr_names_the_model_and_its_metrics(self, task):
        assert repr(MaxSum([Stability(task)])) == (
            "MaxSum([Stability], normalize=False)"
        )

    def test_str_is_the_model_name(self, task):
        assert str(MaxMin([Stability(task)])) == "MaxMin"
