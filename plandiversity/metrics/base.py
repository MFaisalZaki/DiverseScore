"""The :class:`Metric` base class and the distance convention every metric obeys."""

from abc import ABC, abstractmethod

import numpy as np


class InapplicablePlanError(ValueError):
    """Raised when a plan cannot be executed from the task's initial state.

    Metrics that simulate plans (:class:`~plandiversity.metrics.states.States`,
    :class:`~plandiversity.metrics.goal_predicate_ordering.GoalPredicateOrdering`)
    have nothing meaningful to compare when an action's precondition does not
    hold, so they refuse rather than score the plan against a truncated or empty
    state sequence.
    """


class Metric(ABC):
    """Pairwise **distance** between two plans.

    Every metric returns a distance in ``[0.0, 1.0]``:

    * ``0.0`` -- the two plans are indistinguishable under this metric,
    * ``1.0`` -- they are as different as this metric can express.

    That direction is the whole contract. Models sum, average and minimise
    these numbers directly and never re-invert them, so a metric that returns a
    *similarity* silently reverses the meaning of every score built on it.

    Subclasses implement two methods:

    ``_feature(plan)``
        Reduce a plan to whatever this metric actually compares -- a set of
        action names, a list of state fluent sets, a goal ordering.

    ``_distance(fa, fb)``
        Compare two such features.

    Splitting the two lets the base class extract each plan's feature exactly
    once per call instead of once per pair, which matters because feature
    extraction (plan simulation) dominates the runtime of the metrics that need
    it.

    Parameters
    ----------
    task:
        The planning problem the plans were produced for.
    plans:
        Optional. Plans whose features are computed eagerly and cached for the
        lifetime of the metric. Pass the plan set here when you intend to score
        it more than once; leave it out and features are computed per call.
    """

    name = "Metric"

    def __init__(self, task, plans=None):
        self.task = task
        self.plans = [] if plans is None else list(plans)
        # id(plan) -> (plan, feature). The plan is stored alongside its feature
        # for two reasons: it keeps the plan alive, so CPython cannot recycle
        # its id() onto an unrelated object, and it lets _feature_of confirm
        # identity before trusting a hit. SequentialPlan is deliberately not
        # used as a dict key -- its __hash__/__eq__ go through
        # is_semantically_equivalent, which is orders of magnitude slower than
        # the comparison being cached.
        self._cache = {id(plan): (plan, self._feature(plan)) for plan in self.plans}

    @abstractmethod
    def _feature(self, plan):
        """Reduce ``plan`` to the representation this metric compares."""

    @abstractmethod
    def _distance(self, feature_a, feature_b) -> float:
        """Distance in ``[0.0, 1.0]`` between two features from :meth:`_feature`."""

    def __call__(self, plan_a, plan_b) -> float:
        """Distance between two plans."""
        return self._distance(self._feature_of(plan_a), self._feature_of(plan_b))

    def pairwise(self, plans=None) -> np.ndarray:
        """Return the symmetric ``(n, n)`` distance matrix, zero on the diagonal.

        ``pairwise(plans)[i][j]`` is required to equal ``metric(plans[i],
        plans[j])``. Subclasses may override this with a vectorised
        implementation, but not with a different distance.
        """
        features = self._features_of(plans)
        n = len(features)
        distances = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = distances[j, i] = self._distance(
                    features[i], features[j]
                )
        return distances

    def _feature_of(self, plan):
        """The feature of ``plan``, from the cache when it was cached."""
        entry = self._cache.get(id(plan))
        if entry is not None and entry[0] is plan:
            return entry[1]
        return self._feature(plan)

    def _features_of(self, plans):
        """Features of ``plans``, defaulting to the plans given at construction."""
        plans = self.plans if plans is None else plans
        return [self._feature_of(plan) for plan in plans]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def jaccard_distance(a: set, b: set) -> float:
    """``1 - |a & b| / |a | b|``, with two empty sets counted as identical."""
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - len(a & b) / union


def group_codes(features) -> np.ndarray:
    """Map hashable features to integer codes, equal features sharing a code.

    Lets an exact-match metric build its distance matrix from one integer
    comparison per pair instead of one set comparison per pair.
    """
    codes = {}
    return np.fromiter(
        (codes.setdefault(feature, len(codes)) for feature in features),
        dtype=np.int64,
        count=len(features),
    )
