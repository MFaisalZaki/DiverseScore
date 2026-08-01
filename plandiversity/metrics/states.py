"""States: how much the state sequences two plans pass through differ."""

from unified_planning.shortcuts import SequentialSimulator

from .base import InapplicablePlanError, Metric, jaccard_distance


class States(Metric):
    """Distance between the sequences of states two plans pass through.

    Each plan is executed from the task's initial state, giving ``n + 1``
    states for ``n`` actions. States at the same step are compared by Jaccard
    similarity over the propositions true in them, and the similarities are
    averaged over the length of the **longer** plan::

        distance(a, b) = 1 - (1 / k') * sum(J(a_i, b_i) for i in range(k))

    where ``k = min(|a|, |b|)`` and ``k' = max(|a|, |b|)``. Dividing by ``k'``
    rather than ``k`` is what charges a plan for the steps its shorter
    counterpart never took: a plan is not made similar to another merely by
    agreeing on a prefix.

    Only boolean fluents take part. Numeric fluents have no meaning in a set
    intersection, so two plans that differ only in a numeric fluent score
    ``0.0`` here.

    References
    ----------
    .. [1] T. A. Nguyen, M. Do, A. E. Gerevini, I. Serina, B. Srivastava, and
           S. Kambhampati, "Generating diverse plans to handle unknown and
           partially known user preferences," Artificial Intelligence,
           vol. 190, pp. 1-31, 2012.
    """

    name = "States"

    def __init__(self, task, plans=None):
        # Both of these are built before super().__init__, which eagerly
        # extracts features and so already needs them.
        #
        # One simulator, reused for every plan: building one per comparison
        # grounds the task again each time and dominated the runtime.
        self._simulator = SequentialSimulator(problem=task)
        # Every grounded boolean fluent expression of the task, enumerated once.
        # A UPState stores only the values its action changed and defers the
        # rest to its ancestors, so reading a state's own dict yields the
        # action's effects rather than the state; get_value is what resolves
        # the whole assignment.
        self._boolean_fluents = [
            fluent
            for fluent, value in task.initial_values.items()
            if value.is_bool_constant()
        ]
        super().__init__(task=task, plans=plans)

    def _feature(self, plan):
        """The propositions true at each step, one frozenset per state."""
        return [
            frozenset(
                fluent
                for fluent in self._boolean_fluents
                if state.get_value(fluent).is_true()
            )
            for state in self._simulate(plan)
        ]

    def _distance(self, feature_a, feature_b) -> float:
        longer = max(len(feature_a), len(feature_b))
        if longer == 0:
            return 0.0
        similarity = sum(
            1.0 - jaccard_distance(state_a, state_b)
            for state_a, state_b in zip(feature_a, feature_b)
        )
        return 1.0 - similarity / longer

    def _simulate(self, plan):
        """The states visited by ``plan``, starting with the initial state.

        A plan of ``n`` actions yields ``n + 1`` states.

        Raises
        ------
        InapplicablePlanError
            If some action's precondition does not hold when it is reached.
        """
        state = self._simulator.get_initial_state()
        states = [state]
        for step, action in enumerate(plan.actions):
            state = self._simulator.apply(state, action)
            if state is None:
                raise InapplicablePlanError(
                    f"action {action} at step {step} is not applicable in the "
                    f"state reached by the preceding actions, so States cannot "
                    f"score this plan"
                )
            states.append(state)
        return states
