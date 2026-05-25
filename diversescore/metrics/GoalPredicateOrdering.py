from collections import defaultdict

import numpy as np

from unified_planning.shortcuts import SequentialSimulator

from .base import Metric


class GoalPredicateOrdering(Metric):
    """Goal predicate ordering metric class.

    This class implements the goal predicate ordering metric from [1].

    References
    ----------
    .. [1] Abdelwahed, Mustafa F., et al.
          "Bridging the gap between structural and semantic similarity in diverse planning."
          The International Workshop on Human-Aware and Explainable Planning (2023).
    """

    def __init__(self, task, plans):
        from unified_planning.model.walkers.free_vars import FreeVarsExtractor
        super().__init__(name="GoalPredicateOrdering", task=task, plans=plans)

        extractor = FreeVarsExtractor()
        self.vars = [v for goal in task.goals for v in extractor.get(goal)]

        # Build the simulator once and reuse it across every plan simulation.
        # The library's `with SequentialSimulator(...)` per call was ~960s in profile.
        self._simulator = SequentialSimulator(problem=task)

        # Per-plan sequence cache, keyed by id(plan) to avoid SequentialPlan.__hash__/__eq__
        # (which trigger `is_semantically_equivalent` and dominated the runtime).
        self._seq_cache = {id(p): self._get_sequence(p) for p in self.plans}

    def __call__(self, plana, planb):
        sa = self._seq_cache.get(id(plana))
        if sa is None:
            sa = self._get_sequence(plana)
        sb = self._seq_cache.get(id(planb))
        if sb is None:
            sb = self._get_sequence(planb)
        L = len(sb)
        if L <= 1:
            return 0.0
        return sum(1 for x, y in zip(sa, sb) if x != y) / L

    def pairwise(self, plans=None):
        plans = self.plans if plans is None else list(plans)
        n = len(plans)
        seqs = []
        for p in plans:
            s = self._seq_cache.get(id(p))
            if s is None:
                s = self._get_sequence(p)
            seqs.append(s)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        L = len(seqs[0]) if seqs[0] is not None else 0
        if L <= 1:
            return np.zeros((n, n), dtype=np.float64)
        arr = np.asarray(seqs)
        return (arr[:, None, :] != arr[None, :, :]).sum(axis=2) / L

    def _simulate(self, plan):
        """Returns a list of states for plan, reusing self._simulator."""
        sim = self._simulator
        current_state = sim.get_initial_state()
        states = [current_state]
        for action_instance in plan.actions:
            current_state = sim.apply(current_state, action_instance)
            if current_state is None:
                return []
            states.append(current_state)
        return states

    def _get_sequence(self, plan):
        states = self._simulate(plan)
        hist = defaultdict(list)
        for state in states:
            for g in self.vars:
                hist[g].append(state.get_value(g).is_true())
        ordered = sorted(
            ((g, next((i for i, x in enumerate(hist[g]) if x), -1)) for g in self.vars),
            key=lambda e: e[1],
        )
        return [str(g) for g, _ in ordered]