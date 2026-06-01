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
        return 1.0 if sa != sb else 0.0

    def pairwise(self, plans=None):
        return self.pairwise_hamming(plans=plans, normalize=False)
        # return self.pairwise_kendall(plans=plans, normalize=False)
        # return self.pairwise_similarity(plans=plans)

    def pairwise_similarity(self, plans=None):
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
        arr = np.asarray(seqs)
        # 1 if two plans differ in goal-predicate ordering, 0 if identical.
        return (arr[:, None, :] != arr[None, :, :]).any(axis=2).astype(np.float64)

    def pairwise_hamming(self, plans=None, normalize=True):
        """Return an (n, n) np.ndarray of pairwise Hamming distances.

        Unlike :meth:`pairwise` (which is all-or-nothing: 1.0 if the goal-predicate
        orderings differ in any position), this counts how many ranks differ between
        the two orderings. With ``normalize=True`` (default) the count is divided by
        the sequence length, giving a distance in [0, 1] consistent with the other
        metrics; with ``normalize=False`` the raw number of differing positions is
        returned.
        """
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
        arr = np.asarray(seqs)
        # Count of positions where the two orderings differ.
        D = (arr[:, None, :] != arr[None, :, :]).sum(axis=2).astype(np.float64)
        if normalize:
            length = arr.shape[1]
            if length:
                D /= length
        return D

    def pairwise_kendall(self, plans=None, normalize=True):
        """Return an (n, n) np.ndarray of pairwise Kendall tau distances.

        Each plan's goal-predicate ordering is a permutation of the same goal
        predicates. The Kendall tau distance counts the number of *discordant
        pairs* -- pairs of goal predicates (g, h) that the two plans achieve in
        the opposite relative order. Unlike Hamming (:meth:`pairwise`), this is
        invariant to a uniform shift of the ordering and reflects how the
        *relative* achievement order differs.

        With ``normalize=True`` (default) the count is divided by the number of
        pairs ``m*(m-1)/2``, giving a distance in [0, 1]; with
        ``normalize=False`` the raw number of discordant pairs is returned.
        """
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

        # Canonical goal-predicate ordering shared by every plan's sequence.
        items = list(dict.fromkeys(seqs[0]))
        m = len(items)
        if m < 2:
            return np.zeros((n, n), dtype=np.float64)
        index = {name: k for k, name in enumerate(items)}

        # For each plan, the position of each canonical predicate in its ordering.
        positions = np.empty((n, m), dtype=np.int64)
        for r, seq in enumerate(seqs):
            for slot, name in enumerate(seq):
                positions[r, index[name]] = slot

        # Encode every plan as the sign of all i<j pair comparisons: True if
        # predicate i is achieved before predicate j. Two plans' Kendall tau
        # distance is the number of pairs whose sign disagrees.
        I, J = np.triu_indices(m, k=1)
        signs = positions[:, I] < positions[:, J]  # (n, P) bool, P = m*(m-1)/2
        D = (signs[:, None, :] != signs[None, :, :]).sum(axis=2).astype(np.float64)
        if normalize:
            D /= signs.shape[1]
        return D

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