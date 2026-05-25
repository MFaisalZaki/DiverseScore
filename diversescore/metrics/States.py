import numpy as np

from unified_planning.shortcuts import SequentialSimulator

from .base import Metric


class States(Metric):
    """States metric class.

    This class implements the states metric from [1].

    References
    ----------
    .. [1] T. A. Nguyen, M. Do, A. E. Gerevini, I. Serina,
           B. Srivastava, and S. Kambhampati, "Generating diverse plans to handle unknown and partially known user
           preferences," Artificial Intelligence, vol. 190, pp. 1-31, 2012.
    """

    def __init__(self, task, plans):
        super().__init__(name="States", task=task, plans=plans)
        # Build the simulator once and reuse across all plans.
        self._simulator = SequentialSimulator(problem=task)
        # id-keyed per-plan state-fluent-sets cache.
        self._states_cache = {id(p): self._extract_state_fluents(p) for p in self.plans}

    def __call__(self, plana, planb):
        a = self._states_cache.get(id(plana))
        if a is None:
            a = self._extract_state_fluents(plana)
        b = self._states_cache.get(id(planb))
        if b is None:
            b = self._extract_state_fluents(planb)
        return self._compare(a, b)

    def pairwise(self, plans=None):
        plans = self.plans if plans is None else list(plans)
        feats = []
        for p in plans:
            f = self._states_cache.get(id(p))
            if f is None:
                f = self._extract_state_fluents(p)
            feats.append(f)
        n = len(feats)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            ai = feats[i]
            for j in range(i + 1, n):
                d = self._compare(ai, feats[j])
                D[i, j] = d
                D[j, i] = d
        return D

    @staticmethod
    def _compare(a_states, b_states):
        score = 0.0
        k = 0
        for sa, sb in zip(a_states, b_states):
            union = sa | sb
            if not union:
                continue
            inter = len(sa & sb)
            score += inter / len(union)
            k += 1
        if k == 0:
            return 0.0
        k_prime = max(len(a_states), len(b_states))
        return (score + k - k_prime) / k

    def _extract_state_fluents(self, plan):
        return [
            set(e[0] for e in state._values.items()
                if e[1].is_true() and e[1].is_bool_constant())
            for state in self._simulate(plan)
        ]

    def _simulate(self, plan):
        sim = self._simulator
        current = sim.get_initial_state()
        states = [current]
        for ai in plan.actions:
            current = sim.apply(current, ai)
            if current is None:
                return []
            states.append(current)
        return states
