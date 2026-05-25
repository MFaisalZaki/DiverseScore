import numpy as np

from .base import Metric


class Stability(Metric):
    """Stability metric class.

    This class implements the stability metric from [1].

    References
    ----------
    .. [1] T. A. Nguyen, M. Do, A. E. Gerevini, I. Serina,
           B. Srivastava, and S. Kambhampati, "Generating diverse plans to handle unknown and partially known user
           preferences," Artificial Intelligence, vol. 190, pp. 1-31, 2012.
    """

    def __init__(self, task, plans):
        super().__init__(name="Stability", task=task, plans=plans)
        # id-keyed action-set cache; see GoalPredicateOrdering for the rationale.
        self._plans_str_cache = {id(p): self._extract_actions(p) for p in self.plans}

    def __call__(self, plana, planb):
        a = self._plans_str_cache.get(id(plana))
        if a is None:
            a = self._extract_actions(plana)
        b = self._plans_str_cache.get(id(planb))
        if b is None:
            b = self._extract_actions(planb)
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def pairwise(self, plans=None):
        plans = self.plans if plans is None else list(plans)
        sets = []
        for p in plans:
            s = self._plans_str_cache.get(id(p))
            if s is None:
                s = self._extract_actions(p)
            sets.append(s)
        n = len(sets)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            si = sets[i]
            li = len(si)
            for j in range(i + 1, n):
                sj = sets[j]
                inter = len(si & sj)
                union = li + len(sj) - inter
                d = inter / union if union else 0.0
                D[i, j] = d
                D[j, i] = d
        return D

    @staticmethod
    def _extract_actions(plan):
        return set(str(a) for a in plan.actions)
