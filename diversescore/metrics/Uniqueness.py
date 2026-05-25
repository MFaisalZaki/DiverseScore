import numpy as np

from .base import Metric


class Uniqueness(Metric):
    """Uniqueness metric class.

    This class implements the uniqueness metric from [1].

    References
    ----------
    .. [1] Roberts, M.; Howe, A.; and Ray, I. 2014. Evaluating diversity
           in classical planning. In Proceedings of the International
           Conference on Automated Planning and Scheduling,
           volume 24, 253-261.
    """

    def __init__(self, task, plans):
        super().__init__(name="Uniqueness", task=task, plans=plans)
        # id-keyed cache: each entry is (frozenset_of_action_strs, len(set)).
        self._action_cache = {id(p): self._extract_actions(p) for p in self.plans}

    def __call__(self, plana, planb):
        a = self._action_cache.get(id(plana))
        if a is None:
            a = self._extract_actions(plana)
        b = self._action_cache.get(id(planb))
        if b is None:
            b = self._extract_actions(planb)
        return 1.0 if a == b else 0.0

    def pairwise(self, plans=None):
        plans = self.plans if plans is None else list(plans)
        n = len(plans)
        sets = []
        for p in plans:
            s = self._action_cache.get(id(p))
            if s is None:
                s = self._extract_actions(p)
            sets.append(s)
        # Group by identical action-set; entries in the same group score 1.0.
        groups = {}
        for idx, s in enumerate(sets):
            groups.setdefault(s, []).append(idx)
        D = np.zeros((n, n), dtype=np.float64)
        for members in groups.values():
            if len(members) < 2:
                continue
            arr = np.array(members)
            D[np.ix_(arr, arr)] = 1.0
        np.fill_diagonal(D, 0.0)
        return D

    @staticmethod
    def _extract_actions(plan):
        return frozenset(str(a) for a in plan.actions)
