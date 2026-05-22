from unified_planning.plans import SequentialPlan

from .base import Metric

class Stability(Metric):
    """Stability metric class.

    This class implements the stability metric from [1].

    References
    ----------
    .. [1] T. A. Nguyen, M. Do, A. E. Gerevini, I. Serina,
           B. Srivastava, and S. Kambhampati, “Generating diverse plans to handle unknown and partially known user 
           preferences,” Artificial Intelligence, vol. 190, pp. 1–31, 2012.
    """

    def __init__(self, task, plans):
        """Initialize a stability metric object."""
        self._plans_str_cache = {}
        super(Stability, self).__init__(name="Stability", task=task, plans=plans)

    def __call__(self, plana:tuple, planb:tuple):

        if (plana, planb) in self.cache or (planb, plana) in self.cache:
            return self.cache[(plana, planb)]

        if not plana in self._plans_str_cache:
            self._plans_str_cache[plana] = set(str(a) for a in plana.actions)
        plana_actions = self._plans_str_cache[plana]
        
        if not planb in self._plans_str_cache:
            self._plans_str_cache[planb] = set(str(a) for a in planb.actions)
        planb_actions = self._plans_str_cache[planb]

        result = len(set.intersection(plana_actions, planb_actions)) / len(set.union(plana_actions, planb_actions))
        self.cache[(plana, planb)] = result
        self.cache[(planb, plana)] = result
        return result