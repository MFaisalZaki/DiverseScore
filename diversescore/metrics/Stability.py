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
        super(Stability, self).__init__(name="Stability", task=task, plans=plans)

    def __call__(self, plana:tuple, planb:tuple):

        if (plana, planb) in self.cache or (planb, plana) in self.cache:
            return self.cache[(plana, planb)]

        plana_actions = set([str(a) for a in plana.actions])
        planb_actions = set([str(a) for a in planb.actions])

        result = len(set.intersection(plana_actions, planb_actions)) / len(set.union(plana_actions, planb_actions))
        self.cache[(plana, planb)] = result
        self.cache[(planb, plana)] = result
        return result