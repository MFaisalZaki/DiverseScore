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

    def __init__(self):
        """Initialize a stability metric object."""
        super(Stability, self).__init__(name="Stability")
    
    def __call__(self, plana:tuple, planb:tuple):

        if (plana[0], planb[0]) in self.cache or (planb[0], plana[0]) in self.cache:
            return self.cache[(plana[0], planb[0])]

        plana_actions = set([str(a) for a in plana[0].actions])
        planb_actions = set([str(a) for a in planb[0].actions])

        result = len(set.intersection(plana_actions, planb_actions)) / len(set.union(plana_actions, planb_actions))
        self.cache[(plana[0], planb[0])] = result
        self.cache[(planb[0], plana[0])] = result
        return result