
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

    def __init__(self, grounded_task):
        """Initialize a stability metric object."""
        super(Stability, self).__init__(grounded_task.problem, name="Stability")
    
    def __call__(self, plana, planb):
        
        plana = self.constructSequentialPlan(plana)
        planb = self.constructSequentialPlan(planb)

        if len(plana.actions) < len(planb.actions):
            return self(planb, plana)
    
        plana_actions = set([a.action.name for a in plana.actions])
        planb_actions = set([a.action.name for a in planb.actions])

        return 1.0 - len(plana_actions.intersection(planb_actions)) / len(plana_actions.union(planb_actions))