
from .base import Metric

class Uniqueness(Metric):
    """Uniqueness metric class.

    This class implements the uniqueness metric from [1].

    References
    ----------
    .. [1] Roberts, M.; Howe, A.; and Ray, I. 2014. Evaluating diversity 
           in classical planning. In Proceedings of the International 
           Conference on Automated Planning and Scheduling,
           volume 24, 253–261.
    """

    def __init__(self, grounded_task):
        """Initialize a uniqueness metric object."""
        super(Uniqueness, self).__init__(grounded_task.problem, name="Uniqueness")
    
    def __call__(self, plana, planb):

        plana = self.constructSequentialPlan(plana)
        planb = self.constructSequentialPlan(planb)
        
        if len(plana.actions) < len(planb.actions):
            return self(planb, plana)

        plana_actions = set([a.action.name for a in plana.actions])
        planb_actions = set([a.action.name for a in planb.actions])

        for action in plana_actions:
            if not action in planb_actions:
                return 0.0
        return 1.0