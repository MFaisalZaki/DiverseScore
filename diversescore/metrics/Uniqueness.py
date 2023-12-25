from unified_planning.plans import SequentialPlan
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

    def __init__(self):
        """Initialize a uniqueness metric object."""
        super(Uniqueness, self).__init__(name="Uniqueness")
    
    def __call__(self, plana:tuple, planb:tuple):
        if len(plana[0].actions) < len(planb[0].actions):
            return self(planb, plana)

        plana_actions = set([a.action.name for a in plana[0].actions])
        planb_actions = set([a.action.name for a in planb[0].actions])

        for action in plana_actions:
            if not action in planb_actions:
                return 0.0
        return 1.0