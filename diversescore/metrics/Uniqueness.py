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

    def __init__(self, task, plans):
        """Initialize a uniqueness metric object."""
        super(Uniqueness, self).__init__(name="Uniqueness", task=task, plans=plans)

    def __call__(self, plana:tuple, planb:tuple):
        if (plana, planb) in self.cache or (planb, plana) in self.cache:
            return self.cache[(plana, planb)]

        plana_actions = set([str(a) for a in plana.actions])
        planb_actions = set([str(a) for a in planb.actions])

        difference = len(plana_actions - planb_actions) + len(planb_actions - plana_actions)
        result = 1.0 if difference == 0 else 0.0
        self.cache[(plana, planb)] = result
        self.cache[(planb, plana)] = result
        return result