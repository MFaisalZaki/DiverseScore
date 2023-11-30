from unified_planning.shortcuts import *

from .base import Metric

class States(Metric):
    """States metric class.

    This class implements the states metric from [1].

    References
    ----------
    .. [1] T. A. Nguyen, M. Do, A. E. Gerevini, I. Serina,
           B. Srivastava, and S. Kambhampati, “Generating diverse plans to handle unknown and partially known user 
           preferences,” Artificial Intelligence, vol. 190, pp. 1–31, 2012.
    """

    def __init__(self, grounded_task):
        """Initialize a states metric object."""
        super(States, self).__init__(grounded_task.problem, name="States")
    
    def __call__(self, plana, planb):
        # Construct plans as SequentialPlan objects
        plana = self.constructSequentialPlan(plana)
        planb = self.constructSequentialPlan(planb)

        # Create plan states
        plana_states = self.getFluentsValues(self.simlatePlan(plana))
        planb_states = self.getFluentsValues(self.simlatePlan(planb))
        
        # Compute the average states similarity scores.
        k = 0
        score = 0.0
        for plana_state, planb_state in zip(plana_states, planb_states):
            score += self._delta(plana_state, planb_state)
            k += 1
        k_prime = max(len(plana_states), len(planb_states))
        return (score + k - k_prime) / k
    
    def _delta(self, state_a, state_b):
        a = set(state_a)
        b = set(state_b)
        return 1 - len(a.intersection(b)) / len(a.union(b))