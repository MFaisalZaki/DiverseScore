from unified_planning.plans import SequentialPlan
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

    def __init__(self):
        """Initialize a states metric object."""
        super(States, self).__init__(name="States")
    
    def __call__(self, plana:tuple, planb:tuple):
        
        # Create plan states
        plana_states = self._getFluentsValues(plana[1])
        planb_states = self._getFluentsValues(planb[1])
        
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
        return len(a.intersection(b)) / len(a.union(b))
    
    def _getFluentsValues(self, stateslist):
        """Returns a list of states in fluents-values string format."""
        _state_vars = set()
        for var in stateslist[0]._values:
            value = stateslist[0].get_value(var)
            _state_vars.add(var)

        fluents = []
        for state in stateslist[1:]:
            _state_vars_values = set()
            for _state_var in _state_vars:
                value = state.get_value(_state_var)
                _state_vars_values.add("{}-{}".format(str(_state_var), str(value)))
            fluents.append(_state_vars_values)
        return fluents
