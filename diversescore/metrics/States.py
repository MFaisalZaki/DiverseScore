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

    def __init__(self, task, plans):
        """Initialize a states metric object."""
        self._states_cache = {}
        self.state_jaccard = lambda a, b: len(set.intersection(a, b)) / len(set.union(a, b)) if len(set.union(a, b)) > 0 else 1.0
        super(States, self).__init__(name="States", task=task, plans=plans)

    def __call__(self, plana:tuple, planb:tuple):

        if (plana, planb) in self.cache or (planb, plana) in self.cache:
            return self.cache[(plana, planb)]
        
        plana_states_fluents = list()
        if not plana in self._states_cache:
            self._states_cache[plana] = [set(e[0] for e in filter(lambda f: f[1].is_true() and f[1].is_bool_constant(), state._values.items())) for state in self._simulate(plana)]
        plana_states_fluents = self._states_cache[plana]
        
        planb_states_fluents = list()
        if not planb in self._states_cache:
            self._states_cache[planb] = [set(e[0] for e in filter(lambda f: f[1].is_true() and f[1].is_bool_constant(), state._values.items())) for state in self._simulate(planb)]
        planb_states_fluents = self._states_cache[planb]
        
        # Compute the average states similarity scores.
        k = 0
        score = 0.0
        for plana_state, planb_state in zip(plana_states_fluents, planb_states_fluents):
            if len(set.union(plana_state, planb_state)) == 0: continue
            score += self.state_jaccard(plana_state, planb_state)
            k += 1
        k_prime = max(len(plana_states_fluents), len(planb_states_fluents))
        result = (score + k - k_prime) / k
        self.cache[(plana, planb)] = result
        self.cache[(planb, plana)] = result
        return result

    def _simulate(self, plan):
        """Returns a list of states for plan."""
        _states = []
        with SequentialSimulator(problem=self.task) as simulator:
            initial_state = simulator.get_initial_state()
            current_state = initial_state
            _states = [current_state]
            for action_instance in plan.actions:
                current_state = simulator.apply(current_state, action_instance)
                if current_state is None:
                    return []
                _states.append(current_state)
        return _states
