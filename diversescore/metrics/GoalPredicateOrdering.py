from collections import defaultdict
from unified_planning.plans import SequentialPlan
from unified_planning.shortcuts import SequentialSimulator

from .base import Metric

class GoalPredicateOrdering(Metric):
    """Goal predicate ordering metric class.

    This class implements the goal predicate ordering metric from [1].

    References
    ----------
    .. [1] Abdelwahed, Mustafa F., et al. 
          "Bridging the gap between structural and semantic similarity in diverse planning." 
          The International Workshop on Human-Aware and Explainable Planning (2023).
    """

    def __init__(self, task, plans):
        from unified_planning.model.walkers.free_vars import FreeVarsExtractor
        vars = list(map(lambda expr: FreeVarsExtractor().get(expr), task.goals))
        self.vars = [elem for s in vars for elem in s]
        self._seq_cache = {}
        """Initialize a goal predicate ordering metric object."""
        super(GoalPredicateOrdering, self).__init__(name="GoalPredicateOrdering", task=task, plans=plans)

    def __call__(self, plana:tuple, planb:tuple):

        if (plana, planb) in self.cache or (planb, plana) in self.cache:
            return self.cache[(plana, planb)]
        
        plana_states_fluents = list()
        if not plana in self._seq_cache:
            self._seq_cache[plana] = self._get_sequence(plana)
        plana_states_fluents = self._seq_cache[plana]
        
        planb_states_fluents = list()
        if not planb in self._seq_cache:
            self._seq_cache[planb] = self._get_sequence(planb)
        planb_states_fluents = self._seq_cache[planb]
        
        hamming_distance = [x == y for x,y in zip(plana_states_fluents, planb_states_fluents)].count(False)
        result = hamming_distance/len(planb_states_fluents) if len(planb_states_fluents) > 1 else 0.0

        self.cache[(plana, planb)] = result
        self.cache[(planb, plana)] = result
        return result
    
    def _get_sequence(self, plan):
        _states = self._simulate(plan)
        _time_step_history = defaultdict(list)
        for t, state in enumerate(_states):
            for g in self.vars:
                _time_step_history[g].append(state.get_value(g).is_true())
        return list(map(lambda e: str(e[0]), sorted([(g, next((i for i, x in enumerate(_time_step_history[g]) if x), -1)) for g in self.vars], key=lambda e:e[1])))

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
