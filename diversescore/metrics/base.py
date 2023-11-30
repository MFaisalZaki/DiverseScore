from unified_planning.plans import SequentialPlan, ActionInstance
from unified_planning.shortcuts import *

class Metric:
    def __init__(self, grounded_problem, name) -> None:
        self.name = name
        self.grounded_problem = grounded_problem
        
    def __call__(self, x, y):
        raise NotImplementedError
    
    def constructSequentialPlan(self, plan):
        """Creates a SequentialPlan for each plan in planset."""
        actionlist = []
        for action in plan:
            actionname = action.replace("(", "").replace(")", "").replace(" ", "_")
            for problemaction in self.grounded_problem.actions:
                if problemaction.name == actionname:
                    actionlist.append(ActionInstance(problemaction))
                    break
        return SequentialPlan(actionlist, self.grounded_problem.environment)
    
    def simlatePlan(self, plan):
        """Returns a list of states for plan."""
        _states = []
        with SequentialSimulator(problem=self.grounded_problem) as simulator:
            initial_state = simulator.get_initial_state()
            current_state = initial_state
            _states = [current_state]
            for action_instance in plan.actions:
                current_state = simulator.apply(current_state, action_instance)
                if current_state is None:
                    return []
                _states.append(current_state)
        return _states
    
    def getFluentsValues(self, stateslist):
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
