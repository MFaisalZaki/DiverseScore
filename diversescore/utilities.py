
import os
from collections import defaultdict

from unified_planning.plans import SequentialPlan, ActionInstance
from unified_planning.shortcuts import *

def loadPlansDir(directory):
    plans = []
    for filename in os.listdir(directory):
        planfile = os.path.join(directory, filename)
        with open(planfile, 'r') as f:
            plans.append(f.read().splitlines())
    return plans

def computePlanSetStatistics(planset):
    retstats = defaultdict(dict)
    plan_lengths = set()
    for p in planset:
        plan_lengths.add(len(p.plan.actions))
    retstats['plans-count']  = len(planset)
    retstats['plan-lengths'] = list(plan_lengths)
    return retstats

def constructSequentialPlan(plan, grounded_problem):
    """Creates a SequentialPlan for each plan in planset."""
    actionlist = []
    for action in plan:
        actionname = action.replace("(", "").replace(")", "").replace(" ", "_")
        for problemaction in grounded_problem.actions:
            if problemaction.name == actionname:
                actionlist.append(ActionInstance(problemaction))
                break
    return SequentialPlan(actionlist, grounded_problem.environment)

def simlatePlan(plan, grounded_problem):
    """Returns a list of states for plan."""
    _states = []
    with SequentialSimulator(problem=grounded_problem) as simulator:
        initial_state = simulator.get_initial_state()
        current_state = initial_state
        _states = [current_state]
        for action_instance in plan.actions:
            current_state = simulator.apply(current_state, action_instance)
            if current_state is None:
                return []
            _states.append(current_state)
    return _states
