
import os
from collections import defaultdict
import tempfile
import json

from unified_planning.plans import SequentialPlan, ActionInstance
from unified_planning.shortcuts import *

def loadPlansDir(directory):
    plans = []
    for filename in os.listdir(directory):
        planfile = os.path.join(directory, filename)
        with open(planfile, 'r') as f:
            plans.append(f.read().splitlines())
    return plans

def loadPlansJSON(jsonfile):
    with open(jsonfile, 'r') as f:
        plans = json.load(f)
    return plans['plans']

def computePlanSetStatistics(planset):
    retstats = defaultdict(dict)
    plan_lengths = set()
    for p in planset:
        plan_lengths.add(len(p.plan.actions))
    retstats['plans-count']  = len(planset)
    retstats['plan-lengths'] = list(plan_lengths)
    return retstats

def createActionDictFromTask(grounded_problem):
    actiondict = {}
    for action in grounded_problem.actions:
        actiondict[action.name] = ActionInstance(action)
    return actiondict

def constructSequentialPlanFromActionDict(plan, actiondict, grounded_problem):
    """Creates a SequentialPlan for each plan in planset."""
    actionlist = []
    for action in plan:
        if action.startswith(";"): break #this means that we reached the end of plan.
        actionname = action.replace("(", "").replace(" )", "").replace(")", "").replace(" ", "_")
        assert actionname in actiondict, f"Action {actionname} not found in actiondict."
        actionlist.append(actiondict[actionname])
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

def dumpPlans(planset, k, dumpdir):
    for i, plan in enumerate(planset[:k]):
        with open(os.path.join(dumpdir, f'sas_plan.{i+1}'), 'w') as f:
            for action in plan:
                if action.startswith(";"):
                    f.write(f'{action}')
                f.write(f'{action}\n' if action.startswith("(") and action.endswith(")") else f'({action})\n')

def dumpScores(scores, dumpdir, filename):
    with open(os.path.join(dumpdir, f'{filename}.json'), 'w') as f:
        json.dump(scores, f, indent=4)