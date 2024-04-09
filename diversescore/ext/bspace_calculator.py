import json
from collections import defaultdict

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *

from diversescore.utilities import createActionDictFromTask, constructSequentialPlanFromActionDict
from bspace.shortcuts import *

def BSpaceCalculator(domain, problem, planset, task, bspace_cfg, **kwargs):
    bspacecfg = json.load(open(bspace_cfg, 'r'))
    configurations = defaultdict(dict)
    configurations['balance-behaviours'] = eval(bspacecfg['balance-behaviours'])
    configurations['dims'] = []
    
    if eval(bspacecfg.get('include-objs-count-dim', 'False')) and task['resource-details'] is not None:
        configurations['dims'].append(ResourceCount(parsed_resources=task['resource-details']['objects']))

    if eval(bspacecfg.get('include-plan-length-dim', 'False')):
        configurations['dims'].append(PlanLength())

    if eval(bspacecfg.get('include-goal-ordering-dim', 'False')):
        configurations['dims'].append(GoalOrderingPredicates())
    
    if eval(bspacecfg.get('include-objs-utils-dim', 'False')) and task['resource-details'] is not None:
        if len(task['resource-details']['objects']) > 0:
            configurations['dims'].append(Objects(parsed_resources=task['resource-details']['objects']))
    
    if eval(bspacecfg.get('include-fns-utils-dim', 'False')) and task['resource-details'] is not None:
        if len(task['resource-details']['fns']) > 0:
            configurations['dims'].append(Functions(parsed_resources=task['resource-details']['fns']))

    planningtask = PDDLReader().parse_problem(domain, problem)
    _bspace = BehaviourSpace(planningtask, configurations)
    formulalength = max(set([len(plan)-1 for plan in planset]))

    # to reduce the construction time for the planset from txt to SequentialPlan,
    # we will use a dict the maps the action name to 

    actiondictmap = createActionDictFromTask(_bspace.encoder.encoder.ground_problem)
    planset = [constructSequentialPlanFromActionDict(plan, actiondictmap, _bspace.encoder.encoder.ground_problem) for plan in planset]
    
    _, _, _ = _bspace.encode(planset[0], formulalength)
    _bspace.extend(planset[1:], skip_sat_test=True)

    retscores = defaultdict(dict)
    retscores['behaviours-count'] = _bspace.behavioursCountScore()
    retscores['dimensions-count'] = _bspace.dimensionsCountScore()
    return retscores