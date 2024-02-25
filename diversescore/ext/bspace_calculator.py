import json
from collections import defaultdict

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *

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
    with Compiler(problem_kind=planningtask.kind, compilation_kind=CompilationKind.GROUNDING) as grounder:
        grounded_problem = grounder.compile(planningtask, compilation_kind=CompilationKind.GROUNDING).problem
    planningtask_groundedproblem = grounded_problem

    formulalength = max(set([len(plan) for plan in planset]))
    
    planset = [constructSequentialPlan(planningtask_groundedproblem, plan) for plan in planset]

    _bspace = BehaviourSpace(planningtask_groundedproblem, configurations)
    _, _, _ = _bspace.encode(planset[0], formulalength)
    _bspace.extend(planset[1:], skip_sat_test=True)

    retscores = defaultdict(dict)
    retscores['behaviours-count'] = _bspace.behavioursCountScore()
    retscores['dimensions-count'] = _bspace.dimensionsCountScore()
    return retscores