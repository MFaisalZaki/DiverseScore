
import sys
import json

from unified_planning.shortcuts import *
from diversescore.cmd.arguments import *

from diversescore.shortcuts import *

def IBMCalculatorWrapper(args):
    print("Calculating diversity score using IBM DiverseScore.")
    metricslist = []
    if not (args.metric.lower() in ['states', 'stability', 'uniqueness']) and args.metric == 'All':
        metricslist = ['states', 'stability', 'uniqueness', 'states-uniqueness', 'states-stability', 'stability-uniqueness', 'states-stability-uniqueness']
    else :
        metricslist = [args.metric.lower()]
    
    if args.planset_JSON:
        planset = loadPlansJSON(args.planset_JSON)        
    elif args.plansdir:
        planset = loadPlansDir(args.plansdir)
    else:
        raise Exception("No planset provided, you can provide a json file containing the planset or a directory containing the plans.")
    assert args.load_plans_cnt, "You must provide the number of plans to load and compute the score from them."
    return IBMCalculator(args.domain, args.problem, planset, args.load_plans_cnt, metricslist, dir=args.tmpbasedir)

def UPWrapper(args):
    print("Calculating diversity score using python implementation.")
    task = PDDLReader().parse_problem(args.domain, args.problem)

    with Compiler(problem_kind=task.kind, compilation_kind=CompilationKind.GROUNDING) as grounder:
        groudned_task = grounder.compile(task, compilation_kind=CompilationKind.GROUNDING)

    # Read plans from directory
    plans = loadPlansDir(args.plansdir)
    plans = [(constructSequentialPlan(plan, groudned_task), simlatePlan(plan, groudned_task)) for plan in plans]
    
    _diversity_model = eval(args.diversity_model)
    _metric = eval(args.metric)
    
    metric = _metric()
    model  = _diversity_model([metric], args.normalize)
    return {'diversityscore': {str(metric): model(plans)}}

def main(args=None):
    """
    Main planning routine
    """
    if args is None:
        args = sys.argv[1:]

    # Parse planner args
    parser = create_parser()
    args = parser.parse_args(args)

    diversityscore = None

    if args.ibm_diversescore:
        diversityscore = IBMCalculatorWrapper(args)
    else:
        diversityscore = UPWrapper(args)

    if args.out:
        plansetjson = {'diversityscore': diversityscore}
        if args.planset_JSON:
            plansetjson.update(json.load(open(args.planset_JSON, 'r')))
        dumpScores(plansetjson, args.tmpbasedir, args.out)
    else:
        for metric, score in diversityscore.items():
            print(f'The {metric} score is {score}')
    
if __name__ == '__main__':
    main(sys.argv[1:])
