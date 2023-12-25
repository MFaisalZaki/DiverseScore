
import sys
import os

from unified_planning.shortcuts import *
from diversescore.shortcuts import *


from . import arguments

def main(args=None):
    """
    Main planning routine
    """
    if args is None:
        args = sys.argv[1:]

    # Parse planner args
    parser = arguments.create_parser()
    args = parser.parse_args(args)

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
    print(f'The {args.diversity_model} score using {args.metric} metric is {(model(plans))}')

if __name__ == '__main__':
    main(sys.argv[1:])