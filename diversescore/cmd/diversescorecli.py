
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
    plans = []
    for filename in os.listdir(args.plansdir):
        planfile = os.path.join(args.plansdir, filename)
        with open(planfile, 'r') as f:
            plans.append(f.read().splitlines())

    _diversity_model = {
        'MaxSum': MaxSum,
        'MaxMin': MaxMin,
        'MaxMean': MaxMean,
        'NormalizedMaxSum': NormalizedMaxSum
    }

    _metric = {
        'States': States,
        'Stability': Stability,
        'Uniqueness': Uniqueness
    }

    metric = _metric[args.metric](task)
    model  = _diversity_model[args.diversity_model](task, metric)
    print(model(plans))

if __name__ == '__main__':
    main(sys.argv[1:])