import argparse
import os

DESCRIPTION = """Diverse score calculator command line interface."""

def _is_valid_file(arg):
    """
    Checks whether input PDDL files exist and are validate
    """
    if not os.path.exists(arg):
        raise argparse.ArgumentTypeError('{} not found!'.format(arg))
    elif not os.path.splitext(arg)[1] == ".pddl":
        raise argparse.ArgumentTypeError('{} is not a valid PDDL file!'.format(arg))
    else:
        return arg

def create_parser():
    """
    Specifies valid arguments for Diverse score calculator command line interface.
    """

    parser = argparse.ArgumentParser(description = DESCRIPTION,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--domain', help='Path to PDDL domain file', type=_is_valid_file)
    parser.add_argument('--problem', metavar='problem.pddl', help='Path to PDDL problem file', type=_is_valid_file)

    parser.add_argument('--diversity-model', choices=['MaxSum', 'MaxMin', 'MaxMean'], default='MaxSum', help='Diversity model to be used.')
    parser.add_argument('--metric', choices=['States', 'Stability', 'Uniqueness'], default='All', help='Metric to be used.')
    parser.add_argument('--plansdir', help='Path to directory containing plans.', type=str)
    parser.add_argument('--normalize', action='store_true', help='Normalize the metric score.', default=False)
    
    
    parser.add_argument('--load-plans-cnt', help='Number of plans to load and compute the score from them. Note that it takes the first n plans inorder.', type=int)
    parser.add_argument('--planset-JSON', help='Path to JSON file containing plans, any JSON file is valid if it contains this key "plans" which is a list of plans. ', type=str)
    parser.add_argument('--ibm-diversescore', action='store_true', help='Use IBM DiverseScore instead of the python implementation.', default=False)

    parser.add_argument('--tmpbasedir', help='Path to directory to store the temporary files.', type=str, default='/tmp')

    parser.add_argument('--out', help='Path to output file.', type=str, default=None)


    return parser