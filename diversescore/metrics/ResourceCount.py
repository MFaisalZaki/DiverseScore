import os

from collections import defaultdict
from itertools import chain, combinations
from lark import Lark, Transformer, v_args

from .base import Metric

class ResourceCount(Metric):
    """Resource count metric class.

    This class implements the resource count metric from [1].

    References
    ----------
    """

    def __init__(self, task, plans, addinfo):
        """Initialize a resource count metric object."""
        self._ru_cache = {}
        super(ResourceCount, self).__init__(name="ResourceCount", task=task, plans=plans)
        self.resource_list = set(e['name'] for e in parse_resource_file(addinfo).values())

    def __call__(self, plana:tuple, planb:tuple):

        if (plana, planb) in self.cache or (planb, plana) in self.cache:
            return self.cache[(plana, planb)]
        
        plana_ru_cnt = list()
        if not plana in self._ru_cache:
            self._ru_cache[plana] = self._compute_resource_usage(plana)
        plana_ru_cnt = self._ru_cache[plana]
        
        planb_ru_cnt = list()
        if not planb in self._ru_cache:
            self._ru_cache[planb] = self._compute_resource_usage(planb)
        planb_ru_cnt = self._ru_cache[planb]

        return abs(plana_ru_cnt - planb_ru_cnt)
        
    def _compute_resource_usage(self, plan):
        resource_usage = {o: 0 for o in self.resource_list}
        for action in plan.actions:
            for used_resource in set.intersection(set(map(str, action.actual_parameters)), set(self.resource_list)):
                resource_usage[used_resource] += 1
        return len(list(filter(lambda e: e[1] > 0, resource_usage.items())))


class ResourceTransformer(Transformer):
    def resource_line(self, token):
        return {
            'name': token[0].value,
            'min':  int(token[1].value),
            'max':  int(token[2].value),
            'delta': int(token[3].value)
        }

def parse_resource_file(inputfile):
    if inputfile is None: return {}
    def read_resource_file(resource_input):
        def construct_parser():
            grammar = r'''
                start: resource_line+
                resource_line: "(:resource" (NAME | NAME_WITH_PARENTHESIS) MIN MAX DELTA ")"
                NAME: /[a-zA-Z_][\w-]*/
                NAME_WITH_PARENTHESIS: /[a-zA-Z_]\w*\([^)]*\)/
                MIN: /[0-9]+/
                MAX: /[0-9]+/
                DELTA: /[0-9]+/
                %ignore /\s+/
            '''
            parser = Lark(grammar, parser='lalr', transformer=v_args(inline=True))
            return parser
        # readlines in reource_input
        with open(resource_input, 'r') as f:
            resource_input = f.readlines()
        resource_input = ''.join(resource_input)
        parser = construct_parser()
        tree = parser.parse(resource_input)
        transformer = ResourceTransformer()
        resources = transformer.transform(tree)
        return resources.children

    addition_informaion = defaultdict(dict)
    if inputfile:
        assert os.path.exists(inputfile), f'The resources file {inputfile} does not exist.'
        for resource in read_resource_file(inputfile):
            addition_informaion[resource['name']] = resource
    return addition_informaion
