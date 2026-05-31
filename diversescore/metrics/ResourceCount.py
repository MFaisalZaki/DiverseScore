import os
from collections import defaultdict

import numpy as np
from lark import Lark, Transformer, v_args

from .base import Metric


class ResourceCount(Metric):
    """Resource count metric class."""

    def __init__(self, task, plans, addinfo):
        super().__init__(name="ResourceCount", task=task, plans=plans)
        self.resource_list = set(e['name'] for e in parse_resource_file(addinfo).values())
        # id-keyed per-plan cache; see GoalPredicateOrdering for the rationale.
        self._ru_cache = {id(p): self._compute_resource_usage(p) for p in self.plans}

    def __call__(self, plana, planb):
        a = self._ru_cache.get(id(plana))
        if a is None:
            a = self._compute_resource_usage(plana)
        b = self._ru_cache.get(id(planb))
        if b is None:
            b = self._compute_resource_usage(planb)
        union = a | b
        if not union:
            return 0.0
        return 1.0 - len(a & b) / len(union)

    def pairwise(self, plans=None):
        plans = self.plans if plans is None else list(plans)
        usages = []
        for p in plans:
            v = self._ru_cache.get(id(p))
            if v is None:
                v = self._compute_resource_usage(p)
            usages.append(v)
        rlist = sorted(self.resource_list)
        mat = np.zeros((len(plans), len(rlist)), dtype=bool)
        for i, u in enumerate(usages):
            for j, r in enumerate(rlist):
                mat[i, j] = r in u
        inter = (mat[:, None, :] & mat[None, :, :]).sum(axis=2)
        union = (mat[:, None, :] | mat[None, :, :]).sum(axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = 1.0 - inter / union
        dist[union == 0] = 0.0
        return dist

    def _compute_resource_usage(self, plan):
        rlist = self.resource_list
        used = set()
        for action in plan.actions:
            for p in action.actual_parameters:
                s = str(p)
                if s in rlist:
                    used.add(s)
        return used
    

class ResourceUtilisation(Metric):
    """Resource utilisation metric class."""

    def __init__(self, task, plans, addinfo):
        super().__init__(name="ResourceUtilisation", task=task, plans=plans)
        self.resource_list = set(e['name'] for e in parse_resource_file(addinfo).values())
        # id-keyed per-plan cache; see GoalPredicateOrdering for the rationale.
        self._ru_cache = {id(p): self._compute_resource_usage(p) for p in self.plans}

    def __call__(self, plana, planb):
        a = self._ru_cache.get(id(plana))
        if a is None:
            a = self._compute_resource_usage(plana)
        b = self._ru_cache.get(id(planb))
        if b is None:
            b = self._compute_resource_usage(planb)
        union = a | b
        if not union:
            return 0.0
        return 1.0 - len(a & b) / len(union)

    def pairwise(self, plans=None):
        plans = self.plans if plans is None else list(plans)
        usages = []
        for p in plans:
            v = self._ru_cache.get(id(p))
            if v is None:
                v = self._compute_resource_usage(p)
            usages.append(v)
        rlist = sorted(self.resource_list)
        mat = np.zeros((len(plans), len(rlist)), dtype=bool)
        for i, u in enumerate(usages):
            for j, r in enumerate(rlist):
                mat[i, j] = r in u
        inter = (mat[:, None, :] & mat[None, :, :]).sum(axis=2)
        union = (mat[:, None, :] | mat[None, :, :]).sum(axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = inter == union
        dist[union == 0] = 0.0
        return dist

    def _compute_resource_usage(self, plan):
        rlist = self.resource_list
        used = set()
        for action in plan.actions:
            for p in action.actual_parameters:
                s = str(p)
                if s in rlist:
                    used.add(s)
        return used


class ResourceTransformer(Transformer):
    def resource_line(self, token):
        return {
            'name': token[0].value,
            'min':  int(token[1].value),
            'max':  int(token[2].value),
            'delta': int(token[3].value),
        }


def parse_resource_file(inputfile):
    if inputfile is None:
        return {}

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
            return Lark(grammar, parser='lalr', transformer=v_args(inline=True))

        with open(resource_input, 'r') as f:
            resource_input = f.readlines()
        resource_input = ''.join(resource_input)
        parser = construct_parser()
        tree = parser.parse(resource_input)
        transformer = ResourceTransformer()
        resources = transformer.transform(tree)
        return resources.children

    addition_informaion = defaultdict(dict)
    assert os.path.exists(inputfile), f'The resources file {inputfile} does not exist.'
    for resource in read_resource_file(inputfile):
        addition_informaion[resource['name']] = resource
    return addition_informaion