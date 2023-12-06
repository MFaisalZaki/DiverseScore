
import os
from collections import defaultdict

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
