
import os

def readPlansDir(directory):
    plans = []
    for filename in os.listdir(directory):
        planfile = os.path.join(directory, filename)
        with open(planfile, 'r') as f:
            plans.append(f.read().splitlines())
    return plans
