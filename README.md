# DiverseScore
This Python package computes several diversity models using different distance metrics for plan sets, built on top of the [unified-planning](https://github.com/aiplan4eu/unified-planning) framework.

## Available Diversity Models
- [x] MaxSum: The sum of pairwise distance between plans in a provided set of plans.
- [x] MaxMean: Average sum of the pairwise distance between plans in a provided set of plans.
- [x] MaxMin: The minimum distance between a plans in a provided set of plans.

## Available Distance Functions
- [x] Stability: Jaccard measure between two plan's actions.
- [x] States: Average Jaccard measure between two plan's states.
- [x] Uniqueness: Set difference between two plan's actions.
- [ ] Causal-links: Jaccard measure between two plan's causal links.

# How to use
## Installation
```
python -m pip install git+https://github.com/MFaisalZaki/DiverseScore.git
```

## Python Import
Each metric and model in this package operates on `(plan, states)` tuples, where `plan` is a `SequentialPlan` and `states` is the list of states produced by simulating that plan. You no longer need to ground the problem — just parse it, simulate each plan, and pass the resulting tuples to a model.

```python
from unified_planning.shortcuts import *
from diversescore.shortcuts import *

domain   = <Path-to-domain-PDDL-file>
problem  = <Path-to-problem-PDDL-file>
plansdir = <Path-to-plans-dir>

# Parse the domain and problem PDDL files
task = PDDLReader().parse_problem(domain, problem)

plansliststr = [] # <- container for the plans string.

# Compute a diversity score: MaxSum with the Stability metric
planset = list(map(lambda p: PDDLReader().parse_plan_string(task, p), plansliststr))
maxsum_stability_score = MaxSum([Stability()])(planset)
```

A model accepts a list of metrics, so you can combine several distance functions in a single score:

```python
score = MaxMean([Stability(), States(), Uniqueness()])(planset)
```

## Citations
This package implements the distance functions proposed by:
```
@article{nguyen2012generating,
  title={Generating diverse plans to handle unknown and partially known user preferences},
  author={Nguyen, Tuan Anh and Do, Minh and Gerevini, Alfonso Emilio and Serina, Ivan and Srivastava, Biplav and Kambhampati, Subbarao},
  journal={Artificial Intelligence},
  volume={190},
  pages={1--31},
  year={2012},
  publisher={Elsevier}
}
```

As for the diversity models, they are based on:
```
@article{parreno2021measuring,
  title={Measuring diversity. A review and an empirical analysis},
  author={Parre{\~n}o, Francisco and {\'A}lvarez-Vald{\'e}s, Ram{\'o}n and Mart{\'\i}, Rafael},
  journal={European Journal of Operational Research},
  volume={289},
  number={2},
  pages={515--532},
  year={2021},
  publisher={Elsevier}
}

@inproceedings{Sandoya2018DiversityAE,
  title={Diversity and Equity Models},
  author={Fernando Sandoya and Anna Mart{\'i}nez-Gavara and Ricardo Aceves and Abraham Duarte and Rafael Mart{\'i}},
  booktitle={Handbook of Heuristics},
  year={2018}
}
```
