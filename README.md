# DiverseScore
This Python package computes several diversity models using different distance metrics for plan sets in SAS+ format. 
Also it wraps [IBM's diversescore](https://github.com/IBM/diversescore.git).

## Available Diversity Models
- [x] MaxSum: The sum of pairwise distance between plans in a provided set of plans.
- [x] MaxMean: Average sum of the pairwise distance between plans in a provided set of plans.
- [x] MaxMin: The minimum distance between a plans in a provided set of plans.

## Available Distance Functions
- [x] Stability: Jaccard measure between two plan's actions.
- [x] States: Average jaccard measure between two plan's states.
- [x] Uniqueness: Set difference between two plan's actions.
- [ ] Causal-links: Jaccard measure between two plan's casual links.

# How to use
## Installation
```
python -m pip install git+
```

## CLI
DiverseScore has a cli, which can be invoked as follows:
```
diversescorecli [-h] [-domain DOMAIN] [-problem problem.pddl] [-diversity-model {MaxSum,MaxMin,MaxMean}] [-metric {States,Stability,Uniqueness}] [-plansdir PLANSDIR]
```
## Python Import
DiverseScore can be integrated into any Python project, here is a simple code to compute the MaxSum score using Stability Metric.
```
import os

from unified_planning.shortcuts import *
from diversescore.shortcuts import *

domain=<Path-to-domain-PDDL-file>
problem=<Path-to-problem-PDDL-file>
plansdir=<Path-to-plans-dir>
# Read the domain and problem PDDL files
task = PDDLReader().parse_problem(domain, problem)
# Ground the problem
with Compiler(problem_kind=task.kind, compilation_kind=CompilationKind.GROUNDING) as grounder:
    groudned_task = grounder.compile(task, compilation_kind=CompilationKind.GROUNDING)
# Load the plans from directory
plans = loadPlansDir(plansdir)
maxsum_stability_score = MaxSum(Stability(groudned_task))
```

## Citations
This package implements the distance fucntions proposed by:
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

As for the diversity models are based on:
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

If you are using IBM's score computation you should cite:
```
@InProceedings{katz-sohrabi-aaai2020,
  title =        "Reshaping diverse planning",
  author =       "Michael Katz and Shirin Sohrabi",
  booktitle =    "Proceedings of the Thirty-Fourth {AAAI} Conference on
                  Artificial Intelligence ({AAAI} 2020)",
  publisher =    "{AAAI} Press",
  pages =        "9892--9899",
  year =         "2020"
}

@InProceedings{katz-et-al-aaai2022,
  title =        "Bounding Quality in Diverse Planning",
  author =       "Michael Katz and Shirin Sohrabi and Octavian Udrea",
  booktitle =    "Proceedings of the Thirty-Sixth {AAAI} Conference on
                  Artificial Intelligence ({AAAI} 2022)",
  publisher =    "{AAAI} Press",
  year =         "2022"
}
```
